
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <format>
#include <limits>
#include <mutex>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "ansi.h"
#include "arcfour.h"
#include "atomic-util.h"
#include "base/logging.h"
#include "base/stringprintf.h"
#include "big-polyhedra.h"
#include "bignum/big.h"
#include "bounds.h"
#include "color-util.h"
#include "geom/tree-3d.h"
#include "hashing.h"
#include "hull3d.h"
#include "image.h"
#include "periodically.h"
#include "polyhedra.h"
#include "process-util.h"
#include "randutil.h"
#include "status-bar.h"
#include "threadutil.h"
#include "timer.h"
#include "util.h"
#include "vector-util.h"
#include "yocto_matht.h"
#include "z3.h"

using vec3 = yocto::vec<double, 3>;

inline vec3 QuaternionToSpherePoint(const quat4 &q) {
  // The z-column of the rotation matrix represents the rotated Z-axis.
  // return normalize(rotation_frame(q).z);
  return transform_point(rotation_frame(normalize(q)), vec3{0, 0, 1});
}

inline bool AllZero(const BigVec3 &v) {
  return BigRat::IsZero(v.x) && BigRat::IsZero(v.y) && BigRat::IsZero(v.z);
}

struct Boundaries {
  // 1 bit means dot product is positive, 0 means negative.
  uint64_t GetCode(const BigVec3 &v) const {
    uint64_t code = 0;
    for (int i = 0; i < planes.size(); i++) {
      const BigVec3 &normal = planes[i];
      BigRat d = dot(v, normal);
      int sign = BigRat::Sign(d);
      CHECK(sign != 0) << "Points exactly on the boundary are not "
        "handled.";
      if (sign > 0) {
        code |= uint64_t{1} << i;
      }
    }
    return code;
  }

  explicit Boundaries(const BigPoly &poly) : poly(poly) {
    // Now, the boundaries are planes parallel to faces that pass
    // through the origin. First we find all of these planes
    // and give them ids. These planes need an orientation, too,
    // so a normal vector is a good representation. We can't make
    // this unit length, however.
    //
    // We could actually use integer vectors here! Scale by
    // multiplying by all the denominators, then divide by the GCD.
    // This representation is canonical up to sign flips.

    auto AlreadyHave = [&](const BigVec3 &n) {
        for (const BigVec3 &m : planes) {
          if (AllZero(cross(n, m))) {
            return true;
          }
        }
        return false;
      };

    for (const std::vector<int> &face : poly.faces->v) {
      CHECK(face.size() >= 3);
      const BigVec3 &a = poly.vertices[face[0]];
      const BigVec3 &b = poly.vertices[face[1]];
      const BigVec3 &c = poly.vertices[face[2]];
      BigVec3 normal = ScaleToMakeIntegral(cross(c - a, b - a));
      printf("Normal: %s\n", VecString(normal).c_str());
      if (!AlreadyHave(normal)) {
        planes.push_back(normal);
      }
    }

    printf("There are %d distinct planes.\n",
           (int)planes.size());

    // You can switch to a larger word size for more complex
    // polyhedra.
    CHECK(planes.size() <= 64);
  }

  size_t Size() const { return planes.size(); }

  std::vector<BigVec3> planes;
  BigPoly poly;
};

static frame3 FrameFromViewPos(const vec3 &v) {
  return frame_fromz({0, 0, 0}, v);
}

// Visualize a patch under its parameterization. This is
// using doubles and samples.
static void PlotPatch(const Boundaries &boundaries,
                      const BigVec3 &bigv) {
  Timer timer;
  uint64_t code = boundaries.GetCode(bigv);
  std::string code_string = std::format("{:b}", code);
  Polyhedron small_poly = SmallPoly(boundaries.poly);

  const vec3 v = normalize(SmallVec(bigv));

  const std::vector<int> hull = [&]() {
      frame3 frame = FrameFromViewPos(v);
      Mesh2D shadow = Shadow(Rotate(small_poly, frame));
      return QuickHull(shadow.vertices);
    }();

  // Generate vectors that have the same code.
  const int NUM_SAMPLES = 1000;
  ArcFour rc(std::format("plot_{}", code));
  StatusBar status(1);

  int64_t hits = 0, attempts = 0;
  Periodically status_per(5.0);
  std::vector<vec3> samples;
  samples.reserve(NUM_SAMPLES);
  while (samples.size() < NUM_SAMPLES) {
    attempts++;
    vec3 s;
    std::tie(s.x, s.y, s.z) = RandomUnit3D(&rc);
    BigVec3 bs(BigRat::ApproxDouble(s.x, 1000000),
               BigRat::ApproxDouble(s.y, 1000000),
               BigRat::ApproxDouble(s.z, 1000000));

    // Try all the signs. At most one of these will be
    // in the patch, but this should increase the
    // efficiency (because knowing that other signs
    // are *not* in the patch increases the chance
    // that we are).
    for (uint8_t b = 0b000; b < 0b1000; b++) {
      BigVec3 bbs((b & 0b100) ? -bs.x : bs.x,
                  (b & 0b010) ? -bs.y : bs.y,
                  (b & 0b001) ? -bs.z : bs.z);
      uint64_t sample_code = boundaries.GetCode(bbs);
      if (sample_code == code) {
        // Sample is in range.
        samples.push_back(
            vec3((b & 0b100) ? -s.x : s.x,
                 (b & 0b010) ? -s.y : s.y,
                 (b & 0b001) ? -s.z : s.z));
        hits++;
        break;
      }
    }

    status_per.RunIf([&]() {
        status.Progressf(samples.size(), NUM_SAMPLES,
                         ACYAN("%s") " (%.3f%% eff)",
                         code_string.c_str(),
                         (hits * 100.0) / attempts);
      });
  }

  // separate bounds for each parameter. Just using X dimension.
  Bounds rbounds;
  Bounds gbounds;
  Bounds bbounds;

  Bounds bounds;
  // First pass to compute bounds.
  for (const vec3 &s : samples) {
    rbounds.BoundX(s.x);
    gbounds.BoundX(s.y);
    bbounds.BoundX(s.z);

    frame3 frame = FrameFromViewPos(s);
    for (int vidx : hull) {
      const vec3 &vin = small_poly.vertices[vidx];
      vec3 vout = transform_point(frame, vin);
      bounds.Bound(vout.x, vout.y);
    }
  }
  bounds.AddMarginFrac(0.05);

  ImageRGBA img(3840, 2160);
  img.Clear32(0x000000FF);

  Bounds::Scaler scaler = bounds.ScaleToFit(img.Width(), img.Height()).FlipY();

  const double rspan = rbounds.MaxX() - rbounds.MinX();
  const double gspan = gbounds.MaxX() - gbounds.MinX();
  const double bspan = bbounds.MaxX() - bbounds.MinX();
  auto Color = [&](const vec3 &s) {
      float r = (s.x - rbounds.MinX()) / rspan;
      float g = (s.y - gbounds.MinX()) / gspan;
      float b = (s.z - bbounds.MinX()) / bspan;
      return ColorUtil::FloatsTo32(r * 0.9 + 0.1,
                                   g * 0.9 + 0.1,
                                   b * 0.9 + 0.1,
                                   1.0);
    };

  std::vector<vec2> starts;

  for (const vec3 &s : samples) {
    frame3 frame = FrameFromViewPos(s);
    uint32_t color = Color(s);

    auto GetOut = [&](int hidx) {
        const vec3 &vin = small_poly.vertices[hull[hidx]];
        return transform_point(frame, vin);
      };

    for (int hullidx = 0; hullidx < hull.size(); hullidx++) {
      vec3 vout1 = GetOut(hullidx);
      vec3 vout2 = GetOut((hullidx + 1) % hull.size());

      const auto &[x1, y1] = scaler.Scale(vout1.x, vout1.y);
      const auto &[x2, y2] = scaler.Scale(vout2.x, vout2.y);

      img.BlendLine32(x1, y1, x2, y2, color & 0xFFFFFF80);
    }

    vec3 start = GetOut(0);
    const auto &[x, y] = scaler.Scale(start.x, start.y);
    starts.emplace_back(x, y);
  }

  CHECK(samples.size() == starts.size());
  for (int i = 0; i < samples.size(); i++) {
    uint32_t color = Color(samples[i]);
    const vec2 &start = starts[i];
    img.BlendCircle32(start.x, start.y, 4, color);
  }

  img.BlendText2x32(8, 16, 0xFFFF77AA,
                    std::format("{:032b}", code));
  // Draw scale.
  double oneu = scaler.ScaleX(1.0);
  {
    int Y = 32;
    int x0 = 8;
    int x1 = 8 + oneu;
    img.BlendLine32(x0, Y, x1, Y, 0xFFFF77AA);
    img.BlendLine32(x0, Y - 4, x0, Y - 1, 0xFFFF77AA);
    img.BlendLine32(x1, Y - 4, x1, Y - 1, 0xFFFF77AA);
  }

  std::string filename = std::format("patch-{:b}.png", code);
  img.Save(filename);
  printf("Wrote %s in %s\n", filename.c_str(),
         ANSI::Time(timer.Seconds()).c_str());
}

static vec3 RandomPointOnSphere(ArcFour *rc) {
  const quat4 small_quat = normalize(RandomQuaternion(rc));
  return QuaternionToSpherePoint(small_quat);
}

static void BigSnubHulls() {
  BigPoly scube = BigScube(50);
  Boundaries boundaries(scube);

  Polyhedron cube = SnubCube();

  Periodically status_per(1.0);
  constexpr int64_t ITERS = 100'000;
  StatusBar status(1);

  std::mutex m;
  int64_t next_work_idx = 0;
  constexpr int NUM_THREADS = 8;

  // Parallel.
  std::vector<BigVec3> samples;
  std::vector<uint64_t> codes;

  std::unordered_map<uint64_t, BigVec3> examples;

  auto RandomCoord = [](ArcFour *rc) -> BigRat {
      // Between 3 and 10 or -10 and -3.
      // uint64_t n = 300000000 + RandTo(rc, 700000000);
      uint64_t n = RandTo(rc, 2000000000);
      //                       100000000
      return (rc->Byte() & 1) ?
        BigRat(n, 100000000) :
        BigRat(-n, 100000000);
    };

  auto RandomVec = [&](ArcFour *rc) {
      for (;;) {
        BigRat x = RandomCoord(rc);
        BigRat y = RandomCoord(rc);
        BigRat z = RandomCoord(rc);

        if (x * x + y * y + z * z >
            BigRat(9))
          return BigVec3(x, y, z);
      }
    };

  ParallelFan(
      NUM_THREADS,
      [&](int thread_idx) {
        ArcFour rc(std::format("snubhulls.{}.{}", thread_idx,
                               time(nullptr)));
        for (;;) {
          int64_t work_idx = 0;
          {
            MutexLock ml(&m);
            if (next_work_idx == ITERS)
              return;
            work_idx = next_work_idx;
            next_work_idx++;
          }

          BigVec3 sample_point = RandomVec(&rc);

          CHECK(!AllZero(sample_point));

          uint64_t code = boundaries.GetCode(sample_point);

          {
            MutexLock ml(&m);
            CHECK(codes.size() == samples.size());
            codes.push_back(code);
            samples.push_back(sample_point);
            if (!examples.contains(code)) {
              examples[code] = sample_point;
            }
          }
          status_per.RunIf([&]{
              status.Progressf(work_idx, ITERS,
                               "%lld/%lld",
                               work_idx, ITERS);
            });
        }
      });

  for (const BigVec3 &v : samples) {
    CHECK(!AllZero(v));
  }

  CHECK(codes.size() == samples.size());

  ArcFour rc("color");
  std::unordered_map<uint64_t, uint32_t> colored_codes;
  for (uint64_t code : codes) {
    if (!colored_codes.contains(code)) {
      colored_codes[code] =
        ColorUtil::HSVAToRGBA32(
            RandDouble(&rc),
            0.5 + RandDouble(&rc) * 0.5,
            0.5 + RandDouble(&rc) * 0.5,
            1.0);
    }
  }

  printf("\n\n\n");

  printf("There are %lld distinct codes in this sample.\n",
         (int64_t)colored_codes.size());

  for (const auto &[code, color] : colored_codes) {
    printf("%s▉" ANSI_RESET ": %s\n",
           ANSI::ForegroundRGB32(color).c_str(),
           std::format("{:b}", code).c_str());

    CHECK(examples.contains(code));
    const BigVec3 &ex = examples[code];
    PlotPatch(boundaries, ex);
  }

  // As point cloud.
  if (true) {
    std::string outply =
      std::format(
          "ply\n"
          "format ascii 1.0\n"
          "element vertex {}\n"
          "property float x\n"
          "property float y\n"
          "property float z\n"
          "property uchar red\n"
          "property uchar green\n"
          "property uchar blue\n"
          "end_header\n", samples.size());

    for (size_t i = 0; i < samples.size(); i++) {
      CHECK(i < samples.size());
      const vec3 v = normalize(SmallVec(samples[i])) * 100.0;
      uint32_t color = colored_codes[codes[i]];

      const auto &[r, g, b, _] = ColorUtil::Unpack32(color);
      AppendFormat(&outply,
                   "{} {} {} {} {} {}\n",
                   v.x, v.y, v.z,
                   r, g, b);
    }

    std::string filename = "bigsnubcloud.ply";
    Util::WriteFile(filename, outply);
    printf("Wrote %lld bytes to %s.\n",
           outply.size(),
           filename.c_str());
  }
}

// Find the set of patches (as their codes) that are non-empty, by
// shelling out to z3. This could be optimized a lot, but the set is a
// fixed property of the snub cube (given the ordering of vertices and
// faces), so we just need to enumerate them once.
struct PatchEnumerator {
  PatchEnumerator() : scube(BigScube(50)), boundaries(scube), status(1) {
    status.Statusf("Setup.");
    // Find patches that are non-empty.
    // Naively there are 2^31 of them, but the vast majority
    // are completely empty. Z3 is a good way to prove this.
    std::string out;

    int num_bits = boundaries.Size();
    for (int b = 0; b < num_bits; b++) {
      // true = 1 = postive dot product
      bits.emplace_back(NewBool(&out, std::format("bit{}", b)));
    }

    // The hypothesized point. If unsatisfiable, then the patch
    // is empty.
    Z3Vec3 v = NewVec3(&out, "pt");

    // Constrain v based on the bits.
    for (int b = 0; b < num_bits; b++) {
      Z3Vec3 normal{boundaries.planes[b]};
      AppendFormat(&out,
                   "(assert (ite {} (> {} 0.0) (< {} 0.0)))\n",
                   bits[b].s,
                   Dot(v, normal).s,
                   Dot(v, normal).s);
    }
    setup = std::move(out);
    status.Statusf("Setup done.");
  }

  BigPoly scube;
  Boundaries boundaries;
  std::vector<Z3Bool> bits;
  std::string setup;
  int64_t z3calls = 0;
  StatusBar status;

  std::vector<uint64_t> nonempty_patches;

  std::string PartialCodeString(int depth, uint64_t code) {
    if (depth == 0) return AGREY("(empty)");
    std::string ret;
    for (int i = depth - 1; i >= 0; i--) {
      uint64_t bit = 1ULL << i;
      ret.append((code & bit) ? ACYAN("1") : ABLUE("0"));
    }
    return ret;
  }

  // Bits < depth have been assigned to the values in code.
  void EnumerateRec(int depth,
                    uint64_t code) {
    // Is it possible at all?
    std::string out = setup;

    for (int b = 0; b < depth; b++) {
      AppendFormat(&out, "(assert (= {} {}))\n",
                   bits[b].s,
                   (code & (1UL << b)) ? "true" : "false");
    }

    // Don't even need to get the model here.
    AppendFormat(&out, "(check-sat)\n");

    std::string filename = "z3-patches-tmp.z3";
    Util::WriteFile(filename, out);
    // satus.Printf("Wrote %s\n", filename.c_str());

    z3calls++;
    status.Statusf("Z3: %s", std::format("{:b}", code).c_str());
    std::optional<std::string> z3result =
      ProcessUtil::GetOutput(std::format("d:\\z3\\bin\\z3.exe {}", filename));
    status.Statusf("%lld Z3 calls. Depth %d\n", z3calls, depth);

    CHECK(z3result.has_value());
    if (z3result.value().find("unsat") != std::string::npos) {
      status.Printf("Code %s is impossible.\n",
                    PartialCodeString(depth, code).c_str());
      return;
    }

    if (depth == boundaries.Size()) {
      // Then we have a complete code.
      status.Printf(AGREEN("Nonempty") ": %s\n",
                    PartialCodeString(depth, code).c_str());
      nonempty_patches.push_back(code);
      return;
    }
    CHECK(depth < boundaries.Size());

    // Otherwise, we try extending with 0, and with 1.
    EnumerateRec(depth + 1, code);
    EnumerateRec(depth + 1, code | (1ULL << depth));
  }

  void Enumerate() {
    Timer timer;
    EnumerateRec(0, 0);
    std::string all_codes;
    for (uint64_t code : nonempty_patches) {
      AppendFormat(&all_codes, "{:b}\n", code);
    }
    std::string filename = "scube-nonempty-patches.txt";
    Util::WriteFile(filename, all_codes);
    status.Printf("Wrote %s in %s (%lld z3 calls)\n", filename.c_str(),
                  ANSI::Time(timer.Seconds()).c_str(),
                  z3calls);
  }
};

int main(int argc, char **argv) {
  ANSI::Init();

  PatchEnumerator pe;
  pe.Enumerate();

  return 0;
}
