
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <format>
#include <limits>
#include <mutex>
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
#include "patches.h"
#include "periodically.h"
#include "polyhedra.h"
#include "randutil.h"
#include "status-bar.h"
#include "threadutil.h"
#include "timer.h"
#include "util.h"
#include "vector-util.h"
#include "yocto_matht.h"

using vec3 = yocto::vec<double, 3>;

DECLARE_COUNTERS(too_close);

// TODO Normalizing a hull:
// Since the snub cube is vertex transitive, it does
// not matter where we start. The path traced out is
// ...

// TODO:
// - Extract a single patch and compute bounds on
//   the parameters (quaternion parameters?)
// -

struct Hulls {
  size_t Num() { return canonical.size(); }

  size_t GetHullId(const std::vector<int> &hull) {
    CHECK(hull.size() >= 3);

    // Canonicalize:
    //  - The hull should start with the smallest
    //    vertex, numerically. (Handled here)
    //  - We should not include colinear vertices.
    //    (TODO: handled elsewhere?)
    //  - With duplicate points, use the minimum
    //    index (TODO: handled elsewhere?)
    //  - Winding order: The hull could wind in
    //    either direction. Choose the one that
    //    puts the smaller vertex second.

    // It would be great if we modded out by symmetry,
    // too. This might require a more thoughtful
    // assignment of vertex indices?

    int besti = 0;
    for (int i = 1; i < hull.size(); i++) {
      if (hull[i] < hull[besti]) {
        besti = i;
      }
    }

    std::vector<int> hull_key;
    hull_key.reserve(hull.size());
    for (int i = 0; i < hull.size(); i++) {
      hull_key.push_back(hull[(besti + i) % hull.size()]);
    }

    if (hull_key[1] > hull_key.back()) {
      VectorReverse(&hull_key);
      // Because that put the first element last.
      VectorRotateRight(&hull_key, 1);
    }

    // Do we have it?
    auto it = indices.find(hull_key);
    if (it != indices.end()) {
      return it->second;
    } else {
      size_t id = canonical.size();
      indices[hull_key] = id;
      canonical.emplace_back(std::move(hull_key));
      return id;
    }
  }

  std::unordered_map<std::vector<int>, size_t,
                     Hashing<std::vector<int>>> indices;
  std::vector<std::vector<int>> canonical;
};

// Visualize a patch under its parameterization. This is
// using doubles and samples.
static void PlotPatch(const Boundaries &boundaries,
                      const BigVec3 &bigv) {
  Timer timer;
  uint64_t code = boundaries.GetCode(bigv);
  std::string code_string = std::format("{:b}", code);
  Polyhedron small_poly = SmallPoly(boundaries.big_poly);

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

#if 0
// Get the patch that contains v.
static void GeneratePatch(const Boundaries &boundaries,
                          const BigVec3 &v) {
  const BigPoly &poly = boundaries.poly;

  uint64_t code = boundaries.GetCode(v);




  // Here we use exact math. The poly approximates the snub cube
  // (but we still think it is an actual counterexample) and the quat
  // came from a double sample but now exact (it is NOT unit length,
  // however).

  // The patch is the contiguous region around the point p that
  // has the same hull topologically.

  // First, recompute that hull.
  BigFrame frame = NonUnitRotationFrame(pt);
  BigMesh2D shadow = Shadow(Rotate(frame, poly));
  std::vector<int> hull = BigQuickHull(shadow.vertices);

  // Now, the boundaries are planes parallel to faces that pass
  // through the origin.
}
#endif

// Note I may have some mixup between the rotation and its inverse.
// I deleted other uses of this function. Check carefully if you use
// this again.
inline vec3 QuaternionToSpherePoint(const quat4 &q) {
  // The z-column of the rotation matrix represents the rotated Z-axis.
  // PERF: You can skip computing most of this because of the zeroes.
  return transform_point(rotation_frame(normalize(q)), vec3{0, 0, 1});
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

  #if 0
  PlotPatch(boundaries, {BigRat(1), BigRat(1), BigRat(1)});
  PlotPatch(boundaries, {BigRat(1, 2), BigRat(2, 3), BigRat(3, 8)});
  PlotPatch(boundaries, {BigRat(111, 233), BigRat(2, 5), BigRat(11, 8)});
  #endif

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

// Visualize a sample of the distinct convex hulls.
// By distinct convex hull, we mean a specific set of
// points on the hull.
static void SnubHulls() {

  Polyhedron cube = SnubCube();

  Periodically status_per(1.0);
  Hulls hulls;
  constexpr int64_t ITERS = 100'000;
  StatusBar status(1);

  std::mutex m;
  int64_t next_work_idx = 0;
  constexpr int NUM_THREADS = 8;

  // Parallel.
  std::vector<vec3> samples;
  std::vector<size_t> ids;
  // An orientation from each region.
  std::unordered_map<size_t, quat4> examples;

  Tree3D<double, bool> tree;
  static constexpr double TOO_CLOSE = 1.0e-3;

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

          vec3 sphere_point = RandomPointOnSphere(&rc);

          if (AllZero(sphere_point))
            continue;

          {
            MutexLock ml(&m);
            vec3 sphere_point100 = sphere_point * 100.0;

            if (!tree.Empty()) {
              const auto &[pos_, value_, dist] =
                tree.Closest(sphere_point100.x,
                             sphere_point100.y,
                             sphere_point100.z);
              if (dist < TOO_CLOSE) {
                too_close++;
                continue;
              }
            }

            // Reserve it.
            tree.Insert(sphere_point100.x,
                        sphere_point100.y,
                        sphere_point100.z,
                        true);
          }

          frame3 frame =
            inverse(frame_fromz(vec3{0, 0, 0}, sphere_point));
          // frame3 frame = rotation_frame(small_quat);
          Mesh2D shadow = Shadow(Rotate(cube, frame));
          std::vector<int> hull = QuickHull(shadow.vertices);

          {
            MutexLock ml(&m);
            size_t id = hulls.GetHullId(hull);
            CHECK(ids.size() == samples.size());
            ids.push_back(id);
            samples.push_back(sphere_point);
            if (!examples.contains(id)) {
              const auto &[rot, trans] = UnpackFrame(frame);
              examples[id] = normalize(rot);
            }
          }
          status_per.RunIf([&]{
              status.Progressf(work_idx, ITERS,
                               "%lld/%lld  %lld" ARED("≈"),
                               work_idx, ITERS,
                               too_close.Read());
            });
        }
      });

  for (const vec3 &v : samples) {
    CHECK(!AllZero(v));
  }

  CHECK(ids.size() == samples.size());

  printf("There are %lld distinct hulls in this sample.\n",
         (int64_t)hulls.Num());

  ArcFour rc("color");
  std::vector<uint32_t> colors;
  colors.reserve(hulls.Num());
  if (hulls.Num() <= 4) {
    colors.push_back(0xFF0000FF);
    colors.push_back(0x00FF00FF);
    colors.push_back(0x0000FFFF);
    colors.push_back(0xFF00FFFF);
  } else {
    for (int i = 0; i < hulls.Num(); i++) {
      colors.push_back(ColorUtil::HSVAToRGBA32(
                           RandDouble(&rc),
                           0.5 + RandDouble(&rc) * 0.5,
                           0.5 + RandDouble(&rc) * 0.5,
                           1.0));
    }
  }

  if (colors.size() > hulls.Num()) colors.resize(hulls.Num());

  for (int i = 0; i < hulls.Num(); i++) {
    printf("%d. (%s▉" ANSI_RESET "):",
           i, ANSI::ForegroundRGB32(colors[i]).c_str());
    for (int vidx : hulls.canonical[i]) {
      CHECK(vidx >= 0 && vidx < cube.vertices.size());
      printf(" %d", vidx);
    }
    printf("\n");
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
      const vec3 v = samples[i] * 100.0;
      const int id = ids[i];
      CHECK(id < colors.size());
      uint32_t color32 = colors[id];

      const auto &[r, g, b, _] = ColorUtil::Unpack32(color32);
      AppendFormat(&outply,
                   "{} {} {} {} {} {}\n",
                   v.x, v.y, v.z,
                   r, g, b);
    }

    std::string filename = "snubcloud.ply";
    Util::WriteFile(filename, outply);
    printf("Wrote %lld bytes to %s.\n",
           outply.size(),
           filename.c_str());
  }
}


int main(int argc, char **argv) {
  ANSI::Init();

  BigSnubHulls();
  // SnubHulls();

  return 0;
}
