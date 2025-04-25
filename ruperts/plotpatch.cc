
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <format>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "ansi.h"
#include "arcfour.h"
#include "base/logging.h"
#include "base/stringprintf.h"
#include "big-polyhedra.h"
#include "bignum/big.h"
#include "bounds.h"
#include "color-util.h"
#include "image.h"
#include "patches.h"
#include "periodically.h"
#include "polyhedra.h"
#include "randutil.h"
#include "rendering.h"
#include "run-z3.h"
#include "status-bar.h"
#include "threadutil.h"
#include "timer.h"
#include "util.h"
#include "vector-util.h"
#include "yocto_matht.h"
#include "z3.h"

static constexpr int DIGITS = 24;

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

[[maybe_unused]]
static vec3 RandomPointOnSphere(ArcFour *rc) {
  const quat4 small_quat = normalize(RandomQuaternion(rc));
  return QuaternionToSpherePoint(small_quat);
}

[[maybe_unused]]
static void BigSnubHulls() {
  BigPoly scube = BigScube(DIGITS);
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

  [[maybe_unused]]
  auto RandomVec = [&](ArcFour *rc) {
      for (;;) {
        BigRat x = RandomCoord(rc);
        BigRat y = RandomCoord(rc);
        BigRat z = RandomCoord(rc);

        if (x * x + y * y + z * z > BigRat(9)) {
          return BigVec3(x, y, z);
        }
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

          // BigVec3 sample_point = RandomVec(&rc);
          BigQuat sample_quat = RandomBigQuaternion(&rc);
          CHECK(!AllZero(sample_quat));

          BigVec3 sample_point = ViewPosFromNonUnitQuat(sample_quat);
          CHECK(!AllZero(sample_point));

          uint64_t code = boundaries.GetCode(sample_quat);

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
    // const BigVec3 &ex = examples[code];
    // PlotPatch(boundaries, ex);
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

Z3Vec3 NewUnitVector(std::string *out, std::string_view name_hint) {
  #if 1
  Z3Vec3 v = NewVec3(out, name_hint);

  // v must be a unit vector.
  AppendFormat(out,
               "(assert (= 1.0 (+ (* {} {}) (* {} {}) (* {} {}))))\n",
               v.x.s, v.x.s,
               v.y.s, v.y.s,
               v.z.s, v.z.s);
  return v;
  #endif

  #if 0
  // "The Diophantine Equation x^2+y^2+z^2=m^2", Robert Spira, 1962.
  Z3Int u = NewInt(out, std::format("{}_u", name_hint));
  Z3Int v = NewInt(out, std::format("{}_v", name_hint));
  Z3Int w = NewInt(out, std::format("{}_w", name_hint));
  Z3Int t = NewInt(out, std::format("{}_t", name_hint));

  // They can be anything as long as they are not all zero.
  AppendFormat(out, "(assert (not "
               "(and (= {} 0) (= {} 0) (= {} 0) (= {} 0))))\n",
               u.s, v.s, w.s, t.s);

  Z3Int xnumer = Z3Int(2) * (u * w - v * t);
  Z3Int ynumer = Z3Int(2) * (u * t - v * w);
  Z3Int znumer = u * u + v* v - w * w - t * t;
  Z3Int denom = u * u + v * v + w * w + t * t;

  Z3Real x(std::format("(/ (to_real {}) (to_real {}))", xnumer.s, denom.s));
  Z3Real y(std::format("(/ (to_real {}) (to_real {}))", ynumer.s, denom.s));
  Z3Real z(std::format("(/ (to_real {}) (to_real {}))", znumer.s, denom.s));

  return Z3Vec3(x, y, z);
  #endif

  #if 0
  // Same, but use reals instead of ints.
  Z3Real u = NewReal(out, std::format("{}_u", name_hint));
  Z3Real v = NewReal(out, std::format("{}_v", name_hint));
  Z3Real w = NewReal(out, std::format("{}_w", name_hint));
  Z3Real t = NewReal(out, std::format("{}_t", name_hint));

  // They can be anything as long as they are not all zero.
  AppendFormat(out, "(assert (not "
               "(and (= {} 0) (= {} 0) (= {} 0) (= {} 0))))\n",
               u.s, v.s, w.s, t.s);

  Z3Real xnumer = Z3Real(2) * (u * w - v * t);
  Z3Real ynumer = Z3Real(2) * (u * t - v * w);
  Z3Real znumer = u * u + v * v - w * w - t * t;
  Z3Real denom = u * u + v * v + w * w + t * t;

  Z3Real x(std::format("(/ {} {})", xnumer.s, denom.s));
  Z3Real y(std::format("(/ {} {})", ynumer.s, denom.s));
  Z3Real z(std::format("(/ {} {})", znumer.s, denom.s));

  return Z3Vec3(x, y, z);
  #endif
}

void ConstrainToPatch(std::string *out,
                      const Z3Vec3 &view,
                      const Boundaries &boundaries,
                      uint64_t code,
                      uint64_t mask) {

  // Note: Trying closed spaces, which is also valid and may be faster
  // for z3.
  constexpr const char *LESS = "<=";
  constexpr const char *GREATER = ">=";

  // Constrain v based on the bits, but only for the ones in the
  // mask.
  for (int b = 0; b < boundaries.Size(); b++) {
    uint64_t pos = uint64_t{1} << b;
    if (mask & pos) {
      Z3Vec3 normal{boundaries.big_planes[b]};
      // If 1, then positive dot product.
      const char *order = (code & pos) ? GREATER : LESS;
      AppendFormat(out,
                   "(assert ({} {} 0.0))\n",
                   order,
                   Dot(view, normal).s);
    }
  }

}

// Make a view (quat) constrained to the given patch.
Z3Quat GetPatchQuat(std::string *out,
                    const Boundaries &boundaries,
                    uint64_t code,
                    // can just pass all 1 bits
                    uint64_t mask) {

  Z3Quat q = NewQuat(out, "q");

  // q must not be zero.
  AppendFormat(out,
               "(assert (not (and "
               "(= {} 0.0) "
               "(= {} 0.0) "
               "(= {} 0.0) "
               "(= {} 0.0)"
               ")))\n",
               q.x.s,
               q.y.s,
               q.z.s,
               q.w.s);

  // Insist the view point is in the patch.
  Z3Vec3 view = ViewPosFromNonUnitQuat(out, q);
  ConstrainToPatch(out, view, boundaries, code, mask);

  // Optional: The view position will be a unit vector.
  AppendFormat(out, "(assert (= 1.0 {}))\n",
               Sum({view.x * view.x,
                    view.y * view.y,
                    view.z * view.z}).s);

  return q;
}

// Make a view (unit vector) constrained to the given patch.
Z3Vec3 GetPatchView(std::string *out,
                    const Boundaries &boundaries,
                    uint64_t code,
                    uint64_t mask) {

  // The view point, which is in the patch. Unit length.
  Z3Vec3 view = NewUnitVector(out, "view");
  ConstrainToPatch(out, view, boundaries, code, mask);
  return view;
}

// Returns a real s such that s = 1 / sqrt(val)
// Requires val_sq > 0.
inline Z3Real EmitInvSqrt(std::string *out, const Z3Real &val,
                          std::string_view name_hint) {
  Z3Real s = NewReal(out, name_hint);
  // Maybe unnecessary, but these must be true.
  AppendFormat(out, "(assert (> {} 0.0))\n", val.s);
  AppendFormat(out, "(assert (> {} 0.0))\n", s.s);
  // 1.0 = s * s * val      (divide by val)
  // 1.0 / val = s * s      (sqrt both sides)
  // 1.0 / sqrt(val) = s
  AppendFormat(out,
               ";; {} = 1.0 / sqrt({})\n"
               "(assert (= 1.0 (* {} {} {})))\n",
               s.s, val.s,
               s.s, s.s, val.s);
  return s;
}

// q any non-zero quaternion.
std::vector<vec2> ReferenceShadow(const Polyhedron &poly,
                                  const std::vector<int> &hull,
                                  const BigQuat &q) {
  CHECK(!AllZero(q));

  frame3 frame = SmallFrame(NonUnitRotationFrame(q));

  std::vector<vec2> projected_hull;
  projected_hull.reserve(hull.size());

  for (int vidx : hull) {
    CHECK(vidx >= 0 && vidx < poly.vertices.size());
    const vec3 &v_in = poly.vertices[vidx];
    const vec3 v_out = transform_point(frame, v_in);
    projected_hull.emplace_back(vec2{v_out.x, v_out.y});
  }

  return projected_hull;
}

// Transform the hull points (indices into poly.vertices) using
// the given rotation. Return the transformed points as
// a vector of 2d points.
std::vector<Z3Vec2> EmitShadow(std::string *out,
                               const BigPoly &poly,
                               const std::vector<int> &hull,
                               const Z3Quat &view) {

  Z3Frame frame = NonUnitRotationFrame(out, view);

  std::vector<Z3Vec2> projected_hull;
  projected_hull.reserve(hull.size());

  for (size_t i = 0; i < hull.size(); ++i) {
    int vertex_index = hull[i];
    CHECK(vertex_index >= 0 && vertex_index < poly.vertices.size());

    Z3Vec3 v_in = DeclareVec3(out, poly.vertices[vertex_index],
                              std::format("p{}", i));
    Z3Vec3 v_out = TransformPoint(frame, v_in);
    Z3Vec2 v2_out = DeclareVec2(out, Z3Vec2(v_out.x, v_out.y),
                                std::format("v{}", i));

    projected_hull.push_back(v2_out);
  }

  return projected_hull;
}


Z3Real EmitConvexPolyArea(std::string *out,
                          const std::vector<Z3Vec2> &vertices) {
  CHECK(vertices.size() >= 3);

  AppendFormat(out, ";; Hull area\n");
  std::vector<Z3Real> cross_terms;
  cross_terms.reserve(vertices.size());

  for (size_t i = 0; i < vertices.size(); ++i) {
    const Z3Vec2 &v_i = vertices[i];
    const Z3Vec2 &v_next = vertices[(i + 1) % vertices.size()];

    // We're essentially creating a triangle fan with the origin
    // (saving the division by half for the end).
    cross_terms.push_back(Cross(v_i, v_next));
  }

  return NameReal(out, Sum(cross_terms) / Z3Real(2), "area");
}

struct RatBounds {
  RatBounds(BigRat initial_lb, BigRat initial_ub) :
    lb(std::move(initial_lb)),
    ub(std::move(initial_ub)) {}
  BigRat Midpoint() const {
    return (lb + ub) / BigRat(2);
  }

  std::string BriefString() const {
    double l = lb.ToDouble(), u = ub.ToDouble();
    return std::format("{:.17g} {} x {} {:.17g}",
                       l, lclosed ? "<=" : "<",
                       uclosed ? "<=" : "<", u);
  }

  BigRat lb, ub;
  // If closed, then the boundary is included.
  bool lclosed = true, uclosed = true;
};

void BoundArea(const Boundaries &boundaries,
               uint64_t code) {
  Timer timer;
  uint64_t mask = GetCodeMask(boundaries, code);
  // mask = mask | (mask << 1);
  // const uint64_t mask = ~0;
  // mask = ~0;

  printf("Using mask: %s\n",
         std::format("{:b}", mask).c_str());

  // Could just save the hulls with the codes!
  printf("Get quat in patch...\n");
  BigQuat example_q = GetBigQuatInPatch(boundaries, code);
  printf("Get view pos...\n");
  BigVec3 example_v = ViewPosFromNonUnitQuat(example_q);

  static constexpr bool RENDER_HULL = false;

  printf("Compute hull once...\n");
  const std::vector<int> hull = [&]() {
      // const vec3 v = normalize(SmallVec(example_v));
      Polyhedron small_poly = SmallPoly(boundaries.big_poly);

      BigFrame big_frame = NonUnitRotationFrame(example_q);
      frame3 frame = SmallFrame(big_frame);
      Mesh2D shadow = Shadow(Rotate(small_poly, frame));
      std::vector<int> hull = QuickHull(shadow.vertices);

      if (RENDER_HULL) {
        Rendering rendering(small_poly, 1920, 1080);
        rendering.RenderMesh(shadow);
        rendering.RenderHull(shadow, hull);
        rendering.Save(std::format("debug-{:b}.png", code));
      }


      // Make sure winding order is clockwise.
      CHECK(hull.size() >= 3);

      std::vector<vec2> rshadow =
        ReferenceShadow(small_poly, hull, example_q);
      double area = SignedAreaOfConvexPoly(rshadow);
      printf("Area for example hull: %.17g\n", area);
      if (area < 0.0) {
        VectorReverse(&hull);
        rshadow = ReferenceShadow(small_poly, hull, example_q);
        double area = SignedAreaOfConvexPoly(rshadow);
        printf("Area for reversed hull: %.17g\n", area);
      }

      if (RENDER_HULL) {
        Bounds bounds;
        for (const auto &vec : rshadow) {
          bounds.Bound(vec.x, vec.y);
        }
        bounds.AddMarginFrac(0.05);

        ImageRGBA ref(1920, 1080);
        ref.Clear32(0x000000FF);
        Bounds::Scaler scaler =
          bounds.ScaleToFit(ref.Width(), ref.Height()).FlipY();
        for (int i = 0; i < rshadow.size(); i++) {
          const vec2 &v0 = rshadow[i];
          const vec2 &v1 = rshadow[(i + 1) % rshadow.size()];
          const auto &[x0, y0] = scaler.Scale(v0.x, v0.y);
          const auto &[x1, y1] = scaler.Scale(v1.x, v1.y);
          ref.BlendLine32(x0, y0, x1, y1, 0xFFFFFFAA);
        }
        ref.Save(std::format("ref-hull-{:b}.png", code));
      }

      return hull;
    }();

  std::string setup;
  Z3Quat view = GetPatchQuat(&setup, boundaries, code, mask);

  std::vector<Z3Vec2> shadow =
    EmitShadow(&setup, boundaries.big_poly, hull, view);

  Z3Real area = EmitConvexPolyArea(&setup, shadow);

  Z3Real area_v = NewReal(&setup, "area");
  AppendFormat(&setup, "(assert (= {} {}))\n", area_v.s, area.s);

  StatusBar status(3);
  status.Clear();

  // Check that the thing is satisfiable at all.
  if (true) {
    Timer timer;
    std::string sanity = setup;
    AppendFormat(&sanity,
                 "(check-sat)\n"
                 "(get-model)\n");
    status.Printf("Sanity check satisfiability... (%lld bytes)\n",
                  (int64_t)sanity.size());
    CHECK(Z3Result::SAT == RunZ3(sanity, {120.0})) << "Couldn't prove "
      "that the setup is satisfiable?";
    status.Printf("Satisfiable; OK in %s\n",
                  ANSI::Time(timer.Seconds()).c_str());
  }


  // Check again with the example quat asserted.
  {
    Timer timer;
    std::string sanity = setup;
    AppendFormat(&sanity,
                 "(assert (= {} {}))\n"
                 "(assert (= {} {}))\n"
                 "(assert (= {} {}))\n"
                 "(assert (= {} {}))\n",
                 view.x.s, Z3Real(example_q.x).s,
                 view.y.s, Z3Real(example_q.y).s,
                 view.z.s, Z3Real(example_q.z).s,
                 view.w.s, Z3Real(example_q.w).s);

    AppendFormat(&sanity,
                 "(check-sat)\n"
                 "(get-model)\n");
    status.Printf("Sanity check example... (%lld bytes)\n",
                  (int64_t)sanity.size());
    CHECK(Z3Result::SAT == RunZ3(sanity, {120.0}));
    status.Printf("Example satisfiable; OK in %s\n",
                  ANSI::Time(timer.Seconds()).c_str());

  }

  // The area of the parameterized hull is an interval from
  // min_area to max_area (which we don't know). Here we try
  // to put bounds on each.

  // Bounds on the maximum possible area.
  RatBounds max_area(BigRat(0), BigRat(100000));
  // Likewise, bounds on the minimal possible area.
  RatBounds min_area(BigRat(0), BigRat(100000));


  double timeout = 600.0;

  int64_t sat = 0, unsat = 0, unknown = 0;
  for (int64_t iters = 0; true; iters++) {
    std::string stats =
      std::format("{} iters, {} sat, {} unsat, {} unknown, {}",
                  iters, sat, unsat, unknown, ANSI::Time(timer.Seconds()));
    std::string mins =
      std::format("min area: {}", min_area.BriefString());
    std::string maxes =
      std::format("max area: {}", max_area.BriefString());

    status.EmitStatus({stats, mins, maxes});

    std::string out = setup;

    // Helpful to add the existing bounds?

    bool do_min = iters & 1;

    if (do_min) {
      BigRat test_point = min_area.Midpoint();

      AppendFormat(&out,
                   "(assert (<= {} {}))\n",
                   area_v.s, Z3Real(test_point).s);

      AppendFormat(&out,
                   "(check-sat)\n");

      Z3Result lesseq = RunZ3(out, {timeout});
      switch (lesseq) {
      case Z3Result::SAT:
        // Possible for the area to be lesseq than the test point.
        status.Printf("sat: area <= %s", test_point.ToString().c_str());
        if (test_point <= min_area.ub) {
          min_area.ub = test_point;
          min_area.uclosed = true;
        }
        sat++;
        break;
      case Z3Result::UNSAT:
        // The area is always more than the test point.
        status.Printf("unsat: area <= %s", test_point.ToString().c_str());
        if (test_point >= min_area.lb) {
          min_area.lb = test_point;
          min_area.lclosed = false;
        }
        unsat++;
        break;
      case Z3Result::UNKNOWN:
        // No info.
        unknown++;
        status.Printf(ARED("unknown") ": area <= %s",
                      test_point.ToString().c_str());
        break;
      }

    } else {
      // Maximize.
      BigRat test_point = max_area.Midpoint();

      AppendFormat(&out,
                   "(assert (>= {} {}))\n",
                   area_v.s, Z3Real(test_point).s);

      AppendFormat(&out,
                   "(check-sat)\n");

      Z3Result lesseq = RunZ3(out, {timeout});
      switch (lesseq) {
      case Z3Result::SAT:
        status.Printf("sat: area >= %s", test_point.ToString().c_str());
        // Possible for the area to be greatereq than the test point.
        if (test_point >= max_area.lb) {
          max_area.lb = test_point;
          max_area.lclosed = true;
        }
        sat++;
        break;
      case Z3Result::UNSAT:
        status.Printf("unsat: area >= %s", test_point.ToString().c_str());
        // The area is always less than the test point.
        if (test_point <= max_area.ub) {
          max_area.ub = test_point;
          max_area.uclosed = false;
        }
        unsat++;
        break;
      case Z3Result::UNKNOWN:
        // No info.
        status.Printf(ARED("unknown") ": area >= %s",
                      test_point.ToString().c_str());
        unknown++;
        break;
      }
    }
  }
}

// Find the set of patches (as their codes) that are non-empty, by
// shelling out to z3. This could be optimized a lot, but the set is a
// fixed property of the snub cube (given the ordering of vertices and
// faces), so we just need to enumerate them once.
struct PatchEnumerator {
  PatchEnumerator() : scube(BigScube(DIGITS)), boundaries(scube), status(1) {
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
      Z3Vec3 normal{boundaries.big_planes[b]};
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

    z3calls++;
    status.Statusf("Z3: %s", std::format("{:b}", code).c_str());
    Z3Result z3result = RunZ3(out);
    status.Statusf("%lld Z3 calls. Depth %d\n", z3calls, depth);
    CHECK(z3result != Z3Result::UNKNOWN) << "Expecting a definitive "
      "answer here";

    if (z3result == Z3Result::UNSAT) {
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

[[maybe_unused]]
static void MaxArea() {
  BigPoly scube = BigScube(DIGITS);
  Boundaries boundaries(scube);

  // GetMaximumArea(boundaries, uint64_t{0b1010111101010001010010100000});
  BoundArea(boundaries, uint64_t{0b1010111101010001010010100000});
}


static std::string MaskedBits(int num_bits,
                              uint64_t code,
                              uint64_t mask) {
  std::string out;
  uint8_t prev = 0x2A;
  for (int i = num_bits - 1; i >= 0; i--) {
    uint64_t pos = uint64_t{1} << i;
    uint32_t cur = ((!!(code & pos)) << 1) | (!!(mask & pos));
    if (cur != prev) {
      if (mask & pos) {
        // forced bit.
        if (code & pos) {
          out.append(ANSI::ForegroundRGB32(0x76F5F3FF));
        } else {
          out.append(ANSI::ForegroundRGB32(0xB8BBF2FF));
        }
      } else {
        if (code & pos) {
          out.append(ANSI::ForegroundRGB32(0x023540FF));
        } else {
          out.append(ANSI::ForegroundRGB32(0x1A1F6EFF));
        }
      }
      prev = cur;
    }
    out.push_back((code & pos) ? '1' : '0');
  }

  out.append(ANSI_RESET);
  return out;
}

static void ComputeMasks() {
  BigPoly scube = BigScube(DIGITS);
  Boundaries boundaries(scube);

  // uint64_t example_code = uint64_t{0b1010111101010001010010100000};
  // uint64_t example_code = uint64_t{0b1101110011101000001011100000101};
  uint64_t example_code = uint64_t{0b101000010101110101111011111};
  uint64_t mask = GetCodeMask(boundaries, example_code);

  printf("Code: %s\n"
         "Mask: %s\n"
         "Full: %s\n",
         std::format("{:b}", example_code).c_str(),
         std::format("{:b}", mask).c_str(),
         MaskedBits(boundaries.Size(), example_code, mask).c_str());
}

int main(int argc, char **argv) {
  ANSI::Init();

  // BigSnubHulls();

  /*
  PatchEnumerator pe;
  pe.Enumerate();
  */
  MaxArea();
  // ComputeMasks();

  return 0;
}
