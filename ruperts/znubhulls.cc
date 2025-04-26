
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <format>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "ansi.h"
#include "arcfour.h"
#include "base/logging.h"
#include "base/stringprintf.h"
#include "big-polyhedra.h"
#include "bignum/big.h"
#include "numbers.h"
#include "patches.h"
#include "polyhedra.h"
#include "run-z3.h"
#include "status-bar.h"
#include "timer.h"
#include "util.h"
#include "yocto_matht.h"
#include "z3.h"

static constexpr int DIGITS = 24;

#define AMINT(s) ANSI_FG(125, 232, 186) s ANSI_RESET
#define ASKY(s) ANSI_FG(187, 233, 242) s ANSI_RESET

enum class ViewParameterization {
  QUATERNION,
  UNIT_VEC,
};

static constexpr ViewParameterization view_parameterization =
  ViewParameterization::UNIT_VEC;

static constexpr bool POINT_CONSTS = false;

[[maybe_unused]]
static vec3 RandomPointOnSphere(ArcFour *rc) {
  const quat4 small_quat = normalize(RandomQuaternion(rc));
  return QuaternionToSpherePoint(small_quat);
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

    Z3Vec3 v_in = NameVec3(out, poly.vertices[vertex_index],
                           std::format("p{}", i));

    if (POINT_CONSTS) {
      v_in = DeclareVec3(out, v_in, std::format("p{}", i));
    }

    Z3Vec3 v_out = TransformPoint(frame, v_in);
    Z3Vec2 v2_out = NameVec2(out, Z3Vec2(v_out.x, v_out.y),
                             std::format("v{}", i));

    if (POINT_CONSTS) {
      v2_out = DeclareVec2(out, v2_out, std::format("v{}", i));
    }

    projected_hull.push_back(v2_out);
  }

  return projected_hull;
}

std::vector<Z3Vec2> EmitShadow(std::string *out,
                               const BigPoly &poly,
                               const std::vector<int> &hull,
                               const Z3Vec3 &unit_vec_view) {

  Z3Frame frame = FrameFromUnitViewPos(out, unit_vec_view);

  std::vector<Z3Vec2> projected_hull;
  projected_hull.reserve(hull.size());

  for (size_t i = 0; i < hull.size(); ++i) {
    int vertex_index = hull[i];
    CHECK(vertex_index >= 0 && vertex_index < poly.vertices.size());

    Z3Vec3 v_in = NameVec3(out, poly.vertices[vertex_index],
                              std::format("p{}", i));
    Z3Vec3 v_out = TransformPoint(frame, v_in);
    Z3Vec2 v2_out = NameVec2(out, Z3Vec2(v_out.x, v_out.y),
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
    ub(std::move(initial_ub)) {
    Succeeded();
  }

  BigRat Midpoint() const {
    return (lb + ub) / BigRat(2);
  }

  std::string BriefString() const {
    double l = lb.ToDouble(), u = ub.ToDouble();
    return std::format("{:.17g} {} x {} {:.17g}",
                       l, lclosed ? "<=" : "<",
                       uclosed ? "<=" : "<", u);
  }

  BigRat Guess() const {
    BigRat span = ub - lb;
    return lb + (span * BigRat(numer)) / BigRat(denom);
  }

  void Succeeded() {
    // Start with midpoint.
    numer = 1;
    denom = 2;
  }

  void Failed() {
    numer++;
    for (;;) {
      if (numer == denom) {
        denom++;
        numer = 1;
        return;
      }

      // Only relatively prime fractions; otherwise
      // we have already tried them.
      if (GCD(numer, denom) == 1) {
        return;
      }
    }
  }

  int64_t denom = 2;
  int64_t numer = 1;

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

  printf("Compute hull once...\n");
  const std::vector<int> hull =
    ComputeHullForPatch(boundaries, code, mask, {"area"});

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

void BoundEdges(const Boundaries &boundaries,
                uint64_t code) {
  Timer timer;
  uint64_t mask = GetCodeMask(boundaries, code);
  // mask = mask | (mask << 1);
  // const uint64_t mask = ~0;
  mask = ~0;

  printf("For code: %s\n",
         boundaries.ColorMaskedBits(code, mask).c_str());

  printf("Compute hull once...\n");
  const std::vector<int> hull =
    ComputeHullForPatch(boundaries, code, mask, {"edge"});

  // These are just used for sanity checking satisfiability.
  printf("Get quat in patch...\n");
  BigQuat example_q = GetBigQuatInPatch(boundaries, code);
  printf("Get view pos (vector)...\n");
  BigVec3 example_v = ViewPosFromNonUnitQuat(example_q);

  std::string setup;

  std::vector<Z3Vec2> shadow;
  std::string assert_example_view;
  switch (view_parameterization) {
  case ViewParameterization::QUATERNION: {
    Z3Quat view = GetPatchQuat(&setup, boundaries, code, mask);
    shadow = EmitShadow(&setup, boundaries.big_poly, hull, view);

    assert_example_view =
      std::format("(assert (= {} {}))\n"
                  "(assert (= {} {}))\n"
                  "(assert (= {} {}))\n"
                  "(assert (= {} {}))\n",
                  view.x.s, Z3Real(example_q.x).s,
                  view.y.s, Z3Real(example_q.y).s,
                  view.z.s, Z3Real(example_q.z).s,
                  view.w.s, Z3Real(example_q.w).s);

    break;
  }

  case ViewParameterization::UNIT_VEC: {
    Z3Vec3 view = GetPatchView(&setup, boundaries, code, mask);
    shadow = EmitShadow(&setup, boundaries.big_poly, hull, view);

    assert_example_view =
      std::format("(assert (= {} {}))\n"
                  "(assert (= {} {}))\n"
                  "(assert (= {} {}))\n",
                  view.x.s, Z3Real(example_v.x).s,
                  view.y.s, Z3Real(example_v.y).s,
                  view.z.s, Z3Real(example_v.z).s);
    break;
  }
  default:
    LOG(FATAL) << "??";
  }

  CHECK(shadow.size() >= 3);

  // Pick a pair of edges and compute their squared distance.
  const Z3Vec2 &v1 = shadow[0];
  const Z3Vec2 &v2 = shadow[1];

  Z3Real dx = v2.x - v1.x;
  Z3Real dy = v2.y - v1.y;

  const Z3Real sqd =
    DeclareReal(&setup, (dx * dx) + (dy * dy), "sq_dist");

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
    sanity.append(assert_example_view);

    AppendFormat(&sanity,
                 "(check-sat)\n"
                 "(get-model)\n");
    status.Printf("Sanity check example... (%lld bytes)\n",
                  (int64_t)sanity.size());
    CHECK(Z3Result::SAT == RunZ3(sanity, {120.0}));
    status.Printf("Example satisfiable; OK in %s\n",
                  ANSI::Time(timer.Seconds()).c_str());
  }

  #if 0
  AppendFormat(&setup,
               "(maximize {})\n"
               "(check-sat)\n"
               "(get-model)\n",
               sqd.s);

  Util::WriteFile(
      std::format("edge-{}-{}.z3",
                  hull[0], hull[1]),
      setup);
  printf("Wrote it.\n");
  #endif

  // Bounds on the maximum possible.
  RatBounds max_sqlen(BigRat(0), BigRat(100000));
  // Likewise, bounds on the minimal possible.
  RatBounds min_sqlen(BigRat(0), BigRat(100000));

  double timeout = 600.0;

  auto ResetTimeout = [&]() {
      timeout = 600.0;
    };

  int64_t sat = 0, unsat = 0, unknown = 0;
  for (int64_t iters = 0; true; iters++) {
    std::string stats =
      std::format("{} z3s, {} sat, {} unsat, {} " ARED("unk") ", "
                  "{} total {} timeout",
                  iters, sat, unsat, unknown,
                  ANSI::Time(timer.Seconds()),
                  ANSI::Time(timeout));
    std::string mins =
      std::format(AWHITE("min length") ": {} " AGREY("[split {}]"),
                  min_sqlen.BriefString(),
                  min_sqlen.Guess().ToString());
    std::string maxes =
      std::format(AWHITE("max length") ": {} " AGREY("[split {}]"),
                  max_sqlen.BriefString(),
                  max_sqlen.Guess().ToString());

    status.EmitStatus({stats, mins, maxes});

    std::string out = setup;

    // Helpful to add the existing bounds?

    bool do_min = iters & 1;

    if (do_min) {
      BigRat test_point = min_sqlen.Guess();

      AppendFormat(&out,
                   "(assert (<= {} {}))\n",
                   sqd.s, Z3Real(test_point).s);

      AppendFormat(&out,
                   "(check-sat)\n");

      Z3Result lesseq = RunZ3(out, {timeout});
      switch (lesseq) {
      case Z3Result::SAT:
        // Possible for the length to be lesseq than the test point.
        status.Printf(AMINT("sat")
                      ": length <= %s", test_point.ToString().c_str());
        if (test_point <= min_sqlen.ub) {
          min_sqlen.ub = test_point;
          min_sqlen.uclosed = true;
          min_sqlen.Succeeded();
          ResetTimeout();
        }
        sat++;
        break;
      case Z3Result::UNSAT:
        // The length is always more than the test point.
        status.Printf(ASKY("unsat")
                      ": length <= %s", test_point.ToString().c_str());
        if (test_point >= min_sqlen.lb) {
          min_sqlen.lb = test_point;
          min_sqlen.lclosed = false;
          min_sqlen.Succeeded();
          ResetTimeout();
        }
        unsat++;
        break;
      case Z3Result::UNKNOWN:
        // No info.
        unknown++;
        status.Printf(ARED("unknown") ": length <= %s. timeout now %s",
                      test_point.ToString().c_str(),
                      ANSI::Time(timeout).c_str());
        timeout *= 1.25;
        min_sqlen.Failed();
        break;
      }

    } else {
      // Maximize.
      BigRat test_point = max_sqlen.Guess();

      AppendFormat(&out,
                   "(assert (>= {} {}))\n",
                   sqd.s, Z3Real(test_point).s);

      AppendFormat(&out,
                   "(check-sat)\n");

      Z3Result lesseq = RunZ3(out, {timeout});
      switch (lesseq) {
      case Z3Result::SAT:
        status.Printf(AMINT("sat")
                      ": length >= %s", test_point.ToString().c_str());
        // Possible for the length to be greatereq than the test point.
        if (test_point >= max_sqlen.lb) {
          max_sqlen.lb = test_point;
          max_sqlen.lclosed = true;
          max_sqlen.Succeeded();
          ResetTimeout();
        }
        sat++;
        break;
      case Z3Result::UNSAT:
        status.Printf(ASKY("unsat")
                      ": length >= %s", test_point.ToString().c_str());
        // The length is always less than the test point.
        if (test_point <= max_sqlen.ub) {
          max_sqlen.ub = test_point;
          max_sqlen.uclosed = false;
          max_sqlen.Succeeded();
          ResetTimeout();
        }
        unsat++;
        break;
      case Z3Result::UNKNOWN:
        // No info.
        status.Printf(ARED("unknown") ": length >= %s. timeout now {}",
                      test_point.ToString().c_str(),
                      ANSI::Time(timeout).c_str());
        unknown++;
        timeout *= 1.25;
        max_sqlen.Failed();
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

static void EdgeBounds() {
  BigPoly scube = BigScube(DIGITS);
  Boundaries boundaries(scube);

  // GetMaximumArea(boundaries, uint64_t{0b1010111101010001010010100000});
  BoundEdges(boundaries, uint64_t{0b1010111101010001010010100000});
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
         boundaries.ColorMaskedBits(example_code, mask).c_str());
}

int main(int argc, char **argv) {
  ANSI::Init();

  // BigSnubHulls();

  /*
  PatchEnumerator pe;
  pe.Enumerate();
  */
  // MaxArea();
  EdgeBounds();
  // ComputeMasks();

  return 0;
}
