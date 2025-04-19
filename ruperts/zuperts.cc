
// Generate Z3 formula.

#include <cstdint>
#include <cstdio>
#include <format>
#include <print>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <unordered_set>

#include "ansi.h"
#include "base/stringprintf.h"
#include "bignum/polynomial.h"
#include "polyhedra.h"
#include "util.h"

#include "z3.h"

// require no translation of either polyhedron, only rotations.
static constexpr bool REQUIRE_ORIGIN = true;

enum class Symmetry {
  ICOSAHEDRAL,
  OCTAHEDRAL,
  TETRAHEDRAL,
};

enum class Containment {
  TRIANGULATION,
  COMBINATION,
};

inline constexpr Containment containment = Containment::COMBINATION;

enum class Parameterization {
  QUATS,
  MATRICES,
  BOUNDED_MATRICES,
};

inline constexpr Parameterization parameterization =
  Parameterization::BOUNDED_MATRICES;
// Parameterization::QUATS;

inline constexpr Symmetry SYMMETRY = Symmetry::OCTAHEDRAL;

static std::vector<Z3Vec2> EmitShadowFromFrame(std::string *out,
                                               const std::vector<Z3Vec3> &vin,
                                               const Z3Frame &frame,
                                               const Z3Vec2 &trans,
                                               std::string_view name_hint) {
  // Now multiply each vertex.
  std::vector<Z3Vec2> ret;
  ret.reserve(vin.size());
  for (int i = 0; i < vin.size(); i++) {
    const Z3Vec3 &v = vin[i];

    Z3Vec3 x = frame.x * v.x;
    Z3Vec3 y = frame.y * v.y;
    Z3Vec3 z = frame.z * v.z;

    // Just discarding the z coordinates.
    Z3Vec2 proj(x.x + y.x + z.x + trans.x,
                x.y + y.y + z.y + trans.y);

    Z3Vec2 p = NameVec2(out, proj, std::format("s{}{}", name_hint, i));
    ret.push_back(p);
  }

  return ret;
}

static std::vector<Z3Vec2> EmitShadowFromQuat(std::string *out,
                                              const std::vector<Z3Vec3> &vin,
                                              const Z3Quat &q,
                                              const Z3Vec2 &trans,
                                              std::string_view name_hint) {
  Z3Real xx = q.x * q.x;
  Z3Real yy = q.y * q.y;
  Z3Real zz = q.z * q.z;
  Z3Real ww = q.w * q.w;

  // This normalization term is |v|^-2, which is
  // 1/(sqrt(x^2 + y^2 + z^2 + w^2))^2, so we can avoid
  // the square root! It's always multipled by 2 in the
  // terms below, so we also do that here.
  Z3Real two_s = NameReal(out,
                          Z3Real(2) / (xx + yy + zz + ww),
                          std::format("{}_two_s", name_hint));

  Z3Real xy = q.x * q.y;
  Z3Real zx = q.z * q.x;
  Z3Real zw = q.z * q.w;
  Z3Real yw = q.y * q.w;
  Z3Real yz = q.y * q.z;
  Z3Real xw = q.x * q.w;

  /*
    1 - 2s(y^2 + z^2)    ,   2s (x y - z w)      ,   2s (x z + y w)
    2s (x y + z w)       ,   1 - 2s(x^2 + z^2)   ,   2s (y z - x w)
    2s (x z - y w)       ,   2s (y z + x w)      ,   1 - 2s(x^2 + y^2)
  */

  // Following yocto, the matrix consists of three columns called "x",
  // "y", "z", each with "x", "y", "z" coordinates.

  // We give these names since they appear many times (for each vertex!)
  std::string m = std::format("{}m", name_hint);
  // left column
  Z3Vec3 mx =
    NameVec3(out, Z3Vec3(Z3Real(1) - two_s * (yy + zz),
                         two_s * (xy + zw),
                         two_s * (zx - yw)), std::format("{}x", m));
  // middle
  Z3Vec3 my =
    NameVec3(out, Z3Vec3(two_s * (xy - zw),
                         Z3Real(1) - two_s * (xx + zz),
                         two_s * (yz + xw)), std::format("{}y", m));
  // right
  Z3Vec3 mz =
    NameVec3(out, Z3Vec3(two_s * (zx + yw),
                         two_s * (yz - xw),
                         Z3Real(1) - two_s * (xx + yy)),
             std::format("{}z", m));

  Z3Frame frame(mx, my, mz);

  return EmitShadowFromFrame(out, vin, frame, trans, name_hint);
}

static void AssertRotationBounds(std::string *out,
                                 Symmetry sg,
                                 const Z3Frame &frame) {
  // Without loss of generality, we can put a bound α/2 on the angular
  // distance of the rotation. α is the maximum angular distance between
  // two vertices on the corresponding polyhedron (e.g. the icosahedron
  // for the icosahedral group). α/2 is thus a bound on the maximum
  // angular distance between any point on the unit sphere and a vertex
  // of the symmetry group's polyhedron.
  //
  // Thinking about the cube: α will be 90 degrees. If we are trying
  // to rotate some distinguished vertex from the position s to the
  // position e (where θ is the angle subtended), we can always
  // produce an equivalent rotation with θ <= α/2. We do this by
  // picking the closest (angular distance) vertex v (in the original
  // orientation) to e; this can be no more than α/2 away. Then we
  // just apply the symmetry group operations to move s to v, and
  // then rotate from v to e with an angular distance of no more than
  // α/2.
  //
  // This argument assumes vertex transitivity, but applies regardless
  // of the transitive set (e.g. if face transitive, consider the dual
  // polyhedron, which has the same symmetry group).
  //

  // There's a nice way to place bounds on the angle, which is to
  // use the trace of the matrix (sum of main diagonal).
  // trace(M) ≥ 1 + 2cos(β) ensures that the rotation has an
  // angular distance of no more than β.
  //

  Z3Real bound = [out, sg]() -> Z3Real {
      switch (sg) {
      case Symmetry::OCTAHEDRAL: {
        // For the octahedral group, α = 90 degrees, and cos(α/2) is
        // 1/sqrt(2). So we have trace(M) ≥ 1 + 2/sqrt(2)
        // (Simplify: 2/sqrt(2) = (sqrt(2) * sqrt(2))/sqrt(2) = sqrt(2).)
        //
        // trace(M) ≥ 1 + sqrt(2)

        Z3Real sqrt2 = NewReal(out, "sqrt2");
        AppendFormat(out, "(assert (= 2.0 (* sqrt2 sqrt2)))\n");
        return Z3Real(1) + sqrt2;
      }
      case Symmetry::ICOSAHEDRAL:
        // For the icosahedral group, α = arccos(1/sqrt(5)), and
        // cos(α/2) = sqrt((5 + sqrt(5)) / 10)    (wolfram alpha).
        // trace(M) ≥ 1 + 2 * sqrt((5 + sqrt(5)) / 10)

        // XXX We can express the bound exactly, but I don't think
        // it's likely to be helpful since it would involve roots of
        // roots?
        return Z3Real("2.701301616704079864363");

      case Symmetry::TETRAHEDRAL:
        // XXX I didn't check this carefully.
        // For the tetrahedron, the bound seems to be 90 degrees.
        // This gives:
        // trace(M) ≥ 1
        return Z3Real(1);

      default:
        LOG(FATAL) << "unimplemented!";
        return Z3Real("error");
      }
    }();


  Z3Real trace = frame.x.x + frame.y.y + frame.z.z;
  AppendFormat(out, ";; constrain rotation due to symmetries\n");
  AppendFormat(out, "(assert (>= {} {}))\n", trace.s, bound.s);
}

static std::pair<std::vector<Z3Vec2>, std::vector<Z3Vec2>>
MakeParameterization(std::string *out,
                     const SymbolicPolyhedron &outer_poly,
                     const SymbolicPolyhedron &inner_poly,
                     const std::vector<Z3Vec3> &outer_verts,
                     const std::vector<Z3Vec3> &inner_verts) {

  Z3Vec2 trans(NewReal(out, "tx"), NewReal(out, "ty"));
  if (REQUIRE_ORIGIN) {
    AppendFormat(out, ";; require no translation\n");
    AppendFormat(out, "(assert (= {} 0.0))\n", trans.x.s);
    AppendFormat(out, "(assert (= {} 0.0))\n", trans.y.s);
  }

  if (parameterization == Parameterization::QUATS) {
    // Parameters.
    Z3Quat qo = NewQuat(out, "o");
    Z3Quat qi = NewQuat(out, "i");

    // AppendFormat(out, "(assert (= {} 0.0))\n", qi.x.s);
    // AppendFormat(out, "(assert (= {} 0.0))\n", qi.y.s);
    // AppendFormat(out, "(assert (= {} 0.0))\n", qi.z.s);
    // AppendFormat(out, "(assert (= {} 1.0))\n", qi.w.s);

    Z3Vec2 id(Z3Real(0), Z3Real(0));

    // Polyhedra in their rotated orientations.
    std::vector<Z3Vec2> outer_shadow =
      EmitShadowFromQuat(out, outer_verts, qo, id, "o");
    std::vector<Z3Vec2> inner_shadow =
      EmitShadowFromQuat(out, inner_verts, qi, trans, "i");

    return std::make_pair(outer_shadow, inner_shadow);
  } else if (parameterization == Parameterization::MATRICES ||
             parameterization == Parameterization::BOUNDED_MATRICES) {
    Z3Frame outer_frame = NewFrame(out, "o");
    Z3Frame inner_frame = NewFrame(out, "i");

    AssertRotationFrame(out, outer_frame);
    AssertRotationFrame(out, inner_frame);

    if (parameterization == Parameterization::BOUNDED_MATRICES) {
      AssertRotationBounds(out, SYMMETRY, outer_frame);
      AssertRotationBounds(out, SYMMETRY, inner_frame);
    }

    Z3Vec2 id(Z3Real(0), Z3Real(0));

    std::vector<Z3Vec2> outer_shadow =
      EmitShadowFromFrame(out, outer_verts, outer_frame, id, "o");
    std::vector<Z3Vec2> inner_shadow =
      EmitShadowFromFrame(out, inner_verts, inner_frame, trans, "i");

    return std::make_pair(outer_shadow, inner_shadow);
  } else {
    LOG(FATAL) << "Bad parameterization";
    return {};
  }
}

static void EmitProblem(const SymbolicPolyhedron &outer,
                        const SymbolicPolyhedron &inner,
                        std::string_view filename) {
  std::string out;

  // Polyhedra in their base orientations.
  std::vector<Z3Vec3> outer_verts = EmitPolyhedron(outer, "o", &out);
  std::vector<Z3Vec3> inner_verts = EmitPolyhedron(inner, "i", &out);

  const auto &[outer_shadow, inner_shadow] =
    MakeParameterization(&out,
                         outer, inner,
                         outer_verts, inner_verts);

  // Containment constraints...
  if (containment == Containment::TRIANGULATION) {
    for (int i = 0; i < inner_shadow.size(); i++) {
      const Z3Vec2 &pt = inner_shadow[i];
      // Must be in some triangle.
      std::vector<Z3Bool> tris;
      for (const auto &[a, b, c] : outer.faces->triangulation) {
        tris.push_back(EmitInTriangle(&out,
                                      outer_shadow[a],
                                      outer_shadow[b],
                                      outer_shadow[c],
                                      pt));
      }
      AppendFormat(&out,
                   ";; inner point {} in triangle?\n"
                   "(assert {})\n", i, Or(tris).s);
    }
  } else if (containment == Containment::COMBINATION) {
    AppendFormat(&out,
                 "\n\n"
                 ";; Inner shadow is in outer shadow. Each point\n"
                 ";; must be a convex combination of the points in\n"
                 ";; the outer shadow.\n");
    for (int i = 0; i < inner_shadow.size(); i++) {
      const Z3Vec2 &ipt = inner_shadow[i];
      std::vector<Z3Real> weights;
      std::vector<Z3Vec2> weighted_sum;
      for (int j = 0; j < outer_shadow.size(); j++) {
        const Z3Vec2 &opt = outer_shadow[j];
        Z3Real weight = NewReal(&out, std::format("w{}_{}", i, j));
        // Require *strictly* greater than zero.
        AppendFormat(&out, "(assert (> {} 0.0))\n", weight.s);
        // Redundant with the total sum below, but harmless?
        AppendFormat(&out, "(assert (< {} 1.0))\n", weight.s);
        weights.push_back(weight);
        weighted_sum.push_back(opt * weight);
      }
      AppendFormat(&out, ";; inner v{}'s weights sum to 1\n", i);
      AppendFormat(&out, "(assert (= 1.0 {}))\n", Sum(weights).s);

      Z3Vec2 sum = Sum(weighted_sum);
      AppendFormat(&out, ";; inner v{} is in outer hull\n", i);
      AppendFormat(&out, "(assert (= {} {}))\n", ipt.x.s, sum.x.s);
      AppendFormat(&out, "(assert (= {} {}))\n", ipt.y.s, sum.y.s);
    }
  }

  AppendFormat(&out,
               "\n"
               ";; try saving progress\n"
               "(set-option :solver.cancel-backup-file "
               "\"zuperts-{}-{}-partial.z3\")\n",
               outer.name, inner.name);

  AppendFormat(&out, "\n");
  AppendFormat(&out, "(check-sat)\n");
  AppendFormat(&out, "(get-model)\n");

  Util::WriteFile(filename, out);
  std::print("Wrote {}\n", filename);
}

static void Emit() {
  SymbolicPolyhedron cube = SymbolicTetrahedron();
  EmitProblem(cube, cube, "tetrahedron.z3");
}

int main(int argc, char **argv) {
  ANSI::Init();

  Emit();

  printf("OK\n");
  return 0;
}
