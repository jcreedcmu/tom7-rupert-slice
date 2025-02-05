
#include "big-polyhedra.h"

#include <cstdio>
#include <cstdlib>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "ansi.h"
#include "base/stringprintf.h"
#include "bignum/big-numbers.h"
#include "bignum/big-overloads.h"
#include "bignum/big.h"
#include "point-map.h"
#include "polyhedra.h"
#include "yocto_matht.h"

[[maybe_unused]]
static double IsNear(double a, double b) {
  return std::abs(a - b) < 0.0000001;
}

#define CHECK_NEAR(f, g) do {                                           \
  const double fv = (f);                                                \
  const double gv = (g);                                                \
  const double e = std::abs(fv - gv);                                   \
  CHECK(e < 0.0000001) << "Expected " << #f << " and " << #g <<         \
    " to be close, but got: " <<                                        \
    StringPrintf("%.17g and %.17g, with err %.17g", fv, gv, e);         \
  } while (0)


std::string VecString(const BigVec2 &v) {
  return StringPrintf(
      "(x: " ARED("%s") " ≅ %.17g; "
      "y: " AGREEN("%s") " ≅ %.17g)",
      v.x.ToString().c_str(), v.x.ToDouble(),
      v.y.ToString().c_str(), v.y.ToDouble());
}

std::string QuatString(const BigQuat &q) {
  return StringPrintf(
      "x: " ARED("%s") " ≅ %.17g\n"
      "y: " AGREEN("%s") " ≅ %.17g\n"
      "z: " ABLUE("%s") " ≅ %.17g\n"
      "w: " AYELLOW("%s") " ≅ %.17g\n",
      q.x.ToString().c_str(), q.x.ToDouble(),
      q.y.ToString().c_str(), q.y.ToDouble(),
      q.z.ToString().c_str(), q.z.ToDouble(),
      q.w.ToString().c_str(), q.w.ToDouble());
}

std::string PlainVecString(const BigVec2 &v) {
  return StringPrintf(
      "BigVec2{\n"
      "  x = %s ≅ %.17g\n"
      "  y = %s ≅ %.17g\n"
      "}\n",
      v.x.ToString().c_str(), v.x.ToDouble(),
      v.y.ToString().c_str(), v.y.ToDouble());
}

std::string PlainQuatString(const BigQuat &q) {
  return StringPrintf(
      "BigQuat{\n"
      "  x = %s ≅ %.17g\n"
      "  y = %s ≅ %.17g\n"
      "  z = %s ≅ %.17g\n"
      "  w = %s ≅ %.17g\n"
      "}\n",
      q.x.ToString().c_str(), q.x.ToDouble(),
      q.y.ToString().c_str(), q.y.ToDouble(),
      q.z.ToString().c_str(), q.z.ToDouble(),
      q.w.ToString().c_str(), q.w.ToDouble());
}


BigQuat Normalize(const BigQuat &q, int digits) {
  printf("Quat: %s\n", QuatString(q).c_str());
  BigRat norm_sq = q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w;
  printf("Norm^2: %s\n", norm_sq.ToString().c_str());
  CHECK(norm_sq != BigRat(0));
  BigRat norm = BigRat::Sqrt(norm_sq, BigNumbers::Digits(digits));
  printf("Norm: %s\n", norm.ToString().c_str());
  CHECK(norm != BigRat(0));
  return BigQuat(q.x / norm, q.y / norm, q.z / norm, q.w / norm);
}

BigFrame RotationFrame(const BigQuat &v) {
  BigRat two(2);
  return {
    {
      v.w * v.w + v.x * v.x - v.y * v.y - v.z * v.z,
      (v.x * v.y + v.z * v.w) * two,
      (v.z * v.x - v.y * v.w) * two
    },
    {
      (v.x * v.y - v.z * v.w) * two,
      v.w * v.w - v.x * v.x + v.y * v.y - v.z * v.z,
      (v.y * v.z + v.x * v.w) * two
    },
    {
      (v.z * v.x + v.y * v.w) * two,
      (v.y * v.z - v.x * v.w) * two,
      v.w * v.w - v.x * v.x - v.y * v.y + v.z * v.z
    },
    {BigRat(0), BigRat(0), BigRat(0)}
  };
}


BigFrame NonUnitRotationFrame(const BigQuat &v) {
  BigRat two(2);

  BigRat xx = v.x * v.x;
  BigRat yy = v.y * v.y;
  BigRat zz = v.z * v.z;
  BigRat ww = v.w * v.w;

  // This normalization term is |v|^-2, which is
  // 1/(sqrt(x^2 + y^2 + z^2 + w^2))^2, so we can avoid
  // the square root! It's always multiplied by 2 in the
  // terms below, so we also do that here.
  BigRat two_s = BigRat(2) / (xx + yy + zz + ww);

  BigRat xy = v.x * v.y;
  BigRat zx = v.z * v.x;
  BigRat zw = v.z * v.w;
  BigRat yw = v.y * v.w;
  BigRat yz = v.y * v.z;
  BigRat xw = v.x * v.w;

  BigRat one(1);

  /*
    1 - 2s(y^2 + z^2)    ,   2s (x y - z w)      ,   2s (x z + y w)
    2s (x y + z w)       ,   1 - 2s(x^2 + z^2)   ,   2s (y z - x w)
    2s (x z - y w)       ,   2s (y z + x w)      ,   1 - 2s(x^2 + y^2)
  */

  // Note that the yocto matrix looks a bit different from this. On the
  // diagonal, it's doing the trick that since v^2 = 1, we can write
  // 1 + s(y^2 + z^2) as (x^2 + y^2 + z^2 + w^2) - 2(y^2 + z^2) and
  // then cancel terms.
  //
  // Also note: Yocto stores these in column-major format. Since it's
  // a rotation matrix I think the transpose is just the inverse
  // (which should be fine for these problems if we are consistent),
  // but be careful.

  return {
    // Left
    {one - two_s * (yy + zz),
     two_s * (xy + zw),
     two_s * (zx - yw)},
    // Middle
    {two_s * (xy - zw),
     one - two_s * (xx + zz),
     two_s * (yz + xw)},
    // Right
    {two_s * (zx + yw),
     two_s * (yz - xw),
     one - two_s * (xx + yy)}};
}



BigMesh2D Shadow(const BigPoly &poly) {
  std::vector<BigVec2> vertices;
  vertices.reserve(poly.vertices.size());
  for (const BigVec3 &v : poly.vertices) {
    vertices.emplace_back(v.x, v.y);
  }

  return BigMesh2D{.vertices = std::move(vertices), .faces = poly.faces};
}

BigMesh2D RotateAndProject(const BigFrame &f, const BigPoly &poly) {
  std::vector<BigVec2> vertices;
  vertices.reserve(poly.vertices.size());
  for (const BigVec3 &v : poly.vertices) {
    vertices.push_back(TransformAndProjectPoint(f, v));
  }
  return BigMesh2D{.vertices = std::move(vertices), .faces = poly.faces};
}

BigPoly Rotate(const BigFrame &f, const BigPoly &poly) {
  std::vector<BigVec3> vertices;
  vertices.reserve(poly.vertices.size());
  for (const BigVec3 &v : poly.vertices) {
    vertices.push_back(TransformPoint(f, v));
  }
  return BigPoly{.vertices = std::move(vertices), .faces = poly.faces};
}

BigMesh2D Translate(const BigVec2 &t, const BigMesh2D &m) {
  std::vector<BigVec2> vertices;
  vertices.reserve(m.vertices.size());
  for (const BigVec2 &v : m.vertices) {
    vertices.push_back(v + t);
  }

  return BigMesh2D{.vertices = std::move(vertices), .faces = m.faces};
}

BigPoly Rotate(const BigQuat &q, const BigPoly &poly) {
  std::vector<BigVec3> vertices;
  vertices.reserve(poly.vertices.size());
  for (const BigVec3 &v : poly.vertices) {
    vertices.push_back(RotatePoint(q, v));
  }
  return BigPoly{.vertices = std::move(vertices), .faces = poly.faces};
}

// XXX Buggy?
BigVec3 RotatePoint(const BigQuat &q, const BigVec3 &v) {
  vec3 dv = SmallVec(v);
  quat4 dp{.x = dv.x, .y = dv.y, .z = dv.z, .w = 0.0};

  quat4 dq = SmallQuat(q);
  quat4 dqi = quat_conjugate(dq);
  quat4 dpqi = dp * dqi;
  quat4 r = dq * dpqi;

  // PERF: This can be simplified.
  //  - w coefficient for pure quaternion is zero.
  //  - q^-1 * p * q will be a pure quaternion for
  //    unit q, so we don't need to compute the w coefficient.
  //  - We are rotating lots of points with the same quat, so
  //    we should precompute.
  //  - The inverse is just the conjugate for unit q,
  //    so we can inline those too.
  //  - We actually want RotateAndProjectTo2D, since we're
  //    not even using the z coordinate.
  BigQuat p(v.x, v.y, v.z, BigRat(0));
  BigQuat pqi = p * UnitInverse(q);

  CHECK_NEAR(dpqi.x, pqi.x.ToDouble());
  CHECK_NEAR(dpqi.y, pqi.y.ToDouble());
  CHECK_NEAR(dpqi.z, pqi.z.ToDouble());
  CHECK_NEAR(dpqi.w, pqi.w.ToDouble());

  BigQuat pp = q * pqi;
  BigVec3 ret(std::move(pp.x), std::move(pp.y), std::move(pp.z));

  CHECK_NEAR(r.x, ret.x.ToDouble());
  CHECK_NEAR(r.y, ret.y.ToDouble());
  CHECK_NEAR(r.z, ret.z.ToDouble());

  return ret;
}

std::vector<int> BigHull(const std::vector<BigVec2> &bigvs) {
  PointMap2<char> duplicates;
  std::vector<vec2> dvs;
  dvs.reserve(bigvs.size());
  for (int i = 0; i < bigvs.size(); i++) {
    const BigVec2 &v = bigvs[i];
    vec2 dv(v.x.ToDouble(), v.y.ToDouble());
    const std::optional<int> old = duplicates.Get(dv);
    CHECK(!old.has_value()) <<
      StringPrintf("Maybe too close to use "
                   "double QuickHull to compute BigHull (old: %d = %s); "
                   "new: %d = %s\n",
                   old.value(),
                   VecString(dvs[old.value()]).c_str(),
                   i,
                   VecString(dv).c_str());
    duplicates.Add(dv, i);
    dvs.emplace_back(dv);
  }

  return QuickHull(dvs);
}

Polyhedron SmallPoly(const BigPoly &big) {
  std::vector<vec3> vertices;
  vertices.reserve(big.vertices.size());
  for (int i = 0; i < big.vertices.size(); i++) {
    const BigVec3 &v = big.vertices[i];
    vertices.push_back(vec3{
        .x = v.x.ToDouble(),
        .y = v.y.ToDouble(),
        .z = v.z.ToDouble(),
      });
  }

  return Polyhedron{
    .vertices = std::move(vertices),
    .faces = big.faces,
    .name = "converted",
  };
}

Mesh2D SmallMesh(const BigMesh2D &big) {
  std::vector<vec2> vertices;
  vertices.reserve(big.vertices.size());
  for (int i = 0; i < big.vertices.size(); i++) {
    const BigVec2 &v = big.vertices[i];
    vertices.push_back(vec2{
        .x = v.x.ToDouble(),
        .y = v.y.ToDouble(),
      });
  }

  return Mesh2D{
    .vertices = std::move(vertices),
    .faces = big.faces,
  };
}

BigPoly MakeBigPolyFromVertices(std::vector<BigVec3> vertices,
                                const char *name) {
  // XXX check
  PointMap3<int> duplicates;

  std::vector<vec3> dvertices;
  dvertices.reserve(vertices.size());
  for (const BigVec3 &v : vertices) {
    #ifndef BIG_USE_GMP
    LOG(FATAL) << "ToDouble only really works with GMP mode";
    #endif
    double x = v.x.ToDouble();
    double y = v.y.ToDouble();
    double z = v.z.ToDouble();
    // printf("%.17g,%.17g,%.17g\n", x, y, z);
    dvertices.emplace_back(x, y, z);
  }

  std::optional<Polyhedron> opoly =
    PolyhedronFromConvexVertices(dvertices, "bigpoly");
  CHECK(opoly.has_value());

  // Now match up the vertices with the original poly.
  const Polyhedron &poly = opoly.value();
  CHECK(poly.vertices.size() == dvertices.size());

  for (int i = 0; i < poly.vertices.size(); i++) {
    const BigVec3 &b = vertices[i];
    const vec3 &v = poly.vertices[i];
    double x = b.x.ToDouble();
    double y = b.y.ToDouble();
    double z = b.z.ToDouble();

    CHECK(std::abs(x - v.x) < 0.000001 &&
          std::abs(y - v.y) < 0.000001 &&
          std::abs(z - v.z) < 0.000001) << "Expected "
      "ConvexPolyhedronFromVertices to preserve the order of "
      "the vertices.";
  }

  BigPoly bpoly;
  bpoly.vertices = std::move(vertices);
  bpoly.faces = poly.faces;
  bpoly.name = name;
  return bpoly;
}

static void AddEvenPermutations(
    const BigRat &a, const BigRat &b, const BigRat &c,
    std::vector<BigVec3> *vertices) {
  // (a, b, c) - even
  // (b, c, a) - even
  // (c, a, b) - even

  vertices->emplace_back(a, b, c);

  if (a == b && b == c) return;

  vertices->emplace_back(b, c, a);
  vertices->emplace_back(c, a, b);
}

BigPoly BigRidode(int digits) {
  const BigRat epsilon = BigNumbers::Digits(digits);
  const BigRat phi = (BigRat(1) + BigRat::Sqrt(BigRat(5), epsilon)) / BigRat(2);
  const BigRat phi_squared = phi * phi;
  const BigRat phi_cubed = phi_squared * phi;
  const BigRat zero = BigRat(0);
  const BigRat one = BigRat(1);
  const BigRat neg_one = BigRat(-1);

  std::vector<BigVec3> vertices;
  for (int b = 0b000; b < 0b1000; b++) {
    BigRat s1 = (b & 0b100) ? neg_one : one;
    BigRat s2 = (b & 0b010) ? neg_one : one;
    BigRat s3 = (b & 0b001) ? neg_one : one;

    // (±1, ±1, ±φ^3),
    // (±φ^2, ±φ, ±2φ),
    AddEvenPermutations(s1, s2, s3 * phi_cubed, &vertices);
    AddEvenPermutations(s1 * phi_squared,
                        s2 * phi,
                        s3 * BigRat(2) * phi, &vertices);
  }

  for (int b = 0b00; b < 0b100; b++) {
    BigRat s1 = (b & 0b10) ? neg_one : one;
    BigRat s2 = (b & 0b01) ? neg_one : one;
    // (±(2+φ), 0, ±φ^2),
    AddEvenPermutations(s1 * (BigRat(2) + phi),
                        zero,
                        s2 * phi_squared, &vertices);
  }

  CHECK(vertices.size() == 60) << vertices.size();
  return MakeBigPolyFromVertices(std::move(vertices), "rhombicosidodecahedron");
}

// Is pt strictly within the triangle a-b-c? Works with both winding orders.
bool InTriangle(const BigVec2 &a, const BigVec2 &b, const BigVec2 &c,
                const BigVec2 &pt) {
  // The idea behind this test is that for each edge, we check
  // to see if the test point is on the same side as a reference
  // point, which is the third point of the triangle.
  auto SameSide = [](const BigVec2 &u, const BigVec2 &v,
                     const BigVec2 &p1, const BigVec2 &p2) {
      BigVec2 edge = v - u;
      BigRat c1 = cross(edge, p1 - u);
      BigRat c2 = cross(edge, p2 - u);

      int s1 = BigRat::Sign(c1);
      int s2 = BigRat::Sign(c2);

      // Note that this excludes the edge itself.
      return s1 != 0 && s2 != 0 && s1 == s2;
    };

  return SameSide(a, b, c, pt) &&
    SameSide(b, c, a, pt) &&
    SameSide(c, a, b, pt);
}

bool InMesh(const BigMesh2D &mesh, const BigVec2 &pt) {
  // using the triangulation
  for (const auto &[ai, bi, ci] : mesh.faces->triangulation) {
    const BigVec2 &a = mesh.vertices[ai];
    const BigVec2 &b = mesh.vertices[bi];
    const BigVec2 &c = mesh.vertices[ci];

    if (InTriangle(a, b, c, pt)) return true;
  }

  return false;
}

std::optional<std::tuple<int, int, int>>
InMeshExhaustive(const BigMesh2D &mesh, const BigVec2 &pt) {

  for (int ai = 0; ai < mesh.vertices.size(); ai++) {
    const BigVec2 &a = mesh.vertices[ai];
    for (int bi = 0; bi < ai; bi++) {
      const BigVec2 &b = mesh.vertices[bi];
      if (b == a) continue;
      for (int ci = 0; ci < bi; ci++) {
        const BigVec2 &c = mesh.vertices[ci];
        if (c == b || c == a) continue;

        // XXX check for degenerate triangles.
        if (InTriangle(a, b, c, pt)) return std::make_tuple(ai, bi, ci);
      }
    }
  }

  return std::nullopt;
}

int GetClosestPoint(const BigMesh2D &mesh, const BigVec2 &pt) {
  CHECK(!mesh.vertices.empty());
  BigRat best_sqdist = DistanceSquared(mesh.vertices[0], pt);
  int best_idx = 0;

  for (int i = 1; i < mesh.vertices.size(); i++) {
    const BigVec2 &a = mesh.vertices[i];
    BigRat sqdist = DistanceSquared(a, pt);
    if (sqdist < best_sqdist) {
      best_sqdist = std::move(sqdist);
      best_idx = i;
    }
  }

  return best_idx;
}

std::pair<int, BigRat> GetClosestPoint(const std::vector<BigVec2> &vertices,
                                       const std::vector<int> &hull,
                                       const BigVec2 &pt) {
  CHECK(!hull.empty());
  BigRat best_sqdist = DistanceSquared(vertices[hull[0]], pt);
  int best_idx = hull[0];

  for (int i = 1; i < hull.size(); i++) {
    int idx = hull[i];
    const BigVec2 &a = vertices[idx];
    BigRat sqdist = DistanceSquared(a, pt);
    if (sqdist < best_sqdist) {
      best_sqdist = std::move(sqdist);
      best_idx = idx;
    }
  }

  return std::make_pair(best_idx, std::move(best_sqdist));
}


// The QuickHull implementation below and its helper routines are based on code
// by Miguel Vieira (see LICENSES) although I have heavily modified it; see
// the floating point version in polyhedra.cc for more info.

// Returns the index of the farthest point from segment (a, b).
// Requires that all points are to the left of the segment (a,b) (or colinear).
static int GetFarthest(const BigVec2 &a, const BigVec2 &b,
                       const std::vector<BigVec2> &v,
                       const std::vector<int> &pts) {
  CHECK(!pts.empty());
  const BigRat dx = b.x - a.x;
  const BigRat dy = b.y - a.y;
  auto SqDist = [&](const BigVec2 &p) -> BigRat {
      return dx * (a.y - p.y) - dy * (a.x - p.x);
    };

  int best_idx = pts[0];
  BigRat best_dist = SqDist(v[best_idx]);

  for (int i = 1; i < pts.size(); i++) {
    int p = pts[i];
    BigRat d = SqDist(v[p]);
    if (d < best_dist) {
      best_idx = p;
      best_dist = std::move(d);
    }
  }

  return best_idx;
}


// The z-value of the cross product of segments
// (a, b) and (a, c). Positive means c is ccw (to the left)
// from (a, b), negative cw. Zero means it's colinear.
enum Orientation {
  CCW, COLINEAR, CW,
};
static Orientation GetOrientation(
    const BigVec2 &a, const BigVec2 &b, const BigVec2 &c) {
  int s = BigRat::Sign(cross(b - a, c - a));
  if (s < 0) return CW;
  if (s > 0) return CCW;
  return COLINEAR;
}

// Recursive call of the quickhull algorithm.
static void QuickHullRec(const std::vector<BigVec2> &vertices,
                         const std::vector<int> &pts,
                         int a, int b,
                         std::vector<int> *hull) {
  static constexpr bool SELF_CHECK = false;
  static constexpr int VERBOSE = 0;

  if (pts.empty()) {
    return;
  }

  if (SELF_CHECK) {
    for (int x : pts) {
      CHECK(x != a && x != b) << x << " candidate points should "
        "not include the endpoints of the recursed upon segment.";
      for (int y : *hull) {
        CHECK(x != y) << x << " is already in the hull!";
      }
    }
  }

  const BigVec2 &aa = vertices[a];
  const BigVec2 &bb = vertices[b];

  if (SELF_CHECK) {
    for (int x : pts) {
      const BigVec2 &xx = vertices[x];
      CHECK(aa == xx || bb == xx ||
            GetOrientation(aa, bb, xx) == CCW);
    }
  }

  int f = GetFarthest(aa, bb, vertices, pts);
  const BigVec2 &ff = vertices[f];
  if (VERBOSE) printf("Farthest is %d (%s)\n", f, VecString(ff).c_str());

  // Collect points to the left of segment (a, f) and to the left
  // of segment f, b (which we call "right"). A point cannot be in both
  // sets, because that would require it to be farther away than f, but f
  // is maximal.
  //
  //             f
  //       left / \   right
  //      set  /   \   set
  //          a     b
  //
  std::vector<int> left, right;
  for (int i : pts) {
    const BigVec2 &ii = vertices[i];
    // In the presence of exact duplicates for one of the endpoints,
    // we need to filter them out here or else we can end up in
    // infinite loops. Removing duplicates does not affect the hull.
    if (ii == aa || ii == bb || ii == ff) continue;

    if (GetOrientation(aa, ff, ii) == CCW) {
      left.push_back(i);
    } else if (GetOrientation(ff, bb, ii) == CCW) {
      right.push_back(i);
    }
  }
  if (VERBOSE) printf("%d left, %d right vertices.\n",
                      (int)left.size(), (int)right.size());
  QuickHullRec(vertices, left, a, f, hull);

  // Add f to the hull
  hull->push_back(f);

  if (VERBOSE) printf("%d right vertices.\n", (int)right.size());
  QuickHullRec(vertices, right, f, b, hull);
}

// QuickHull algorithm.
// https://en.wikipedia.org/wiki/QuickHull
std::vector<int> BigQuickHull(const std::vector<BigVec2> &vertices) {
  std::vector<int> hull;
  if (vertices.empty()) return {};
  if (vertices.size() == 1) return {0};
  if (vertices.size() == 2) return {0, 1};

  // Returns true if a is lexicographically before b.
  auto LeftOf = [](const BigVec2 &a, const BigVec2 &b) -> bool {
      return (a.x < b.x || (a.x == b.x && a.y < b.y));
    };

  // Get the leftmost (a) and rightmost (b) points.
  int a = 0, b = 0;
  for (int i = 1; i < (int)vertices.size(); i++) {
    if (LeftOf(vertices[i], vertices[a])) a = i;
    if (LeftOf(vertices[b], vertices[i])) b = i;
  }

  CHECK(a != b);

  // Split the points on either side of segment (a, b).
  std::vector<int> left, right;
  for (int i = 0; i < (int)vertices.size(); i++) {
    if (i != a && i != b) {
      Orientation side = GetOrientation(vertices[a], vertices[b], vertices[i]);
      if (side == CCW) left.push_back(i);
      else if (side == CW) right.push_back(i);
      // Ignore if colinear.
    }
  }

  // Be careful to add points to the hull
  // in the correct order. Add our leftmost point.
  hull.push_back(a);

  // Add hull points from the left (top)
  QuickHullRec(vertices, left, a, b, &hull);

  // Add our rightmost point
  hull.push_back(b);

  // Add hull points from the right (bottom)
  QuickHullRec(vertices, right, b, a, &hull);

  return hull;
}

bool InHull(const std::vector<BigVec2> &vertices,
            const std::vector<int> &hull,
            const BigVec2 &pt) {
  // No area.
  if (hull.size() < 3) return false;

  std::optional<Orientation> same_side;
  for (int i = 0; i < hull.size(); i++) {
    const BigVec2 &a = vertices[hull[i]];
    const BigVec2 &b = vertices[hull[(i + 1) % hull.size()]];

    Orientation side = GetOrientation(a, b, pt);
    if (side == COLINEAR) return false;
    if (!same_side.has_value()) {
      same_side = side;
    } else {
      if (same_side.value() != side) return false;
    }
  }
  // Always on the same side.
  return true;
}

// Return the closest point (to x,y) on the given line segment.
// It may be one of the endpoints.
BigRat SquaredDistanceToClosestPointOnSegment(
    // Line segment
    const BigVec2 &v0,
    const BigVec2 &v1,
    // Point to test
    const BigVec2 &pt) {
  const BigRat zero(0);
  const BigRat one(1);

  BigVec2 v = v1 - v0;

  BigRat sqlen = LengthSquared(v);
  if (sqlen == zero) {
    // Degenerate case where line segment is just a point,
    // so there is only one choice.
    return DistanceSquared(pt, v0);
  }

  BigVec2 w = pt - v0;
  BigRat t = BigRat::Max(zero, BigRat::Min(one, dot(v, w) / sqlen));

  BigVec2 closest = v0 + (v * t);
  return DistanceSquared(pt, closest);
}


BigRat SquaredDistanceToHull(const std::vector<BigVec2> &vertices,
                             const std::vector<int> &hull,
                             const BigVec2 &pt) {
  std::optional<BigRat> min_sqlen;
  for (int i = 0; i < hull.size(); i++) {
    const BigVec2 &a = vertices[hull[i]];
    const BigVec2 &b = vertices[hull[(i + 1) % hull.size()]];

    BigRat sqlen = SquaredDistanceToClosestPointOnSegment(a, b, pt);
    if (!min_sqlen.has_value() || sqlen < min_sqlen.value()) {
      min_sqlen = {std::move(sqlen)};
    }
  }

  CHECK(min_sqlen.has_value()) << "empty hull?";
  return min_sqlen.value();
}
