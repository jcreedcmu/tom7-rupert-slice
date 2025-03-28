
#include "smallest-sphere.h"

#include <cmath>
#include <cstdio>
#include <format>
#include <optional>
#include <span>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include <yocto_matht.h>

#include "ansi.h"
#include "arcfour.h"
#include "base/logging.h"
#include "randutil.h"

using vec3 = yocto::vec<double, 3>;

// bool SmallestSphere::verbose = false;
// #define VERBOSE SmallestSphere::verbose
static constexpr bool VERBOSE = false;

static constexpr bool SELF_CHECK = true;

// Since yocto uses column major, we expand the arguments
// to these to avoid confusion.

// Returns (a, b, c) such that
// [ m00  m01  m02 ] [ a ]   [ v0 ]
// [ m10  m11  m12 ] [ b ] = [ v1 ]
// [ m20  m21  m22 ] [ c ]   [ v2 ]
// For singular matrices, nullopt.
[[maybe_unused]]
static std::optional<std::tuple<double, double>>
Solve22(double m00, double m01,
        double m10, double m11,
        double v0, double v1) {

  double d = (m00 * m11 - m01 * m10);
  if (std::abs(d) < 1.0e-20) return std::nullopt;
  return std::make_tuple((m11 * v0 - m01 * v1) / d,
                         (m00 * v1 - m10 * v0) / d);
}

static inline double Det33(double m00, double m01, double m02,
                           double m10, double m11, double m12,
                           double m20, double m21, double m22) {
  return (m00 * m11 * m22 + m01 * m12 * m20 +
          m10 * m21 * m02 - m02 * m11 * m20 -
          m00 * m12 * m21 - m01 * m10 * m22);
}

static double Det44(double m00, double m01, double m02, double m03,
                    double m10, double m11, double m12, double m13,
                    double m20, double m21, double m22, double m23,
                    double m30, double m31, double m32, double m33) {

  return
    m00 * Det33(m11, m12, m13,
                m21, m22, m23,
                m31, m32, m33) -
    m01 * Det33(m10, m12, m13,
                m20, m22, m23,
                m30, m32, m33) +
    m02 * Det33(m10, m11, m13,
                m20, m21, m23,
                m30, m31, m33) -
    m03 * Det33(m10, m11, m12,
                m20, m21, m22,
                m30, m31, m32);
}

// Returns (a, b, c) such that
// [ m00  m01  m02 ] [ a ]   [ v0 ]
// [ m10  m11  m12 ] [ b ] = [ v1 ]
// [ m20  m21  m22 ] [ c ]   [ v2 ]
// For singular matrices, nullopt.
[[maybe_unused]]
static std::optional<std::tuple<double, double, double>>
Solve33(double m00, double m01, double m02,
        double m10, double m11, double m12,
        double m20, double m21, double m22,
        double v0, double v1, double v2) {

  double d = Det33(m00, m01, m02,
                   m10, m11, m12,
                   m20, m21, m22);
  if (std::abs(d) < 1.0e-20) return std::nullopt;

  double d0 = Det33(v0, m01, m02,
                    v1, m11, m12,
                    v2, m21, m22);
  double d1 = Det33(m00, v0, m02,
                    m10, v1, m12,
                    m20, v2, m22);
  double d2 = Det33(m00, m01, v0,
                    m10, m11, v1,
                    m20, m21, v2);
  return {std::make_tuple(d0 / d, d1 / d, d2 / d)};
}

// Same idea for 4x4 system.
[[maybe_unused]]
static std::optional<std::tuple<double, double, double, double>>
Solve44(double m00, double m01, double m02, double m03,
        double m10, double m11, double m12, double m13,
        double m20, double m21, double m22, double m23,
        double m30, double m31, double m32, double m33,
        double v0, double v1, double v2, double v3) {

  double d = Det44(m00, m01, m02, m03,
                   m10, m11, m12, m13,
                   m20, m21, m22, m23,
                   m30, m31, m32, m33);
  if (std::abs(d) < 1.0e-20) return std::nullopt;

  double d0 = Det44(v0, m01, m02, m03,
                    v1, m11, m12, m13,
                    v2, m21, m22, m23,
                    v3, m31, m32, m33);

  double d1 = Det44(m00, v0, m02, m03,
                    m10, v1, m12, m13,
                    m20, v2, m22, m23,
                    m30, v3, m32, m33);

  double d2 = Det44(m00, m01, v0, m03,
                    m10, m11, v1, m13,
                    m20, m21, v2, m23,
                    m30, m31, v3, m33);

  double d3 = Det44(m00, m01, m02, v0,
                    m10, m11, m12, v1,
                    m20, m21, m22, v2,
                    m30, m31, m32, v3);

  return std::make_tuple(d0 / d, d1 / d, d2 / d, d3 / d);
}

// This is like vector<vec3>, but it can only hold up to
// four elements. Moreover, the points are a proper simplex
// (no close duplicates; if four, they are not coplanar; etc.)
namespace {
struct SmallSimplex {
  size_t size() const { return num; }
  const vec3 *data() const { return pts; }
  const vec3 &operator[](size_t i) const {
    if (SELF_CHECK) {
      CHECK(i < num);
    }
    return pts[i];
  }

  SmallSimplex() {}
  SmallSimplex(const SmallSimplex &other) = default;
  SmallSimplex &operator =(const SmallSimplex &other) = default;

  void Push(const vec3 &pt) {
    switch (num) {
    case 0: {
      pts[0] = pt;
      num = 1;
      break;
    }

    case 1: {
      if (TooClose(pts[0], pt))
        break;

      pts[1] = pt;
      num = 2;
      break;
    }

    case 2: {
      if (TooClose(pts[0], pt) ||
          TooClose(pts[1], pt))
        break;

      // XXX check colinearity.

      pts[2] = pt;
      num = 3;
      break;
    }

    case 3: {
      if (TooClose(pts[0], pt) ||
          TooClose(pts[1], pt) ||
          TooClose(pts[2], pt))
        break;

      // Check that they are not coplanar.
      // PERF: Can store cross product to make this
      // cheaper.
      auto v1 = pts[1] - pts[0];
      auto v2 = pts[2] - pts[0];
      auto v3 = pt - pts[0];
      double scalar_triple_product = dot(v1, cross(v2, v3));
      if (std::abs(scalar_triple_product) < flatness_threshold) {
        // Don't add the point.
        // TODO: We could replace a point if it makes the
        // triangle less narrow.
        break;
      }
      pts[3] = pt;
      num = 4;
      break;
    }

    case 4: {
      // Ignore the point, since we already have a tetrahedron.
      // TODO: We could replace a point if it makes the tetrahedron
      // less narrow.
      break;
    }

    default:
      LOG(FATAL) << "Invalid state.";
      break;
    }
  }

  // Points that are closer than this are considered duplicates.
  static constexpr double distance_threshold = 1.0e-10;
  static constexpr double flatness_threshold = 1.0e-10;
  static bool TooClose(const vec3 &a, const vec3 &b) {
    // PERF avoid square roots
    return distance(a, b) < distance_threshold;
  }

 private:
  vec3 pts[4] = {};
  size_t num = 0;
};
}

// For [0..4] vertices. Beyond that, the circumsphere may
// not exist. You probably want SmallestSphere anyway.
static std::pair<vec3, double>
Circumsphere(const SmallSimplex &simplex) {

  switch (simplex.size()) {
  case 0:
    return std::make_pair(vec3(0, 0, 0), 0.0);

  case 1:
    return std::make_pair(simplex[0], 0.0);

  case 2: {
    vec3 o = (simplex[0] + simplex[1]) * 0.5;
    return std::make_pair(o, distance(simplex[1], o));
  }

  case 3: {
    // a,b,c form a triangle, but we want the edge b-c to be
    // the (a) longest one.
    vec3 a = simplex[0];
    vec3 b = simplex[1];
    vec3 c = simplex[2];

    {
      double dist_ab = distance_squared(a, b);
      double dist_bc = distance_squared(b, c);
      double dist_ca = distance_squared(c, a);

      if (dist_ab >= dist_bc && dist_ab >= dist_ca) {
        // ab is longest. Make it bc.
        std::swap(a, c);
      } else if (dist_ca >= dist_bc && dist_ca >= dist_ab) {
        // ca is longest. Make it bc.
        std::swap(a, b);
      } else {
        // bc was already the longest.
      }
    }

    // Two angles
    const vec3 v1 = b - a;
    const vec3 v2 = c - a;

    vec3 normal = cross(v1, v2);
    double len_normal = length(normal);
    if (len_normal < 1.0e-10) {
      // Colinear or coincident points.
      // TODO: There is no "circumsphere", but we could
      // remove a duplicate (or the center point)
      // and treat this as a smaller set of points...
      if (VERBOSE) { printf("Colinear points.\n"); }
      return std::make_pair(vec3(0, 0, 0), 0.0);
    }

    vec3 unormal = normal / len_normal;

    const vec3 midab = (a + b) * 0.5;
    const vec3 midac = (a + c) * 0.5;

    // Bisectors
    const vec3 b1 = cross(v1, unormal);
    const vec3 b2 = cross(v2, unormal);

    // now the center o = midab + t * b1
    // and also       o = midac + s * b2
    // so we have
    // t * b1 - s * b2 = midac - midab
    // Take this equation dot b1, and dot b2, to
    // get the following system
    //
    // |b1 dot b1   -b2 dot b1| |t|   | (midac - midab) dot b1 |
    // |b1 dot b2   -b2 dot b2| |s| = | (midac - midab) dot b2 |

    auto dotb1b2 = dot(b1, b2);
    auto mm = midac - midab;

    auto sol =
      Solve22(dot(b1, b1), -dotb1b2,
              dotb1b2,     -dot(b2, b2),
              dot(mm, b1),
              dot(mm, b2));

    CHECK(sol.has_value()) <<
      "There should be a solution whenever the points are "
      "not colinear! The points were:\n" <<
      std::format("a: {}, {}, {}\n"
                  "b: {}, {}, {}\n"
                  "c: {}, {}, {}\n"
                  "and we had\n"
                  "v1: {}, {}, {}\n"
                  "v2: {}, {}, {}\n"
                  "normal: {}, {}, {}\n"
                  "length(normal): {}\n"
                  "unormal: {}, {}, {}\n"
                  "length(unormal): {}\n",
                  a.x, a.y, a.z,
                  b.x, b.y, b.z,
                  c.x, c.y, c.z,
                  v1.x, v1.y, v1.z,
                  v2.x, v2.y, v2.z,
                  normal.x, normal.y, normal.z,
                  length(normal),
                  unormal.x, unormal.y, unormal.z,
                  length(unormal));

    const auto &[t, s] = sol.value();
    vec3 o = midab + t * b1;

    double r = distance(o, a);
    return {std::make_pair(o, r)};
  }

  case 4: {

    // Then we know
    // (a - o)^2 = r^2
    // (b - o)^2 = r^2
    // (c - o)^2 = r^2
    // (d - o)^2 = r^2

    // We subtract one equation from the other 3 to get a 3x3 system.

    const vec3 &a = simplex[0];
    const vec3 &b = simplex[1];
    const vec3 &c = simplex[2];
    const vec3 &d = simplex[3];
    if (VERBOSE) {
      printf("(%.5g, %.5g, %.5g)\n", a.x, a.y, a.z);
      printf("(%.5g, %.5g, %.5g)\n", b.x, b.y, b.z);
      printf("(%.5g, %.5g, %.5g)\n", c.x, c.y, c.z);
      printf("(%.5g, %.5g, %.5g)\n", d.x, d.y, d.z);
    }

    if (SELF_CHECK) {
      // If all four are coplanar, then the tetrahedron is degenerate,
      // so don't use that method.
      auto v1 = b - a;
      auto v2 = c - a;
      auto v3 = d - a;
      double scalar_triple_product = dot(v1, cross(v2, v3));
      CHECK(std::abs(scalar_triple_product) >= 1.0e-10);
    }

    double m00 = 2 * (b.x - a.x);
    double m01 = 2 * (b.y - a.y);
    double m02 = 2 * (b.z - a.z);
    double v0 =
      b.x * b.x - a.x * a.x + b.y * b.y - a.y * a.y + b.z * b.z - a.z * a.z;

    double m10 = 2 * (c.x - a.x);
    double m11 = 2 * (c.y - a.y);
    double m12 = 2 * (c.z - a.z);
    double v1 =
      c.x * c.x - a.x * a.x + c.y * c.y - a.y * a.y + c.z * c.z - a.z * a.z;

    double m20 = 2 * (d.x - a.x);
    double m21 = 2 * (d.y - a.y);
    double m22 = 2 * (d.z - a.z);
    double v2 =
      d.x * d.x - a.x * a.x + d.y * d.y - a.y * a.y + d.z * d.z - a.z * a.z;

    auto sol = Solve33(m00, m01, m02,
                       m10, m11, m12,
                       m20, m21, m22,
                       v0, v1, v2);

    if (SELF_CHECK) {
      CHECK(sol.has_value()) << "Since the simplex is not degenerate, "
        "this should have a solution. But it might come from a "
        "disagreement about 'epsilon'.";
    } else {
      if (!sol.has_value()) {
        // Coplanar or coincident as above. Could handle this case by
        // trying subsets of the points.
        return std::make_pair(vec3(0, 0, 0), 0.0);
      }
    }

    const auto &[x, y, z] = sol.value();
    vec3 o{x, y, z};
    double radius = distance(o, a);
    return std::make_pair(o, radius);
  }

  default:
    LOG(FATAL) << "Bad simplex";
  }
}

static std::string Indent(int depth) {
  std::string s;
  while (depth--) s += AGREY("▏ ");
  return s; // std::string(depth * 2, ' ');
}

static std::pair<vec3, double> SmallestSphereRec(
    ArcFour *rc,
    int depth,
    const std::vector<vec3> &pts,
    // Points remaining in arbitrary order.
    // We modify what this span points to, but only by reordering.
    std::span<int> p,
    // Points on the boundary
    const SmallSimplex &simplex) {

  if (VERBOSE) {
    printf("%sSmallest (",
           Indent(depth).c_str());
    for (int x : p) printf("%d", x);
    printf(", ");
    for (int i = 0; i < simplex.size(); i++) {
      const vec3 &v = simplex[i];
      printf("(%.3f,%.3f,%.3f)", v.x, v.y, v.z);
    }
    printf(")\n");
  }

  // It is possible to stop early if we have a proper
  // simplex with four points.
  if (simplex.size() == 4) {
    if (VERBOSE) {
      printf("%sHave four points:\n",
             Indent(depth).c_str());
      for (int i = 0; i < simplex.size(); i++) {
        const vec3 &v = simplex[i];
        printf("%s(%.3f,%.3f,%.3f)", Indent(depth).c_str(), v.x, v.y, v.z);
      }
    }

    // If four points are required to be on the sphere, then
    // we are done.
    return Circumsphere(simplex);
  }

  if (p.empty()) {
    #if 0
    CHECK(r.size() <= 4) << "It should not be possible for "
      "more than four points to be required on the surface.";
    if (VERBOSE) {
      printf("%sHave %d points:",
             Indent(depth).c_str(),
             (int)r.size());
      for (int x : r) printf(" %d", x);
      printf("\n");
    }
    #endif

    return Circumsphere(simplex);
  }

  // Pick a point uniformly at random and put it first.
  {
    CHECK(!p.empty());
    int pidx = RandTo(rc, p.size());
    std::swap(p[0], p[pidx]);
  }

  int pt = p[0];
  p = p.subspan(1);
  const auto &[o, radius] = SmallestSphereRec(rc, depth + 1, pts, p, simplex);
  const vec3 &v = pts[pt];
  if (distance(v, o) > radius) {
    if (VERBOSE) {
      double excess = distance(v, o) - radius;
      printf("%sPt %d outside sphere((%.5g, %.5g, %.5g), %.6g) by %.5g\n",
             Indent(depth).c_str(),
             pt, o.x, o.y, o.z, radius,
             excess);
    }
    // then actually pt is on the smallest sphere. Add it
    // to r (unless it is "very close" to a point already).
    SmallSimplex ssimplex = simplex;
    ssimplex.Push(v);
    return SmallestSphereRec(rc, depth + 1, pts, p, ssimplex);
  } else {
    return std::make_pair(o, radius);
  }
}

[[maybe_unused]]
std::pair<vec3, double> SmallestSphere::Smallest(
    ArcFour *rc,
    const std::vector<vec3> &pts) {
  std::vector<int> p;
  p.reserve(pts.size());
  for (int i = 0; i < pts.size(); i++) p.push_back(i);
  // PERF: Shuffling at the beginning instead of repeatedly
  // choosing a random index will generate fewer random numbers,
  // but I saw unlucky cases that took a very long time (especially
  // with a bunch of cubes; see TestSlow). Stopping early when
  // the simplex has size 4 might have addressed this, so I
  // might be able to switch back here.
  // Shuffle(rc, &p);

  if (VERBOSE) {
    printf("----------------\n");
    for (int i = 0; i < pts.size(); i++) {
      printf("%d. (%.11g, %.11g, %.11g)\n",
             i, pts[i].x, pts[i].y, pts[i].z);
    }
  }

  SmallSimplex simplex;
  auto sphere = SmallestSphereRec(rc, 0, pts, p, simplex);
  const auto &[o, radius] = sphere;
  if (VERBOSE) {
    printf(AYELLOW("final") ": sphere((%.11g, %.11g, %.11g), %.11g)\n",
           o.x, o.y, o.z, radius);
  }

  return sphere;
}
