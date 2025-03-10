
#include "smallest-sphere.h"

#include <cmath>
#include <format>
#include <optional>
#include <span>
#include <tuple>
#include <utility>
#include <vector>
#include <yocto_matht.h>

#include "arcfour.h"
#include "base/logging.h"
#include "randutil.h"

using vec3 = yocto::vec<double, 3>;

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


// For [0..4] vertices. Beyond that, the circumsphere may
// not exist. You probably want SmallestSphere anyway.
static std::pair<vec3, double>
Circumsphere(const std::vector<vec3> &pts,
             const std::vector<int> &r_in) {

  // Remove nearly coincident points.
  std::vector<int> r;
  r.reserve(r_in.size());
  for (int i = 0; i < r_in.size(); i++) {
    int p1 = r_in[i];
    for (int j = 0; j < i; j++) {
      int p2 = r_in[j];
      CHECK(p1 != p2) << "Bug: Should not have literal duplicates.";
      if (distance(pts[i], pts[j]) < 1.0e-10) {
        goto skip_point;
      }
    }
    r.push_back(p1);

    skip_point:;
  }

  if (r.size() == 0) {
    return std::make_pair(vec3(0, 0, 0), 0.0);
  } else if (r.size() == 1) {
    return std::make_pair(pts[r[0]], 0.0);
  } else if (r.size() == 2) {
    vec3 o = (pts[r[0]] + pts[r[1]]) * 0.5;
    return std::make_pair(o, length(pts[r[1]] - o));
  } else if (r.size() == 3) {

    // a,b,c form a triangle, but we want the edge b-c to be
    // the (a) longest one.
    vec3 a = pts[r[0]];
    vec3 b = pts[r[1]];
    vec3 c = pts[r[2]];

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

  } else if (r.size() == 4) {

    // Then we know
    // (a - o)^2 = r^2
    // (b - o)^2 = r^2
    // (c - o)^2 = r^2
    // (d - o)^2 = r^2

    // We subtract one equation from the other 3 to get a 3x3 system.

    const vec3& a = pts[r[0]];
    const vec3& b = pts[r[1]];
    const vec3& c = pts[r[2]];
    const vec3& d = pts[r[3]];

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

    if (!sol.has_value()) {
      // Coplanar or coincident as above. Could handle this case by
      // trying subsets of the points.
      return std::make_pair(vec3(0, 0, 0), 0.0);
    }

    const auto &[x, y, z] = sol.value();
    vec3 o{x, y, z};
    double radius = distance(o, a);
    return std::make_pair(o, radius);

  } else {
    LOG(FATAL) << "Need four or fewer points.";
  }
}

static std::pair<vec3, double> SmallestSphereRec(
    const std::vector<vec3> &pts,
    // Points remaining
    std::span<const int> p,
    // Points on the boundary
    const std::vector<int> &r) {

  if (r.size() == 4) {
    // If four points are required to be on the sphere, then
    // we are done.
    return Circumsphere(pts, r);
  }

  if (p.empty()) {
    CHECK(r.size() <= 4) << "It should not be possible for "
      "more than four points to be required on the surface.";
    return Circumsphere(pts, r);
  }

  int pt = p[0];
  p = p.subspan(1);
  const auto &[o, radius] = SmallestSphereRec(pts, p, r);
  const vec3 &v = pts[pt];
  if (distance(v, o) > radius) {
    // then actually pt is on the smallest sphere. Add it
    // to r (unless it is "very close" to a point already).
    std::vector<int> rr = r;
    rr.push_back(pt);
    return SmallestSphereRec(pts, p, rr);
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
  Shuffle(rc, &p);
  return SmallestSphereRec(pts, p, {});
}
