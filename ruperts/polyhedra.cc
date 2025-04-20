
#include "polyhedra.h"

#include <format>
#include <limits>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numbers>
#include <optional>
#include <string>
#include <cstdint>
#include <utility>
#include <vector>

#include "ansi.h"
#include "arcfour.h"
#include "base/logging.h"
#include "base/stringprintf.h"
#include "hashing.h"
#include "hull3d.h"
#include "mesh.h"
#include "randutil.h"
#include "set-util.h"
#include "util.h"

#include "yocto_matht.h"

// XXX for debugging
#include "rendering.h"

[[maybe_unused]]
static double consteval SqrtNewtons(double x, double cur, double prev) {
  return cur == prev
    ? cur
    : SqrtNewtons(x, 0.5 * (cur + x / cur), cur);
}

static bool consteval IsFinite(double x) {
  return x == x && x < std::numeric_limits<double>::infinity() &&
                       x > -std::numeric_limits<double>::infinity();
}

[[maybe_unused]]
static double consteval Abs(double x) {
  return x < 0.0 ? -x : x;
}

template<class F, class DF>
static double consteval Solve(double initial,
                              const F &f,
                              const DF &df) {
  long double x = initial;
  long double fx = f(x);
  for (;;) {
    long double dfx = df(x);

    long double xn = x - fx / dfx;
    long double fxn = f(xn);

    if (Abs(fxn) >= Abs(fx)) {
      // If the solution is no longer getting closer, then
      // return the best solution.
      return x;
    }

    x = xn;
    fx = fxn;
  }
}

// Compile-time sqrt using Newton's method.
static double consteval Sqrt(double u) {
  return u >= 0.0 && IsFinite(u) ?
    SqrtNewtons(u, u, 0.0) :
    0.0 / 0.0;
}

[[maybe_unused]]
static double consteval SqrtSolve(double u) {
  return u >= 0.0 && IsFinite(u) ?
    Solve(u,
          // solve f(x) = u - x^2
          [u](double x) { return u - x * x; },
          // derivative is -2x
          [](double x) { return -2.0 * x; }) :
    0.0 / 0.0;
}

static_assert(Sqrt(4.0) == 2.0, "Sqrt does not work!");
static_assert(Sqrt(1.0) == 1.0, "Sqrt does not work!");

// The Solve approach doesn't work, probably because the convergence
// criteria is wrong? Or when the initial guess is bad?
/*
static constexpr double d = 0.05684874075611912;
static_assert(SqrtSolve(d) == Sqrt(d));
*/

// Note that Sqrt is slightly wrong (probably one ulp?):
/*
static_assert(Sqrt(2.0) == std::numbers::sqrt2,
              "Sqrt does not work!");
*/

static double consteval CbrtNewtons(double x, double cur, double prev) {
  return cur == prev
    ? cur
    : CbrtNewtons(x, (2.0 * cur + x / (cur * cur)) / 3.0, cur);
}

// Compile-time sqrt using Newton's method.
static double consteval Cbrt(double x) {
  return IsFinite(x) ? CbrtNewtons(x, x, 0.0) : 0.0 / 0.0;
}

std::string VecString(const vec3 &v) {
  return StringPrintf(
      "(" ARED("%.4f") "," AGREEN("%.4f") "," ABLUE("%.4f") ")",
      v.x, v.y, v.z);
}

std::string VecString(const vec2 &v) {
  return StringPrintf(
      "(" ARED("%.4f") "," AGREEN("%.4f") ")",
      v.x, v.y);
}

std::string Points2DString(const std::vector<vec2> &v) {
  std::vector<std::string> s;
  for (const vec2 &pt : v) s.push_back(VecString(pt));
  return std::format("[{}]", Util::Join(s, ", "));
}

std::string QuatString(const quat4 &q) {
  return StringPrintf(
      "quat4{.x = %.17g,\n"
      "      .y = %.17g,\n"
      "      .z = %.17g,\n"
      "      .w = %.17g}",
      q.x, q.y, q.z, q.w);
}

std::string FrameString(const frame3 &f) {
  return StringPrintf(
      "frame3{.x = vec3(%.17g, %.17g, %.17g),\n"
      "       .y = vec3(%.17g, %.17g, %.17g),\n"
      "       .z = vec3(%.17g, %.17g, %.17g),\n"
      "       .o = vec3(%.17g, %.17g, %.17g)}",
      f.x.x, f.x.y, f.x.z,
      f.y.x, f.y.y, f.y.z,
      f.z.x, f.z.y, f.z.z,
      f.o.x, f.o.y, f.o.z);
}

std::string FormatNum(uint64_t n) {
  if (n > 1'000'000) {
    double m = n / 1'000'000.0;
    if (m >= 1'000'000.0) {
      return StringPrintf("%.1fT", m / 1'000'000.0);
    } else if (m >= 1000.0) {
      return StringPrintf("%.1fB", m / 1000.0);
    } else if (m >= 100.0) {
      return StringPrintf("%dM", (int)std::round(m));
    } else if (m > 10.0) {
      return StringPrintf("%.1fM", m);
    } else {
      // TODO: Integer division. color decimal place and suffix.
      return StringPrintf("%.2fM", m);
    }
  } else {
    return Util::UnsignedWithCommas(n);
  }
}

template<typename T>
inline int sgn(T val) {
  return (T(0) < val) - (val < T(0));
}

double TriangleSignedDistance(vec2 p0, vec2 p1, vec2 p2, vec2 p) {
  // This function only:
  // Derived from code by Inigo Quilez; MIT license. See LICENSES.

  vec2 e0 = p1 - p0;
  vec2 e1 = p2 - p1;
  vec2 e2 = p0 - p2;

  vec2 v0 = p - p0;
  vec2 v1 = p - p1;
  vec2 v2 = p - p2;

  vec2 pq0 = v0 - e0 * std::clamp(dot(v0, e0) / dot(e0, e0), 0.0, 1.0);
  vec2 pq1 = v1 - e1 * std::clamp(dot(v1, e1) / dot(e1, e1), 0.0, 1.0);
  vec2 pq2 = v2 - e2 * std::clamp(dot(v2, e2) / dot(e2, e2), 0.0, 1.0);

  double s = e0.x * e2.y - e0.y * e2.x;
  vec2 d = min(min(vec2(dot(pq0, pq0), s * (v0.x * e0.y - v0.y * e0.x)),
                   vec2(dot(pq1, pq1), s * (v1.x * e1.y - v1.y * e1.x))),
               vec2(dot(pq2, pq2), s * (v2.x * e2.y - v2.y * e2.x)));

  return -sqrt(d.x) * sgn(d.y);
}

bool InTriangle(const vec2 &a, const vec2 &b, const vec2 &c,
                const vec2 &pt) {
  // The idea behind this test is that for each edge, we check
  // to see if the test point is on the same side as a reference
  // point, which is the third point of the triangle.
  auto SameSide = [](const vec2 &u, const vec2 &v,
                     const vec2 &p1, const vec2 &p2) {
      vec2 edge = v - u;
      double c1 = cross(edge, p1 - u);
      double c2 = cross(edge, p2 - u);

      int s1 = sgn(c1);
      int s2 = sgn(c2);

      // Note that this excludes the edge itself.
      return s1 != 0 && s2 != 0 && s1 == s2;
    };

  return SameSide(a, b, c, pt) &&
    SameSide(b, c, a, pt) &&
    SameSide(c, a, b, pt);
}

Faces *Faces::Create(int num_vertices, std::vector<std::vector<int>> v) {

  Faces *faces = new Faces;
  if (faces->Init(num_vertices, std::move(v))) {
    return faces;
  } else {
    delete faces;
    return nullptr;
  }
}

bool Faces::Init(int num_vertices, std::vector<std::vector<int>> v_in) {
  static constexpr int VERBOSE = 0;
  v = std::move(v_in);

  std::vector<std::unordered_set<int>> collated(num_vertices);
  for (const std::vector<int> &face : v) {
    if (VERBOSE) {
      printf("Face:");
      for (int i = 0; i < (int)face.size(); i++) {
        printf(" %d", face[i]);
      }
      printf("\n");
    }
    // Add each edge forwards and backwards.
    for (int i = 0; i < (int)face.size(); i++) {
      int v0 = face[i];
      int v1 = face[(i + 1) % face.size()];
      CHECK(v0 >= 0 && v0 < num_vertices);
      CHECK(v1 >= 0 && v1 < num_vertices);
      collated[v0].insert(v1);
      collated[v1].insert(v0);
    }
  }

  // Now flatten into vector.
  neighbors.resize(num_vertices);
  for (int i = 0; i < (int)collated.size(); i++) {
    neighbors[i] = SetToSortedVec(collated[i]);
    // printf("#%d has %d neighbors\n", i, (int)neighbors[i].size());
    // e.g. if there are points that are not on faces. One cause
    // of this would be if the convex hull and facetization disagree
    // on epsilon, and so there are disconnected points.
    if (neighbors[i].empty()) {
      if (VERBOSE) {
        printf("Point %d is not on any face\n", i);
      }
      return false;
    }
  }

  // And triangulate. Since the faces are convex, we can
  // just do this by creating triangle fans.
  for (const std::vector<int> &face : v) {
    if (face.size() < 3) return false;
    int p0 = face[0];
    for (int i = 1; i + 1 < face.size(); i++) {
      triangulation.emplace_back(p0, face[i], face[i + 1]);
    }
  }

  return true;
}

Faces::Faces(int num_vertices, std::vector<std::vector<int>> v_in) {
  CHECK(Init(num_vertices, std::move(v_in))) << num_vertices;
}


// Return the closest point (to x,y) on the given line segment.
// It may be one of the endpoints.
static vec2 ClosestPointOnSegment(
    // Line segment
    const vec2 &v0,
    const vec2 &v1,
    // Point to test
    const vec2 &pt) {

  auto SqDist = [](double x0, double y0,
                   double x1, double y1) {
      const double dx = x1 - x0;
      const double dy = y1 - y0;
      return dx * dx + dy * dy;
    };

  const double sqlen = SqDist(v0.x, v0.y, v1.x, v1.y);
  if (sqlen == 0.0) {
    // Degenerate case where line segment is just a point,
    // so there is only one choice.
    return v0;
  }

  const double tf =
    ((pt.x - v0.x) * (v1.x - v0.x) + (pt.y - v0.y) * (v1.y - v0.y)) / sqlen;
  // Make sure it is on the segment.
  const double t = std::max(0.0, std::min(1.0, tf));
  // Closest point, which is on the segment.

  const double xx = v0.x + t * (v1.x - v0.x);
  const double yy = v0.y + t * (v1.y - v0.y);
  return vec2{xx, yy};
}

// Return the minimum distance between the point and the line segment.
inline double SquaredPointLineDistance(
    // Line segment
    const vec2 &v0, const vec2 &v1,
    // Point to test
    const vec2 &pt) {

  const vec2 c = ClosestPointOnSegment(v0, v1, pt);
  const double dx = pt.x - c.x;
  const double dy = pt.y - c.y;
  return dx * dx + dy * dy;
}

inline double PointLineDistance(
    // Line segment
    const vec2 &v0, const vec2 &v1,
    // Point to test
    const vec2 &pt) {
  return sqrt(SquaredPointLineDistance(v0, v1, pt));
}

[[maybe_unused]]
double BuggyDistanceToEdge(const vec2 &v0, const vec2 &v1, const vec2 &p) {
  vec2 edge = v1 - v0;
  vec2 p_edge = p - v0;
  double cx = yocto::cross(edge, p_edge);
  double dist = std::abs(cx / yocto::length(edge));
  double d0 = yocto::length(p_edge);
  double d1 = yocto::length(p - v1);

  return std::min(std::min(d0, d1), dist);
}

// Create the shadow of the polyhedron on the x-y plane.
Mesh2D Shadow(const Polyhedron &p) {
  Mesh2D mesh;
  mesh.vertices.resize(p.vertices.size());
  for (int i = 0; i < p.vertices.size(); i++) {
    const vec3 &v = p.vertices[i];
    mesh.vertices[i] = vec2{.x = v.x, .y = v.y};
  }
  mesh.faces = p.faces;
  return mesh;
}

static double SquaredDistanceToHull(
    const std::vector<vec2> &points, const std::vector<int> &hull,
    const vec2 &pt) {
  std::optional<double> best_sqdist;
  for (int i = 0; i < hull.size(); i++) {
    const vec2 &v0 = points[hull[i]];
    const vec2 &v1 = points[hull[(i + 1) % hull.size()]];

    double sqdist = SquaredPointLineDistance(v0, v1, pt);
    if (!best_sqdist.has_value() || sqdist < best_sqdist.value()) {
      best_sqdist = {sqdist};
    }
  }
  CHECK(best_sqdist.has_value());
  return best_sqdist.value();
}

double DistanceToHull(
    const std::vector<vec2> &points, const std::vector<int> &hull,
    const vec2 &pt) {
  return sqrt(SquaredDistanceToHull(points, hull, pt));
}

std::pair<vec2, double> ClosestPointOnHull(
    const std::vector<vec2> &points, const std::vector<int> &hull,
    const vec2 &pt) {
  std::optional<double> best_sqdist;
  vec2 best;
  for (int i = 0; i < hull.size(); i++) {
    const vec2 &v0 = points[hull[i]];
    const vec2 &v1 = points[hull[(i + 1) % hull.size()]];

    const vec2 o = ClosestPointOnSegment(v0, v1, pt);
    double sqdist = distance_squared(o, pt);
    if (!best_sqdist.has_value() || sqdist < best_sqdist.value()) {
      best_sqdist = {sqdist};
      best = o;
    }
  }
  CHECK(best_sqdist.has_value());
  return {best, std::sqrt(best_sqdist.value())};
}


double DistanceToMesh(const Mesh2D &mesh, const vec2 &pt) {
  std::optional<double> best_sqdist;
  for (const std::vector<int> &polygon : mesh.faces->v) {
    double sqdist = SquaredDistanceToHull(mesh.vertices, polygon, pt);
    if (!best_sqdist.has_value() || sqdist < best_sqdist.value()) {
      best_sqdist = {sqdist};
    }
  }
  CHECK(best_sqdist.has_value());
  return sqrt(best_sqdist.value());
}

double HullClearance(const std::vector<vec2> &outer_points,
                     const std::vector<int> &outer_hull,
                     const std::vector<vec2> &inner_points,
                     const std::vector<int> &inner_hull) {
  // The minimum distance must be between a vertex on one and
  // an edge on the other.

  double min_sqdist = std::numeric_limits<double>::infinity();
  for (int i = 0; i < inner_hull.size(); i++) {
    const vec2 &i1 = inner_points[inner_hull[i]];
    const vec2 &i2 = inner_points[inner_hull[(i + 1) % inner_hull.size()]];
    for (int o = 0; o < outer_hull.size(); o++) {
      const vec2 &o1 = outer_points[outer_hull[o]];
      const vec2 &o2 = outer_points[outer_hull[(o + 1) % outer_hull.size()]];

      double di = SquaredPointLineDistance(i1, i2, o1);
      double ii = SquaredPointLineDistance(o1, o2, i1);
      min_sqdist = std::min(min_sqdist, std::min(di, ii));
    }
  }

  return std::sqrt(min_sqdist);
}


std::vector<int> GiftWrapConvexHull(const std::vector<vec2> &vertices) {
  constexpr bool VERBOSE = false;
  constexpr bool SELF_CHECK = false;
  CHECK(vertices.size() > 2);

  auto ColorIndex = [](int i) {
      return StringPrintf(
          "%s%d" ANSI_RESET,
          ANSI::ForegroundRGB32(Rendering::Color(i)).c_str(),
          i);
    };

  // Explicitly mark vertices as used to avoid reusing them. This may
  // not actually be necessary (I think the real issue was that I used
  // to always start with "next" being node cur+1, even if that was an
  // invalid choice) but it is pretty cheap and colinear/coincident
  // points can cause tests to behave in countergeometric ways.
  std::vector<bool> used(vertices.size(), false);

  // Find the starting point. This must be a point on
  // the convex hull. The leftmost bottommost point is
  // one.
  const int start = [&]() {
      int besti = 0;
      for (int i = 1; i < vertices.size(); i++) {
        if ((vertices[i].y < vertices[besti].y) ||
            (vertices[i].y == vertices[besti].y &&
             vertices[i].x < vertices[besti].x)) {
          besti = i;
        }
      }
      return besti;
    }();

  if (VERBOSE) {
    printf("Start idx: %s\n", ColorIndex(start).c_str());

    for (const vec2 &v : vertices) {
      printf("vec2{%.17g, %.17g}, ", v.x, v.y);
    }
  }

  const vec2 &vstart = vertices[start];

  if (VERBOSE) {
    printf("\n");
  }

  std::vector<int> hull;
  int cur = start;
  do {
    if (VERBOSE) {
      printf("Loop with cur=%s\n", ColorIndex(cur).c_str());
    }

    if (SELF_CHECK) {
      for (int a : hull) {
        if (a == cur) {
          fprintf(stderr, "About to add duplicate point %d to hull.\n"
                  "Points so far:\n",
                  cur);
          for (int i = 0; i < (int)vertices.size(); i++) {
            fprintf(stderr, "%d. vec2{%.17g, %.17g}\n",
                    i, vertices[i].x, vertices[i].y);
          }
          fprintf(stderr, "Hull so far:");
          for (int x : hull) {
            fprintf(stderr, " %d%s", x,
                    used[x] ? " used" : ARED(" not used??"));
          }
          fprintf(stderr, "\n");
          LOG(FATAL) << "Infinite loop!";
        }
      }
    }

    hull.push_back(cur);
    used[cur] = true;

    // We consider every other point, finding the one with
    // the smallest angle from the current point.

    // We need to choose a point that's not already used (unless
    // we're closing the loop). So start is a good choice, except
    // on the first step (because then the start-start edge is
    // degenerate).
    // int next = (cur == start) ? ((cur + 1) % vertices.size()) :
    // start;
    int next = -1;

    // First, find any point that's unused and not exactly the same as
    // the current point.
    for (int i = 0; i < vertices.size(); i++) {
      if (i != cur && (i == start || !used[i]) &&
          vertices[i] != vertices[cur]) {
        next = i;
        break;
      }
    }

    // We exhausted all of the nodes, so we must be done.
    if (next == -1) {
      if (VERBOSE) {
        printf(ACYAN("No more nodes.") "\n");
      }
      return hull;
    }

    for (int i = 0; i < vertices.size(); i++) {
      if (VERBOSE) {
        printf("Inner loop at i=%s w/ next=%s.\n",
               ColorIndex(i).c_str(),
               ColorIndex(next).c_str());
      }
      // We need to consider the start point as a candidate
      // (which will always have been marked 'used') because
      // it's how we actually end. It prevents us from choosing
      // invalid points (that are not "to the left").
      if ((i == start || !used[i]) && i != cur && i != next) {
        const vec2 &vcur = vertices[cur];
        const vec2 &vnext = vertices[next];
        const vec2 &vi = vertices[i];

        const vec2 to_next = vnext - vcur;
        const vec2 to_i = vi - vcur;

        // Compare against the current candidate, using cross
        // product to find the "leftmost" one.
        double angle = yocto::cross(to_next, to_i);

        const bool is_strictly_left = angle < 0.0;

        bool take = is_strictly_left;

        if (VERBOSE) {
          printf("  Angle: %.17g. %s %s\n", angle,
                 is_strictly_left ? " left" : "",
                 take ? " take." : "");
        }

        if (take) {
          // However, if we get back to a point that's exactly equal
          // to start, we want to use the start index (for one thing,
          // so that this loop terminates).
          if (vi == vstart) {
            next = start;
          } else {
            next = i;
          }
        }
      }
    }

    cur = next;
  } while (cur != start);

  if (VERBOSE) {
    printf(ACYAN("Returned to start.") "\n");
  }
  return hull;
}

std::vector<int> GrahamScan(const std::vector<vec2> &vertices) {

  // Place the leftmost bottommost point first in the
  // working hull. This will be on the hull and it is the
  // reference point.
  const int ref = [&]() {
      int besti = 0;
      for (int i = 1; i < vertices.size(); i++) {
        if ((vertices[i].y < vertices[besti].y) ||
            (vertices[i].y == vertices[besti].y &&
             vertices[i].x < vertices[besti].x)) {
          besti = i;
        }
      }
      return besti;
    }();

  const vec2 &vref = vertices[ref];

  struct Point {
    int idx;
    double angle;
    double distsq;
  };
  // Get the remainder of the point indices.
  std::vector<Point> points;
  points.reserve(vertices.size() - 1);
  for (int i = 0; i < vertices.size(); i++) {
    if (i != ref) {
      const vec2 &v = vertices[i];
      const vec2 e = v - vref;
      points.emplace_back(Point{
          .idx = i,
          .angle = std::atan2(e.y, e.x),
          .distsq = length_squared(e),
        });
    }
  }

  // Sort the remaining points by their polar angle.
  std::sort(points.begin(),
            points.end(),
            [&](const Point &a, const Point &b) {
              if (a.angle == b.angle) {
                if (a.distsq == b.distsq) {
                  return a.idx < b.idx;
                } else {
                  return a.distsq < b.distsq;
                }
              } else {
                return a.angle < b.angle;
              }
            });

  // Remove collinear points now. We keep the farthest
  // one (and in the case of ties, break them using the
  // largest index).
  int new_size = 0;
  for (int i = 0; i < points.size(); i++) {
    // Skip a point if it has the same angle as the
    // next one; the last point in such a sequence is the
    // best one.
    if (i + 1 < points.size() &&
        points[i].angle == points[i + 1].angle) {
      continue;
    } else {
      points[new_size++] = points[i];
    }
  }
  points.resize(new_size);

  // Now we have the candidate set.
  // If it's trivial or degenerate, return those points.
  if (points.size() <= 2) {
    std::vector<int> hull = {ref};
    for (const Point &p : points) hull.push_back(p.idx);
    return hull;
  }

  // A proper set. Do the scan.
  std::vector<int> hull = {ref, points[0].idx, points[1].idx};

  // Taking vertex indices.
  auto CCW = [&vertices] (int a, int b, int c) {
      const vec2 &va = vertices[a];
      const vec2 &vb = vertices[b];
      const vec2 &vc = vertices[c];

      return cross(vb - va, vc - va) < 0.0;
    };

  for (int i = 2; i < points.size(); i++) {
    const Point &point = points[i];

    while (hull.size() >= 2 && CCW(hull[hull.size() - 2],
                                          hull[hull.size() - 1],
                                          point.idx)) {
      hull.pop_back();
    }

    hull.push_back(point.idx);
  }

  return hull;
}

// The QuickHull implementation below and its helper routines are based on code
// by Miguel Vieira (see LICENSES) although I have heavily modified it.
// Some changes:
//  - Uses yocto library for more stuff
//  - Uses vertex indices so it can be run directly on Polyhedron/Mesh2D.
//  - Fixes some bugs relating to exactly equal or colinear input points
//  - Some algorithmic improvements (e.g. left and right sets in the recursive
//    calls must be disjoint).

// Returns the index of the farthest point from segment (a, b).
// Requires that all points are to the left of the segment (a,b) (or colinear).
// (If you need it to handle both, you can just use fabs in the Dist function,
// but for quickhull our candidate set always lies on the left of the edge.)
static int GetFarthest(const vec2 &a, const vec2 &b,
                       const std::vector<vec2> &v,
                       const std::vector<int> &pts) {
  CHECK(!pts.empty());
  const double dx = b.x - a.x;
  const double dy = b.y - a.y;
  auto Dist = [&](const vec2 &p) -> double {
      return dx * (a.y - p.y) - dy * (a.x - p.x);
    };

  int best_idx = pts[0];
  double best_dist = Dist(v[best_idx]);

  for (int i = 1; i < pts.size(); i++) {
    int p = pts[i];
    double d = Dist(v[p]);
    if (d < best_dist) {
      best_idx = p;
      best_dist = d;
    }
  }

  return best_idx;
}

// The z-value of the cross product of segments
// (a, b) and (a, c). Positive means c is ccw (to the left)
// from (a, b), negative cw. Zero means it's colinear.
static inline double CounterClockwise(
    const vec2 &a, const vec2 &b, const vec2 &c) {
  return yocto::cross(b - a, c - a);
}


// Recursive call of the quickhull algorithm.
static void QuickHullRec(const std::vector<vec2> &vertices,
                         const std::vector<int> &pts,
                         int a, int b,
                         std::vector<int> *hull) {
  static constexpr bool SELF_CHECK = false;
  static constexpr int VERBOSE = 0;

  if (VERBOSE == 1) {
    printf("QuickHullRec(%d vs, %d pts, %d (%s), %d (%s))\n",
           (int)vertices.size(), (int)pts.size(),
           a, VecString(vertices[a]).c_str(),
           b, VecString(vertices[b]).c_str());
  } else if (VERBOSE > 0) {
    printf("QuickHullRec(%d vs, %d pts, %d (%s), %d (%s)):",
           (int)vertices.size(), (int)pts.size(),
           a, VecString(vertices[a]).c_str(),
           b, VecString(vertices[b]).c_str());
    for (int p : pts) {
      if (p == a || p == b) {
        printf(" " ANSI_BG(0, 0, 128) "%s" ANSI_RESET,
               VecString(vertices[p]).c_str());
      } else {
        printf(" %s", VecString(vertices[p]).c_str());
      }
    }
    printf("\n");
  }
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

  const vec2 &aa = vertices[a];
  const vec2 &bb = vertices[b];

  if (SELF_CHECK) {
    for (int x : pts) {
      const vec2 &xx = vertices[x];
      CHECK(aa == xx || bb == xx ||
            CounterClockwise(aa, bb, xx) > 0.0);
    }
  }

  int f = GetFarthest(aa, bb, vertices, pts);
  const vec2 &ff = vertices[f];
  if (VERBOSE) printf("Farthest is %d (%s)\n", f, VecString(ff).c_str());

  // CHECK(CounterClockwise(aa, ff, aa) <= 0.0);
  // CHECK(CounterClockwise(aa, ff, ff) <= 0.0);

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
    const vec2 &ii = vertices[i];
    // In the presence of exact duplicates for one of the endpoints,
    // we need to filter them out here or else we can end up in
    // infinite loops. Removing duplicates does not affect the hull.
    if (ii == aa || ii == bb || ii == ff) continue;

    if (CounterClockwise(aa, ff, ii) > 0.0) {
      left.push_back(i);
    } else if (CounterClockwise(ff, bb, ii) > 0.0) {
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
std::vector<int> QuickHull(const std::vector<vec2> &vertices) {
  std::vector<int> hull;
  if (vertices.empty()) return {};
  if (vertices.size() == 1) return {0};
  if (vertices.size() == 2) return {0, 1};

  // Returns true if a is lexicographically before b.
  auto LeftOf = [](const vec2 &a, const vec2 &b) -> bool {
      return (a.x < b.x || (a.x == b.x && a.y < b.y));
    };


  // Get the leftmost (a) and rightmost (b) points.
  int a = 0, b = 0;
  for (int i = 1; i < (int)vertices.size(); i++) {
    if (LeftOf(vertices[i], vertices[a])) a = i;
    if (LeftOf(vertices[b], vertices[i])) b = i;
  }

  if (a == b) [[unlikely]] {
    fprintf(stderr, "Quickhull failure on:\n");
    for (const vec2 &v : vertices) {
      printf("  {%.17g, %.17g},\n",
             v.x, v.y);
    }
    LOG(FATAL) << "Should not be possible!";
  }

  // Split the points on either side of segment (a, b).
  std::vector<int> left, right;
  for (int i = 0; i < (int)vertices.size(); i++) {
    if (i != a && i != b) {
      double side = CounterClockwise(vertices[a], vertices[b], vertices[i]);
      if (side > 0.0) left.push_back(i);
      else if (side < 0.0) right.push_back(i);
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

// This can be done faster with "rotating calipers" although
// it sounds pretty fiddly in 3D (especially since we will
// have many parallel faces for these regular shapes).
double Diameter(const Polyhedron &p) {
  double dist = 0.0;
  for (int i = 0; i < p.vertices.size(); i++) {
    for (int j = i + 1; j < p.vertices.size(); j++) {
      dist = std::max(dist, yocto::distance_squared(p.vertices[i],
                                                    p.vertices[j]));
    }
  }
  return std::sqrt(dist);
}

Polyhedron NormalizeRadius(const Polyhedron &p) {
  double max_dist = 0.0;
  for (const vec3 &v : p.vertices) {
    max_dist = std::max(max_dist, yocto::length(v));
  }
  Polyhedron copy = p;
  for (vec3 &v : copy.vertices) {
    v /= max_dist;
  }
  return copy;
}

Polyhedron Recenter(const Polyhedron &p) {
  vec3 avg(0.0, 0.0, 0.0);
  for (const vec3 &v : p.vertices) {
    avg += v;
  }

  avg /= p.vertices.size();
  Polyhedron copy = p;
  for (vec3 &v : copy.vertices) {
    v -= avg;
  }
  return copy;
}

double PlanarityError(const Polyhedron &p) {
  double error = 0.0;
  for (const std::vector<int> &face : p.faces->v) {
    // Only need to check for quads and larger.
    if (face.size() > 3) {
      // The first three vertices define a plane.
      const vec3 &v0 = p.vertices[face[0]];
      const vec3 &v1 = p.vertices[face[1]];
      const vec3 &v2 = p.vertices[face[2]];

      vec3 normal = yocto::normalize(yocto::cross(v1 - v0, v2 - v0));

      // Check error against this plane.
      for (int i = 3; i < face.size(); i++) {
        vec3 v = p.vertices[face[i]];
        double err = std::abs(yocto::dot(v - v0, normal));
        error += err;
      }
    }
  }
  return error;
}

bool IsConvex(const std::vector<vec2> &vertices,
              const std::vector<int> &polygon) {
  if (polygon.size() <= 3) return true;
  std::optional<int> s;
  for (int i = 0; i < polygon.size(); i++) {
    const vec2 &p0 = vertices[polygon[i]];
    const vec2 &p1 = vertices[polygon[(i + 1) % polygon.size()]];
    const vec2 &p2 = vertices[polygon[(i + 2) % polygon.size()]];

    vec2 e1 = p1 - p0;
    vec2 e2 = p2 - p1;

    double cx = cross(e1, e2);
    if (std::abs(cx) < 1e-10) continue;
    int sign = sgn(cx);
    if (s.has_value() && s.value() != sign)
      return false;
    s = {sign};
  }
  return true;
}

template<class GetPt>
inline static bool PointInPolygonT(const vec2 &point,
                                   int size,
                                   const GetPt &get_pt) {
  int winding_number = 0;
  for (int i = 0; i < size; i++) {
    const vec2 p0 = get_pt(i);
    const vec2 p1 = get_pt((i + 1) % size);

    // Check if the ray from the point to infinity intersects the edge
    if (point.y > std::min(p0.y, p1.y)) {
      if (point.y <= std::max(p0.y, p1.y)) {
        if (point.x <= std::max(p0.x, p1.x)) {
          if (p0.y != p1.y) {
            double vt = (point.y - p0.y) / (p1.y - p0.y);
            if (point.x < p0.x + vt * (p1.x - p0.x)) {
              winding_number++;
            }
          }
        }
      }
    }
  }

  // Point is inside if the winding number is odd
  return !!(winding_number & 1);
}

bool PointInPolygon(const vec2 &point,
                    const std::vector<vec2> &vertices,
                    const std::vector<int> &polygon) {
  return PointInPolygonT(point, polygon.size(),
                         [&](int idx) {
                           return vertices[polygon[idx]];
                         });
}

bool PointInPolygon(const vec2 &point,
                    const std::vector<vec2> &polygon) {
  return PointInPolygonT(point, polygon.size(),
                         [&](int idx) {
                           return polygon[idx];
                         });
}

double SignedAreaOfConvexPoly(const std::vector<vec2> &pts) {
  if (pts.size() < 3) return 0.0;
  double area = 0.0;
  // Iterate through the polygon vertices, using the shoelace formula.
  for (size_t i = 0; i < pts.size(); i++) {
    const vec2 &v0 = pts[i];
    const vec2 &v1 = pts[(i + 1) % pts.size()];
    area += v0.x * v1.y - v1.x * v0.y;
  }

  return area * 0.5;
}

// via https://en.wikipedia.org/wiki/Shoelace_formula
double SignedAreaOfHull(const Mesh2D &mesh, const std::vector<int> &hull) {
  if (hull.size() < 3) return 0.0;
  double area = 0.0;
  // Iterate through the polygon vertices, using the shoelace formula.
  for (size_t i = 0; i < hull.size(); i++) {
    const vec2 &v0 = mesh.vertices[hull[i]];
    const vec2 &v1 = mesh.vertices[hull[(i + 1) % hull.size()]];
    area += v0.x * v1.y - v1.x * v0.y;
  }

  return area * 0.5;
}

double AreaOfHull(const Mesh2D &mesh, const std::vector<int> &hull) {
  // Sign depends on the winding order, but we always want a positive
  // area.
  return std::abs(SignedAreaOfHull(mesh, hull));
}

bool FacesParallel(const Polyhedron &poly, int face1, int face2) {
  if (face1 == face2) return true;
  CHECK(poly.faces->v[face1].size() >= 3);
  CHECK(poly.faces->v[face2].size() >= 3);


  auto Normal = [&poly](int f) {
      const auto &face = poly.faces->v[f];
      const vec3 &v0 = poly.vertices[face[0]];
      const vec3 &v1 = poly.vertices[face[1]];
      const vec3 &v2 = poly.vertices[face[2]];

      return yocto::normalize(yocto::cross(v1 - v0, v2 - v0));
    };

  double dot = yocto::dot(Normal(face1), Normal(face2));
  double angle = std::acos(std::abs(dot));
  // Don't need to worry too much about "close to zero" since these
  // polyhedra don't have nearly-parallel faces.
  return angle < 1.0e-6;
}

std::pair<int, int> TwoNonParallelFaces(ArcFour *rc, const Polyhedron &poly) {
  const int num_faces = (int)poly.faces->v.size();
  for (;;) {
    int f1 = RandTo(rc, num_faces);
    int f2 = RandTo(rc, num_faces);
    if (!FacesParallel(poly, f1, f2)) {
      return std::make_pair(f1, f2);
    }
  }
}

// PERF: See polyehdra_benchmark for different approaches. This
// was the winner for the snub cube, but the tradeoffs are likely
// different for other shapes. (In particular, QuickHull may be
// a better choice for large numbers of vertices, and computing
// the hull/hullcircle is probably pointless for something like
// the tetrahedron).
double LossFunctionContainsOrigin(const Polyhedron &poly,
                                  const frame3 &outer_frame,
                                  const frame3 &inner_frame) {
  Mesh2D souter = Shadow(Rotate(poly, outer_frame));
  Mesh2D sinner = Shadow(Rotate(poly, inner_frame));

  // Although computing the convex hull is expensive, the tests
  // below are O(n*m), so it is helpful to significantly reduce
  // one of the factors.
  const std::vector<int> outer_hull = GrahamScan(souter.vertices);
  if (outer_hull.size() < 3) {
    // If the outer hull is degenerate, then the inner hull
    // cannot be strictly within it. We don't have a good
    // way to measure the gradient here, though.
    return 1'000'000.0;
  }

  HullInscribedCircle circle(souter.vertices, outer_hull);

  // Does every vertex in inner fall inside the outer shadow?
  double error = 0.0;
  int errors = 0;
  for (const vec2 &iv : sinner.vertices) {
    if (circle.DefinitelyInside(iv))
      continue;

    if (!InHull(souter, outer_hull, iv)) {
      // slow :(
      error += DistanceToHull(souter.vertices, outer_hull, iv);
      errors++;
    }
  }

  if (error == 0.0 && errors > 0) [[unlikely]] {
    // If they are not in the mesh, don't return an actual zero.
    return std::numeric_limits<double>::min() * errors;
  } else {
    return error;
  }
}

double LossFunction(const Polyhedron &poly,
                    const frame3 &outer_frame,
                    const frame3 &inner_frame) {
  Mesh2D souter = Shadow(Rotate(poly, outer_frame));
  Mesh2D sinner = Shadow(Rotate(poly, inner_frame));

  // Although computing the convex hull is expensive, the tests
  // below are O(n*m), so it is helpful to significantly reduce
  // one of the factors.
  const std::vector<int> outer_hull = GrahamScan(souter.vertices);
  if (outer_hull.size() < 3) {
    // If the outer hull is degenerate, then the inner hull
    // cannot be strictly within it. We don't have a good
    // way to measure the gradient here, though.
    return 1'000'000.0;
  }

  // Does every vertex in inner fall inside the outer shadow?
  double error = 0.0;
  int errors = 0;
  for (const vec2 &iv : sinner.vertices) {
    if (!InHull(souter, outer_hull, iv)) {
      // slow :(
      error += DistanceToHull(souter.vertices, outer_hull, iv);
      errors++;
    }
  }

  if (error == 0.0 && errors > 0) [[unlikely]] {
    // If they are not in the mesh, don't return an actual zero.
    return std::numeric_limits<double>::min() * errors;
  } else {
    return error;
  }
}

double FullLossContainsOrigin(
    const Polyhedron &poly,
    const frame3 &outer_frame, const frame3 &inner_frame) {

  Polyhedron outer = Rotate(poly, outer_frame);
  Polyhedron inner = Rotate(poly, inner_frame);
  Mesh2D souter = Shadow(outer);
  Mesh2D sinner = Shadow(inner);

  if (AllZero(souter.vertices) ||
      AllZero(sinner.vertices)) {

    return 1.0e6;
  }

  std::vector<int> outer_hull = QuickHull(souter.vertices);

  if (outer_hull.size() < 3) {
    return 1.0e6;
  }

  HullInscribedCircle circle(souter.vertices, outer_hull);

  // Does every vertex in inner fall inside the outer shadow?
  double error = 0.0;
  int errors = 0;
  for (const vec2 &iv : sinner.vertices) {
    if (circle.DefinitelyInside(iv))
      continue;

    if (!InHull(souter, outer_hull, iv)) {
      // slow :(
      error += DistanceToHull(souter.vertices, outer_hull, iv);
      errors++;
    }
  }

  if (errors > 0) {
    if (error == 0.0) {
      [[unlikely]]
      return std::numeric_limits<double>::min() * errors;
    }
    return error;
  } else {
    std::vector<int> inner_hull = QuickHull(sinner.vertices);
    double clearance = HullClearance(souter.vertices, outer_hull,
                                     sinner.vertices, inner_hull);
    return std::min(-clearance, 0.0);
  }
}

std::optional<double> GetRatio(const Polyhedron &poly,
                               const frame3 &outer_frame,
                               const frame3 &inner_frame) {
  // Compute new error ratio.
  Polyhedron outer = Rotate(poly, outer_frame);
  Polyhedron inner = Rotate(poly, inner_frame);
  Mesh2D souter = Shadow(outer);
  Mesh2D sinner = Shadow(inner);

  if (AllZero(souter.vertices) ||
      AllZero(sinner.vertices)) {
    /*
    fprintf(stderr, "Outer:\n%s\nInner:\n%s\n",
            FrameString(outer_frame).c_str(),
            FrameString(inner_frame).c_str());
    LOG(FATAL) << "???";
    */
    return std::nullopt;
  }

  std::vector<int> outer_hull = QuickHull(souter.vertices);
  std::vector<int> inner_hull = QuickHull(sinner.vertices);

  for (const vec2 &iv : sinner.vertices) {
    if (!InHull(souter, outer_hull, iv)) {
      return std::nullopt;
    }
  }

  double outer_area = AreaOfHull(souter, outer_hull);
  double inner_area = AreaOfHull(sinner, inner_hull);

  double ratio = inner_area / outer_area;
  return {ratio};
}

std::optional<double> GetClearance(const Polyhedron &poly,
                                   const frame3 &outer_frame,
                                   const frame3 &inner_frame) {
  Polyhedron outer = Rotate(poly, outer_frame);
  Polyhedron inner = Rotate(poly, inner_frame);
  Mesh2D souter = Shadow(outer);
  Mesh2D sinner = Shadow(inner);

  if (AllZero(souter.vertices) ||
      AllZero(sinner.vertices)) {
    /*
    fprintf(stderr, "Outer:\n%s\nInner:\n%s\n",
            FrameString(outer_frame).c_str(),
            FrameString(inner_frame).c_str());
    LOG(FATAL) << "???";
    */
    return std::nullopt;
  }

  std::vector<int> outer_hull = QuickHull(souter.vertices);
  std::vector<int> inner_hull = QuickHull(sinner.vertices);

  for (const vec2 &iv : sinner.vertices) {
    if (!InHull(souter, outer_hull, iv)) {
      return std::nullopt;
    }
  }

  return {HullClearance(souter.vertices, outer_hull,
                        sinner.vertices, inner_hull)};
}

static TriangularMesh3D ToTriangularMesh(const Polyhedron &poly) {
  return TriangularMesh3D{.vertices = poly.vertices,
    .triangles = poly.faces->triangulation};
}

void SaveAsSTL(const Polyhedron &poly, std::string_view filename) {
  TriangularMesh3D mesh = ToTriangularMesh(poly);
  OrientMesh(&mesh);
  return SaveAsSTL(mesh, filename, poly.name);
}

void DebugPointCloudAsSTL(const std::vector<vec3> &vertices,
                          std::string_view filename) {
  static constexpr double OBJECT_SCALE = 2.40;
  // For each vertex, generate a tiny tetrahedron.
  static constexpr double TETRAHEDRON_SCALE = 0.05;
  std::string contents = "solid debug\n";

  Polyhedron tet = Tetrahedron();

  for (const vec3 &c_orig : vertices) {
    const vec3 c = c_orig * OBJECT_SCALE;
    // Generate a tetrahedron at this point.
    for (const std::vector<int> &face : tet.faces->v) {
      CHECK(face.size() == 3);
      vec3 p0 = c + tet.vertices[face[0]] * TETRAHEDRON_SCALE;
      vec3 p1 = c + tet.vertices[face[1]] * TETRAHEDRON_SCALE;
      vec3 p2 = c + tet.vertices[face[2]] * TETRAHEDRON_SCALE;

      vec3 normal = yocto::normalize(yocto::cross(p1 - p0, p2 - p0));

      StringAppendF(&contents, "  facet normal %f %f %f\n", normal.x, normal.y,
                    normal.z);
      StringAppendF(&contents, "    outer loop\n");
      StringAppendF(&contents, "      vertex %f %f %f\n", p0.x, p0.y, p0.z);
      StringAppendF(&contents, "      vertex %f %f %f\n", p1.x, p1.y, p1.z);
      StringAppendF(&contents, "      vertex %f %f %f\n", p2.x, p2.y, p2.z);
      StringAppendF(&contents, "    endloop\n");
      StringAppendF(&contents, "  endfacet\n");
    }
  }

  StringAppendF(&contents, "endsolid debug\n");

  std::string f = (std::string)filename;
  Util::WriteFile(f, contents);
  printf("Wrote " AGREEN("%s") "\n", f.c_str());

  delete tet.faces;
}

void SaveAsJSON(const Polyhedron &poly, std::string_view filename) {
  std::string contents = "{\n";
  AppendFormat(&contents, " \"name\": \"{}\",\n", poly.name);
  StringAppendF(&contents, " \"verts\": [\n");
  const auto &verts = poly.vertices;
  for (int ii = 0; ii < verts.size(); ++ ii) {
    const auto &v = verts[ii];
    AppendFormat(&contents, "  [{},{},{}]", v[0], v[1], v[2]);
    if (ii + 1 < verts.size()) {
      AppendFormat(&contents, ",\n");
    }
  }
  AppendFormat(&contents, "],\n");
  AppendFormat(&contents, " \"faces\": [\n");
  for (int ii = 0; ii < poly.faces->v.size(); ++ ii) {
    AppendFormat(&contents, "  [");
    const auto &face = poly.faces->v[ii];
    int n = face.size();
    for (int jj = 0; jj < n; ++jj) {
      int vidx = face[jj];
      AppendFormat(&contents, "{}", vidx);
      if (jj + 1 < n) {
        AppendFormat(&contents, ",");
      }
    }
    AppendFormat(&contents, "]");
    if (ii + 1 < poly.faces->v.size()) {
      AppendFormat(&contents, ",\n");
    }
  }
  AppendFormat(&contents, "]");

  StringAppendF(&contents, "\n}\n");

  std::string f = (std::string)filename;
  Util::WriteFile(f, contents);
  printf("Wrote " AGREEN("%s") "\n", f.c_str());
}

void SaveAsJSON(const frame3 &outer_frame,
                const frame3 &inner_frame,
                std::string_view filename) {
  std::string contents = "{\n";
  AppendFormat(
      &contents,
      " \"outerframe\": "
      "[\n  {},{},{},\n  {},{},{},\n  {},{},{},\n  {},{},{}],",
      outer_frame.x.x, outer_frame.x.y, outer_frame.x.z,
      outer_frame.y.x, outer_frame.y.y, outer_frame.y.z,
      outer_frame.z.x, outer_frame.z.y, outer_frame.z.z,
      outer_frame.o.x, outer_frame.o.y, outer_frame.o.z);
  AppendFormat(
      &contents,
      " \"innerframe\": "
      "[\n  {},{},{},\n  {},{},{},\n  {},{},{},\n  {},{},{}]",
      inner_frame.x.x, inner_frame.x.y, inner_frame.x.z,
      inner_frame.y.x, inner_frame.y.y, inner_frame.y.z,
      inner_frame.z.x, inner_frame.z.y, inner_frame.z.z,
      inner_frame.o.x, inner_frame.o.y, inner_frame.o.z);
  StringAppendF(&contents, "\n}\n");

  std::string f = (std::string)filename;
  Util::WriteFile(f, contents);
  printf("Wrote " AGREEN("%s") "\n", f.c_str());
}


Polyhedron PolyhedronByName(std::string_view name) {
  if (name == "tetrahedron") return Tetrahedron();
  if (name == "cube") return Cube();
  if (name == "dodecahedron") return Dodecahedron();
  if (name == "icosahedron") return Icosahedron();
  if (name == "octahedron") return Octahedron();
  if (name == "truncatedtetrahedron") return TruncatedTetrahedron();
  if (name == "cuboctahedron") return Cuboctahedron();
  if (name == "truncatedcube") return TruncatedCube();
  if (name == "truncatedoctahedron") return TruncatedOctahedron();
  if (name == "rhombicuboctahedron") return Rhombicuboctahedron();
  if (name == "truncatedcuboctahedron") return TruncatedCuboctahedron();
  if (name == "snubcube") return SnubCube();
  if (name == "icosidodecahedron") return Icosidodecahedron();
  if (name == "truncateddodecahedron") return TruncatedDodecahedron();
  if (name == "truncatedicosahedron") return TruncatedIcosahedron();
  if (name == "rhombicosidodecahedron") return Rhombicosidodecahedron();
  if (name == "truncatedicosidodecahedron") return TruncatedIcosidodecahedron();
  if (name == "snubdodecahedron") return SnubDodecahedron();
  if (name == "triakistetrahedron") return TriakisTetrahedron();
  if (name == "rhombicdodecahedron") return RhombicDodecahedron();
  if (name == "triakisoctahedron") return TriakisOctahedron();
  if (name == "tetrakishexahedron") return TetrakisHexahedron();
  if (name == "deltoidalicositetrahedron") return DeltoidalIcositetrahedron();
  if (name == "disdyakisdodecahedron") return DisdyakisDodecahedron();
  if (name == "deltoidalhexecontahedron") return DeltoidalHexecontahedron();
  if (name == "pentagonalicositetrahedron") return PentagonalIcositetrahedron();
  if (name == "rhombictriacontahedron") return RhombicTriacontahedron();
  if (name == "triakisicosahedron") return TriakisIcosahedron();
  if (name == "pentakisdodecahedron") return PentakisDodecahedron();
  if (name == "disdyakistriacontahedron") return DisdyakisTriacontahedron();
  if (name == "pentagonalhexecontahedron") return PentagonalHexecontahedron();
  LOG(FATAL) << "Unknown polyhedron " << name;
}

namespace {
struct NameMap {
  NameMap() : names(
      std::vector<std::tuple<std::string, std::string, std::string>>{
        {"tetra", "tetrahedron", "tetrahedron"},
        {"cube", "cube", "cube"},
        {"dode", "dodecahedron", "dodecahedron"},
        {"icos", "icosahedron", "icosahedron"},
        {"octa", "octahedron", "octahedron"},
        {"ttetra", "truncatedtetrahedron", "truncated tetrahedron"},
        {"cocta", "cuboctahedron", "cuboctahedron"},
        {"tcube", "truncatedcube", "truncated cube"},
        {"tocta", "truncatedoctahedron", "truncated octahedron"},
        {"rcocta", "rhombicuboctahedron", "rhombicuboctahedron"},
        {"tcocta", "truncatedcuboctahedron", "truncated cuboctahedron"},
        {"scube", "snubcube", "snub cube"},
        {"idode", "icosidodecahedron", "icosidodecahedron"},
        {"tdode", "truncateddodecahedron", "truncated dodecahedron"},
        {"ticos", "truncatedicosahedron", "truncated icosahedron"},
        {"ridode", "rhombicosidodecahedron", "rhombicosidodecahedron"},
        {"tidode", "truncatedicosidodecahedron", "truncated icosidodecahedron"},
        {"sdode", "snubdodecahedron", "snub dodecahedron"},
        {"ktetra", "triakistetrahedron", "triakis tetrahedron"},
        {"rdode", "rhombicdodecahedron", "rhombic dodecahedron"},
        {"kocta", "triakisoctahedron", "triakis octahedron"},
        {"thexa", "tetrakishexahedron", "tetrakis hexahedron"},
        {"ditet", "deltoidalicositetrahedron", "deltoidal icositetrahedron"},
        {"ddode", "disdyakisdodecahedron", "disdyakis dodecahedron"},
        {"dhexe", "deltoidalhexecontahedron", "deltoidal hexecontahedron"},
        {"pitet", "pentagonalicositetrahedron", "pentagonal icositetrahedron"},
        {"rtriac", "rhombictriacontahedron", "rhombic triacontahedron"},
        {"kicos", "triakisicosahedron", "triakis icosahedron"},
        {"pdode", "pentakisdodecahedron", "pentakis dodecahedron"},
        {"dtriac", "disdyakistriacontahedron", "disdyakis triacontahedron"},
        {"phexe", "pentagonalhexecontahedron", "pentagonal hexecontahedron"}
      }) {}
  std::vector<std::tuple<std::string, std::string, std::string>> names;
};
}  // namespace

static const NameMap &GetNameMap() {
  static const NameMap *m = new NameMap;
  return *m;
}

std::string PolyhedronShortName(std::string_view name) {
  for (const auto &[a, b, c] : GetNameMap().names) {
    if (b == name) return a;
  }
  LOG(FATAL) << "Unknown polyhedron identifier: " << name;
}

std::string PolyhedronIdFromNickname(std::string_view name) {
  for (const auto &[a, b, c] : GetNameMap().names) {
    if (a == name) return b;
  }
  LOG(FATAL) << "Unknown polyhedron nickname: " << name;
}

std::string PolyhedronHumanName(std::string_view name) {
  for (const auto &[a, b, c] : GetNameMap().names) {
    if (b == name) return c;
  }
  LOG(FATAL) << "Unknown polyhedron identifier: " << name;
}

Polyhedron Dodecahedron() {
  constexpr bool VERBOSE = false;

  // double phi = (1.0 + sqrt(5.0)) / 2.0;
  constexpr double phi = std::numbers::phi;

  // The vertices have a nice combinatorial form.
  std::vector<vec3> vertices;

  // It's a beauty: The unit cube is in here.
  // The first eight vertices (0b000 - 0b111)
  // will be the corners of the cube, where a zero bit means
  // a coordinate of -1, and a one bit +1. The coordinates
  // are in xyz order.
  for (int i = 0b000; i < 0b1000; i++) {
    vertices.emplace_back(vec3{
        (i & 0b100) ? 1.0 : -1.0,
        (i & 0b010) ? 1.0 : -1.0,
        (i & 0b001) ? 1.0 : -1.0,
      });
  }

  for (bool j : {false, true}) {
    for (bool k : {false, true}) {
      double b = j ? phi : -phi;
      double c = k ? 1.0 / phi : -1.0 / phi;
      vertices.emplace_back(vec3{
          .x = 0.0, .y = b, .z = c});
      vertices.emplace_back(vec3{
          .x = c, .y = 0.0, .z = b});
      vertices.emplace_back(vec3{
          .x = b, .y = c, .z = 0.0});
    }
  }

  CHECK(vertices.size() == 20);

  // TODO: Can just use the convex hull here.

  // Rather than hard code faces, we find them from the
  // vertices. Every vertex has exactly three edges,
  // and they are to the three closest (other) vertices.

  std::vector<std::vector<int>> neighbors(20);
  for (int i = 0; i < vertices.size(); i++) {
    std::vector<std::pair<int, double>> others;
    others.reserve(19);
    for (int o = 0; o < vertices.size(); o++) {
      if (o != i) {
        others.emplace_back(o, distance(vertices[i], vertices[o]));
      }
    }
    std::sort(others.begin(), others.end(),
              [](const auto &a, const auto &b) {
                if (a.second == b.second) return a.first < b.first;
                return a.second < b.second;
              });
    others.resize(3);
    for (const auto &[idx, dist_] : others) {
      // printf("src %d, n %d, dist %.11g\n", i, idx, dist_);
      neighbors[i].push_back(idx);
    }
  }

  if (VERBOSE) {
    for (int i = 0; i < vertices.size(); i++) {
      const vec3 &v = vertices[i];
      printf("v " AWHITE("%d")
             ". (" ARED("%.3f") ", " AGREEN("%.3f") ", " ABLUE("%.3f")
             ") neighbors:", i, v.x, v.y, v.z);
      for (int n : neighbors[i]) {
        printf(" %d", n);
      }
      printf("\n");
    }
  }

  // Return the common neighbor. Aborts if there is no such
  // neighbor.
  auto CommonNeighbor = [&neighbors](int a, int b) {
      for (int aa : neighbors[a]) {
        for (int bb : neighbors[b]) {
          if (aa == bb) return aa;
        }
      }
      LOG(FATAL) << "Vertices " << a << " and " << b << " do not "
        "have a common neighbor.";
      return -1;
    };

  // Get the single neighbor of a that lies on the plane a,b,c (and
  // is not one of the arguments). Aborts if there is no such neighbor.
  auto CoplanarNeighbor = [&vertices, &neighbors](int a, int b, int c) {
      const vec3 &v0 = vertices[a];
      const vec3 &v1 = vertices[b];
      const vec3 &v3 = vertices[c];

      const vec3 normal = yocto::normalize(yocto::cross(v1 - v0, v3 - v0));

      for (int o : neighbors[a]) {
        if (o == b || o == c) continue;

        const vec3 &v = vertices[o];
        double err = std::abs(yocto::dot(v - v0, normal));
        // The other points won't even be close.
        if (err < 0.00001) {
          return o;
        }
      }

      LOG(FATAL) << "Vertices " << a << ", " << b << ", " << c <<
        " do not have a coplanar neighbor (for a).\n";
      return -1;
    };

  std::vector<std::vector<int>> fs;

  // Each face corresponds to an edge on the cube.
  for (int a = 0b000; a < 0b1000; a++) {
    for (int b = 0b000; b < 0b1000; b++) {
      // When the Hamming distance is exactly 1, this is an edge
      // of the cube. Consider only the case where a < b though.
      if (a < b && std::popcount<uint8_t>(a ^ b) == 1) {

        //       tip
        //       ,'.
        //    a,'   `.b
        //     \     /
        //      \___/
        //     o1   o2

        // These two points will share a neighbor, which is the
        // tip of the pentagon illustrated above.
        int tip = CommonNeighbor(a, b);

        int o1 = CoplanarNeighbor(a, b, tip);
        int o2 = CoplanarNeighbor(b, a, tip);

        fs.push_back(std::vector<int>{tip, b, o2, o1, a});
      }
    }
  }

  Faces *faces = new Faces(vertices.size(), std::move(fs));
  return Polyhedron{
    .vertices = std::move(vertices),
    .faces = faces,
    .name = "dodecahedron",
  };
}

quat4 RotationFromAToB(const vec3 &a, const vec3 &b) {
  vec3 norma = normalize(a);
  vec3 normb = normalize(b);
  double d = dot(norma, normb);
  vec3 axis = cross(norma, normb);
  if (length_squared(axis) < 1e-10) {
    if (d > 0) {
      return quat4{0, 0, 0, 1};
    } else {
      // Rotate around any perpendicular axis.
      vec3 perp_axis = orthogonal(norma);
      return QuatFromVec(yocto::rotation_quat(perp_axis, std::numbers::pi));
    }
  }

  double angle = std::acos(std::clamp(d, -1.0, 1.0));
  return QuatFromVec(yocto::rotation_quat(axis, angle));

  // TODO: We should be able to do this without the special cases?
#if 0
  double d = dot(a, b);
  vec3 axis = cross(a, b);

  double s = sqrt((1.0 + d) * 2.0);
  double inv_s = 1.0 / s;
  return normalize(quat4(axis.x * inv_s, axis.y * inv_s, axis.z * inv_s,
                         s * 0.5));
#endif
}

std::pair<quat4, vec3> UnpackFrame(const frame3 &f) {
  using mat3 = yocto::mat<double, 3>;

  const mat3 m = rotation(f);

  double w = sqrt(std::max(0.0, 1.0 + m[0][0] + m[1][1] + m[2][2])) * 0.5;
  double x = sqrt(std::max(0.0, 1.0 + m[0][0] - m[1][1] - m[2][2])) * 0.5;
  double y = sqrt(std::max(0.0, 1.0 - m[0][0] + m[1][1] - m[2][2])) * 0.5;
  double z = sqrt(std::max(0.0, 1.0 - m[0][0] - m[1][1] + m[2][2])) * 0.5;

  if (m[1][2] - m[2][1] < 0.0) x = -x;
  if (m[2][0] - m[0][2] < 0.0) y = -y;
  if (m[0][1] - m[1][0] < 0.0) z = -z;

  return std::make_pair(normalize(quat4(x, y, z, w)),
                        yocto::translation(f));
}


TriangularMesh3D ApproximateSphere(int depth) {

  #if 0
  // Start with tetrahedron.
  // You get a cool looking shape, but it's actually pretty
  // irregular.
  TriangularMesh3D mesh;
  mesh.vertices = {
    normalize(vec3{1.0,   1.0,  1.0}),
    normalize(vec3{1.0,  -1.0, -1.0}),
    normalize(vec3{-1.0,  1.0, -1.0}),
    normalize(vec3{-1.0, -1.0,  1.0}),
  };

  for (int i = 0; i < 4; i++) {
    for (int j = i + 1; j < 4; j++) {
      for (int k = j + 1; k < 4; k++) {
        mesh.triangles.emplace_back(i, j, k);
      }
    }
  }
  #endif

  // Icosahedron is way better!
  TriangularMesh3D mesh = []() {
      Polyhedron icos = Icosahedron();
      TriangularMesh3D mesh{
        .vertices = icos.vertices,
        .triangles = icos.faces->triangulation,
      };
      delete icos.faces;

      for (vec3 &v : mesh.vertices) {
        v = normalize(v);
      }

      OrientMesh(&mesh);

      return mesh;
    }();

  // Triforce Subdivision.
  while (depth--) {
    std::unordered_map<std::pair<int, int>, int,
                       Hashing<std::pair<int, int>>> midpoints;
    TriangularMesh3D submesh;
    submesh.vertices = mesh.vertices;

    auto MidPoint = [&](int a, int b) {
        if (a > b) std::swap(a, b);
        auto it = midpoints.find(std::make_pair(a, b));
        if (it == midpoints.end()) {
          CHECK(b < mesh.vertices.size());
          // We want the average, but since we are normalizing anyway,
          // we can skip the scale.
          vec3 m = normalize(mesh.vertices[a] + mesh.vertices[b]);
          int id = submesh.vertices.size();
          midpoints[std::make_pair(a, b)] = id;
          submesh.vertices.push_back(m);
          return id;
        }
        else return it->second;
      };

    for (const auto &[a, b, c] : mesh.triangles) {
      //
      //    a---d---b
      //     \ / \ /
      //      e---f
      //       \ /
      //        c
      //
      int d = MidPoint(a, b);
      int e = MidPoint(a, c);
      int f = MidPoint(b, c);

      // Preserve clockwise winding.
      submesh.triangles.emplace_back(a, d, e);
      submesh.triangles.emplace_back(d, b, f);
      submesh.triangles.emplace_back(d, f, e);
      submesh.triangles.emplace_back(e, f, c);
    }
    mesh = std::move(submesh);
  }

  return mesh;
}

// Take all planes where all of the other vertices
// are on one side. (Basically, the 3D convex hull.)
// This is not fast but it should work for any convex polyhedron,
// so it's a clean way to generate a wide variety from just the
// vertices.
//
// If FRAGILE is true, then it aborts if anything is wrong.
// Otherwise, returns false if anything is wrong.
template<bool FRAGILE>
static bool InitPolyhedronInternal(
    std::vector<vec3> vertices,
    std::string_view name,
    Polyhedron *out) {
  static constexpr int VERBOSE = 0;

  // All faces (as a set of vertices) we've already found. The
  // vertices in the face have not yet been ordered; they appear in
  // ascending sorted order.
  std::unordered_set<std::vector<int>, Hashing<std::vector<int>>>
    all_faces;

  bool degenerate = false;
  // Given a plane defined by distinct points (v0, v1, v2), classify
  // all of the other points as either above, below, or on the plane.
  // If there are nonzero points both above and below, then this is
  // not a face; return nullopt. Otherwise, return the indices of the
  // vertices on the face, which will include at least i, j, and k.
  // The vertices are returned in sorted order (by index), which may
  // not be a proper winding for the polygon.
  auto GetFace = [&vertices, &degenerate](int i, int j, int k) ->
    std::optional<std::vector<int>> {
    (void)degenerate;
    // Three vertices define a plane.
    const vec3 &v0 = vertices[i];
    const vec3 &v1 = vertices[j];
    const vec3 &v2 = vertices[k];

    // Classify every point depending on what side it's on (or
    // whether it's on the plane). We don't need to worry about
    // ambiguity from numerical error here, as for the polyhedra
    // we consider, the points are either exactly on the plane or
    // comfortably far from it.

    const vec3 normal =
      yocto::normalize(yocto::cross(v1 - v0, v2 - v0));

    if (VERBOSE > 1) {
      printf("Try %s;%s;%s\n   Normal: %s\n",
             VecString(v0).c_str(),
             VecString(v1).c_str(),
             VecString(v2).c_str(),
             VecString(normal).c_str());
    }

    std::vector<int> coplanar;

    bool above = false, below = false;

    // Now for every other vertex...
    for (int o = 0; o < vertices.size(); o++) {
      if (o != i && o != j && o != k) {
        const vec3 &v = vertices[o];
        double dot = yocto::dot(v - v0, normal);
        if (dot < -0.00001) {
          if (above) return std::nullopt;
          below = true;
        } else if (dot > 0.00001) {
          if (below) return std::nullopt;
          above = true;
        } else {
          // On plane.
          coplanar.push_back(o);
        }
      }
    }

    if constexpr (FRAGILE) {
      CHECK(!(below && above));
      CHECK(below || above) << "This would only happen if we had "
        "a degenerate, volumeless polyhedron, which we do not.";
    } else {
      if (below == above) {
        degenerate = true;
        return std::nullopt;
      }
    }

    coplanar.push_back(i);
    coplanar.push_back(j);
    coplanar.push_back(k);
    std::sort(coplanar.begin(), coplanar.end());
    return coplanar;
  };

  if (VERBOSE > 0) {
    printf("%s: There are %d vertices.\n",
           std::string(name).c_str(), (int)vertices.size());
  }

  // wlog i > j > k.
  for (int i = 0; i < vertices.size(); i++) {
    for (int j = 0; j < i; j++) {
      for (int k = 0; k < j; k++) {
        if (std::optional<std::vector<int>> fo =
            GetFace(i, j, k)) {
          all_faces.insert(std::move(fo.value()));
        }
      }
    }
  }

  if (degenerate) return false;

  if (VERBOSE > 0) {
    printf("%s: There are %d distinct faces.\n",
           std::string(name).c_str(), (int)all_faces.size());
  }

  // Make it deterministic.
  std::vector<std::vector<int>> sfaces = SetToSortedVec(all_faces);

  std::vector<std::vector<int>> fs;
  for (const std::vector<int> &vec : sfaces) {
    CHECK(vec.size() >= 3);
    const vec3 &v0 = vertices[vec[0]];
    const vec3 &v1 = vertices[vec[1]];
    const vec3 &v2 = vertices[vec[2]];

    // But we'll need to express the vertices on the face in terms of
    // an orthonormal basis derived from the face's plane. This means
    // computing two perpendicular vectors on the face. We'll use one
    // of the edges (normalized) and then compute the other to lie
    // in the same plane.

    vec3 a = yocto::normalize(v1 - v0);
    vec3 b = yocto::normalize(v2 - v0);
    const vec3 u = yocto::orthonormalize(a, b);
    const vec3 v = b;

    // We'll compute the angle around the centroid.
    vec3 centroid{0.0, 0.0, 0.0};
    for (int index : vec) {
      centroid += vertices[index];
    }
    centroid /= vec.size();

    std::vector<std::pair<int, double>> iangle;
    for (int i : vec) {
      const vec3 &dir = vertices[i] - centroid;
      double angle = std::atan2(yocto::dot(dir, v), yocto::dot(dir, u));
      iangle.emplace_back(i, angle);
    }

    // Now order them according to the angle.
    std::sort(iangle.begin(), iangle.end(),
              [](const auto &a, const auto &b) {
                return a.second < b.second;
              });

    // And output to the face.
    std::vector<int> face;
    face.reserve(iangle.size());
    for (const auto &[i, angle_] : iangle) {
      face.push_back(i);
    }

    fs.push_back(std::move(face));
  }

  out->faces = Faces::Create(vertices.size(), std::move(fs));
  if (out->faces == nullptr)
    return false;

  out->vertices = std::move(vertices);
  out->name = std::string(name);
  return true;
}

static Polyhedron MakeConvexOrDie(
    std::vector<vec3> vertices, const char *name,
    SymmetryGroup symmetry) {
  Polyhedron poly;
  (void)InitPolyhedronInternal<true>(vertices, name, &poly);
  poly.symmetry = SYM_UNKNOWN;
  return poly;
}

std::optional<Polyhedron> PolyhedronFromConvexVertices(
    std::vector<vec3> vertices, std::string_view name) {
  Polyhedron poly;
  if (!InitPolyhedronInternal<false>(vertices, name, &poly)) {
    return std::nullopt;
  }
  return std::make_optional<Polyhedron>(std::move(poly));
}

// PERF: We can do this much more efficiently by using the faces
// that the hull code generates. We can fuse coplanar faces, or
// in the internals I think it has an even better representation.
std::optional<Polyhedron> PolyhedronFromVertices(
    std::vector<vec3> vertices, std::string_view name) {
  if (vertices.size() < 4) return {};
  std::vector<int> hull = Hull3D::HullPoints(vertices);
  std::vector<vec3> hull_vertices;
  hull_vertices.reserve(hull.size());
  for (int i : hull) {
    hull_vertices.push_back(vertices[i]);
  }
  return PolyhedronFromConvexVertices(std::move(hull_vertices), name);
}

Polyhedron SnubCube() {
  const double tribonacci =
    (1.0 + std::cbrt(19.0 + 3.0 * std::sqrt(33.0)) +
     std::cbrt(19.0 - 3.0 * std::sqrt(33.0))) / 3.0;

  const double a = 1.0;
  const double b = 1.0 / tribonacci;
  const double c = tribonacci;

  std::vector<vec3> vertices;

  // All even permutations with an even number of plus signs.
  //    (odd number of negative signs)
  // (a, b, c) - even
  // (b, c, a) - even
  // (c, a, b) - even

  // 1 = negative, 0 = positive
  for (const uint8_t s : {0b100, 0b010, 0b001, 0b111}) {
    vec3 signs{
      .x = (s & 0b100) ? -1.0 : 1.0,
      .y = (s & 0b010) ? -1.0 : 1.0,
      .z = (s & 0b001) ? -1.0 : 1.0,
    };

    vertices.emplace_back(vec3(a, b, c) * signs);
    vertices.emplace_back(vec3(b, c, a) * signs);
    vertices.emplace_back(vec3(c, a, b) * signs);
  }

  // And all odd permutations with an odd number of plus signs.
  //    (even number of negative signs).

  // (a, c, b) - odd
  // (b, a, c) - odd
  // (c, b, a) - odd
  // 1 = negative, 0 = positive
  for (const uint8_t s : {0b011, 0b110, 0b101, 0b000}) {
    vec3 signs{
      .x = (s & 0b100) ? -1.0 : 1.0,
      .y = (s & 0b010) ? -1.0 : 1.0,
      .z = (s & 0b001) ? -1.0 : 1.0,
    };

    vertices.emplace_back(vec3(a, c, b) * signs);
    vertices.emplace_back(vec3(b, a, c) * signs);
    vertices.emplace_back(vec3(c, b, a) * signs);
  }

  return MakeConvexOrDie(std::move(vertices), "snubcube",
                         SYM_OCTAHEDRAL);
}

Polyhedron Tetrahedron() {
  std::vector<vec3> vertices{
    vec3{1.0,   1.0,  1.0},
    vec3{1.0,  -1.0, -1.0},
    vec3{-1.0,  1.0, -1.0},
    vec3{-1.0, -1.0,  1.0},
  };

  return MakeConvexOrDie(std::move(vertices), "tetrahedron",
                         SYM_TETRAHEDRAL);
}

Polyhedron Cube() {
  //                  +y
  //      a------b     | +z
  //     /|     /|     |/
  //    / |    / |     0--- +x
  //   d------c  |
  //   |  |   |  |
  //   |  e---|--f
  //   | /    | /
  //   |/     |/
  //   h------g

  std::vector<vec3> vertices;
  auto AddVertex = [&vertices](double x, double y, double z) {
      int idx = (int)vertices.size();
      vertices.emplace_back(vec3{.x = x, .y = y, .z = z});
      return idx;
    };
  int a = AddVertex(-1, +1, +1);
  int b = AddVertex(+1, +1, +1);
  int c = AddVertex(+1, +1, -1);
  int d = AddVertex(-1, +1, -1);

  int e = AddVertex(-1, -1, +1);
  int f = AddVertex(+1, -1, +1);
  int g = AddVertex(+1, -1, -1);
  int h = AddVertex(-1, -1, -1);

  std::vector<std::vector<int>> fs;
  fs.reserve(6);

  // top
  fs.push_back({a, b, c, d});
  // bottom
  fs.push_back({e, f, g, h});
  // left
  fs.push_back({a, e, h, d});
  // right
  fs.push_back({b, f, g, c});
  // front
  fs.push_back({d, c, g, h});
  // back
  fs.push_back({a, b, f, e});

  Faces *faces = new Faces(8, std::move(fs));
  return Polyhedron{
    .vertices = std::move(vertices),
    .faces = faces,
    .name = "cube",
  };
}

Polyhedron Octahedron() {
  std::vector<vec3> vertices;
  for (double s : {-1.0, 1.0}) {
    vertices.emplace_back(s, 0.0, 0.0);
    vertices.emplace_back(0.0, s, 0.0);
    vertices.emplace_back(0.0, 0.0, s);
  }

  CHECK(vertices.size() == 6);
  return MakeConvexOrDie(std::move(vertices), "octahedron",
                         SYM_OCTAHEDRAL);
}

Polyhedron Icosahedron() {
  constexpr double phi = std::numbers::phi;
  // (±1, ±φ, 0)
  // (±φ, 0, ±1)
  // (0, ±1, ±φ)
  std::vector<vec3> vertices;
  for (int b = 0b00; b < 0b100; b++) {
    double s1 = (b & 0b10) ? -1 : +1;
    double s2 = (b & 0b01) ? -1 : +1;
    vertices.push_back(vec3{.x = s1, .y = s2 * phi, .z = 0.0});
    vertices.push_back(vec3{.x = s1 * phi, .y = 0.0, .z = s2});
    vertices.push_back(vec3{.x = 0.0, .y = s1, .z = s2 * phi});
  }

  return MakeConvexOrDie(std::move(vertices), "icosahedron",
                         SYM_ICOSAHEDRAL);
}

Polyhedron Cuboctahedron() {
  std::vector<vec3> vertices;
  vertices.reserve(24);
  for (int b = 0b00; b < 0b100; b++) {
    double s1 = (b & 0b10) ? -1 : +1;
    double s2 = (b & 0b01) ? -1 : +1;
    vertices.emplace_back(s1, s2, 0.0);
    vertices.emplace_back(s1, 0.0, s2);
    vertices.emplace_back(0.0, s1, s2);
  }
  return MakeConvexOrDie(
      std::move(vertices), "cuboctahedron",
      SYM_TETRAHEDRAL | SYM_OCTAHEDRAL);
}

Polyhedron Rhombicuboctahedron() {
  constexpr double u = 1.0 + std::numbers::sqrt2;
  std::vector<vec3> vertices;
  vertices.reserve(24);
  for (int b = 0b000; b < 0b1000; b++) {
    double s1 = (b & 0b100) ? -1 : +1;
    double s2 = (b & 0b010) ? -1 : +1;
    double s3 = (b & 0b001) ? -1 : +1;

    vertices.emplace_back(s1 * u, s2, s3);
    vertices.emplace_back(s1, s2 * u, s3);
    vertices.emplace_back(s1, s2, s3 * u);
  }

  // printf("Get faces..\n");
  return MakeConvexOrDie(
      std::move(vertices), "rhombicuboctahedron",
      SYM_OCTAHEDRAL
      // I think also "SYM_TETRAHEDRAL" may be correct here.
      // It is formally "pyritohedral"
                         );
}

Polyhedron TruncatedCuboctahedron() {
  std::vector<vec3> vertices;
  constexpr double a = 1.0;
  constexpr double b = 1.0 + std::numbers::sqrt2;
  constexpr double c = 1.0 + 2.0 * std::numbers::sqrt2;

  for (int bits = 0b000; bits < 0b1000; bits++) {
    const double s1 = (bits & 0b100) ? -1 : +1;
    const double s2 = (bits & 0b010) ? -1 : +1;
    const double s3 = (bits & 0b001) ? -1 : +1;

    const double aa = s1 * a;
    const double bb = s2 * b;
    const double cc = s3 * c;

    vertices.emplace_back(aa, bb, cc);
    vertices.emplace_back(aa, cc, bb);
    vertices.emplace_back(bb, aa, cc);
    vertices.emplace_back(bb, cc, aa);
    vertices.emplace_back(cc, aa, bb);
    vertices.emplace_back(cc, bb, aa);
  }

  CHECK(vertices.size() == 48);
  return MakeConvexOrDie(
      std::move(vertices), "truncatedcuboctahedron",
      SYM_OCTAHEDRAL);
}

static void AddEvenPermutations(double a, double b, double c,
                                std::vector<vec3> *vertices) {
  // (a, b, c) - even
  // (b, c, a) - even
  // (c, a, b) - even

  vertices->emplace_back(a, b, c);

  if (a == b && b == c) return;

  vertices->emplace_back(b, c, a);
  vertices->emplace_back(c, a, b);
}

static void AddOddPermutations(double a, double b, double c,
                               std::vector<vec3> *vertices) {
  // (a, c, b) - odd
  // (b, a, c) - odd
  // (c, b, a) - odd
  vertices->emplace_back(a, c, b);

  if (a == b && b == c) return;

  vertices->emplace_back(b, a, c);
  vertices->emplace_back(c, b, a);
}


Polyhedron Icosidodecahedron() {
  constexpr double phi = std::numbers::phi;
  constexpr double phi_squared = phi * phi;

  std::vector<vec3> vertices;

  for (int b = 0; b < 2; b++) {
    double c = b ? -phi : phi;
    vertices.emplace_back(0.0, 0.0, c);
    vertices.emplace_back(0.0, c, 0.0);
    vertices.emplace_back(c, 0.0, 0.0);
  }

  for (int b = 0b000; b < 0b1000; b++) {
    const double s1 = (b & 0b100) ? -1 : +1;
    const double s2 = (b & 0b010) ? -1 : +1;
    const double s3 = (b & 0b001) ? -1 : +1;

    AddEvenPermutations(s1 * 0.5,
                        s2 * phi * 0.5,
                        s3 * phi_squared * 0.5,
                        &vertices);
  }

  CHECK(vertices.size() == 30);
  return MakeConvexOrDie(
      std::move(vertices), "icosidodecahedron",
      SYM_ICOSAHEDRAL);
}

Polyhedron TruncatedDodecahedron() {
  constexpr double phi = std::numbers::phi;
  constexpr double inv_phi = 1.0 / phi;
  std::vector<vec3> vertices;

  for (int b = 0b00; b < 0b100; b++) {
    const double s1 = (b & 0b10) ? -1 : +1;
    const double s2 = (b & 0b01) ? -1 : +1;

    AddEvenPermutations(0.0, s1 * inv_phi, s2 * (2.0 + phi),
                        &vertices);
  }

  for (int b = 0b000; b < 0b1000; b++) {
    const double s1 = (b & 0b100) ? -1 : +1;
    const double s2 = (b & 0b010) ? -1 : +1;
    const double s3 = (b & 0b001) ? -1 : +1;

    AddEvenPermutations(s1 * inv_phi, s2 * phi, s3 * 2.0 * phi,
                        &vertices);
    AddEvenPermutations(s1 * phi, s2 * 2.0, s3 * (phi + 1.0),
                        &vertices);
  }

  CHECK(vertices.size() == 60);
  return MakeConvexOrDie(
      std::move(vertices), "truncateddodecahedron",
      SYM_ICOSAHEDRAL);
}

Polyhedron TruncatedIcosahedron() {
  std::vector<vec3> vertices;

  // Derive from the icosahedron.
  Polyhedron ico = Icosahedron();
  CHECK(ico.vertices.size() == ico.faces->neighbors.size()) <<
    ico.vertices.size() << " vs " << ico.faces->neighbors.size();
  for (int i = 0; i < ico.faces->neighbors.size(); i++) {
    for (int j : ico.faces->neighbors[i]) {
      // Consider every edge, but only once.
      if (i < j) {
        const vec3 &v0 = ico.vertices[i];
        const vec3 &v1 = ico.vertices[j];
        const vec3 v = v1 - v0;
        // Shrink the edge to its middle third.
        vertices.emplace_back(v0 + v / 3.0);
        vertices.emplace_back(v0 + (2.0 * v) / 3.0);
      }
    }
  }
  delete ico.faces;
  ico.faces = nullptr;

  CHECK(vertices.size() == 60);
  return MakeConvexOrDie(
      std::move(vertices), "truncatedicosahedron",
      SYM_ICOSAHEDRAL);
}

Polyhedron TruncatedIcosidodecahedron() {
  constexpr double phi = std::numbers::phi;
  constexpr double inv_phi = 1.0 / phi;

  std::vector<vec3> vertices;
  for (int b = 0b000; b < 0b1000; b++) {
    double s1 = (b & 0b100) ? -1 : +1;
    double s2 = (b & 0b010) ? -1 : +1;
    double s3 = (b & 0b001) ? -1 : +1;

    AddEvenPermutations(s1 * inv_phi,
                        s2 * inv_phi,
                        s3 * (3.0 + phi),
                        &vertices);
    AddEvenPermutations(s1 * 2.0 * inv_phi,
                        s2 * phi,
                        s3 * (1.0 + 2.0 * phi),
                        &vertices);
    AddEvenPermutations(s1 * inv_phi,
                        s2 * phi * phi,
                        s3 * (3.0 * phi - 1.0),
                        &vertices);
    AddEvenPermutations(s1 * (2.0 * phi - 1.0),
                        s2 * 2.0,
                        s3 * (2.0 + phi),
                        &vertices);
    AddEvenPermutations(s1 * phi, s2 * 3.0, s3 * 2.0 * phi,
                        &vertices);
  }

  CHECK(vertices.size() == 120);
  return MakeConvexOrDie(
      std::move(vertices), "truncatedicosidodecahedron",
      SYM_ICOSAHEDRAL);
}



Polyhedron Rhombicosidodecahedron() {
  constexpr double phi = std::numbers::phi;
  constexpr double phi_squared = phi * phi;
  constexpr double phi_cubed = phi_squared * phi;

  std::vector<vec3> vertices;
  for (int b = 0b000; b < 0b1000; b++) {
    double s1 = (b & 0b100) ? -1 : +1;
    double s2 = (b & 0b010) ? -1 : +1;
    double s3 = (b & 0b001) ? -1 : +1;

    // (±1, ±1, ±φ^3),
    // (±φ^2, ±φ, ±2φ),
    AddEvenPermutations(s1, s2, s3 * phi_cubed, &vertices);
    AddEvenPermutations(s1 * phi_squared, s2 * phi, s3 * 2.0 * phi, &vertices);
  }

  for (int b = 0b00; b < 0b100; b++) {
    double s1 = (b & 0b10) ? -1 : +1;
    double s2 = (b & 0b01) ? -1 : +1;
    // (±(2+φ), 0, ±φ^2),
    AddEvenPermutations(s1 * (2.0 + phi), 0.0, s2 * phi_squared, &vertices);
  }

  CHECK(vertices.size() == 60) << vertices.size();
  return MakeConvexOrDie(
      std::move(vertices), "rhombicosidodecahedron",
      SYM_ICOSAHEDRAL);
}

Polyhedron TruncatedTetrahedron() {
  constexpr double sqrt2 = std::numbers::sqrt2;

  std::vector<vec3> vertices;
  constexpr double a = 3.0 * sqrt2 / 4.0;
  constexpr double b = sqrt2 / 4.0;

  for (uint8_t bits = 0b000; bits < 0b1000; bits++) {
    double s1 = (bits & 0b100) ? -1 : +1;
    double s2 = (bits & 0b010) ? -1 : +1;
    double s3 = (bits & 0b001) ? -1 : +1;

    // All permutations of (a, b, b), with even number of minus
    // signs.
    if ((std::popcount<uint8_t>(bits) & 1) == 0) {
      vertices.emplace_back(s1 * a, s2 * b, s3 * b);
      vertices.emplace_back(s3 * b, s1 * a, s2 * b);
      vertices.emplace_back(s2 * b, s3 * b, s1 * a);
    }
  }

  CHECK(vertices.size() == 12);
  return MakeConvexOrDie(std::move(vertices), "truncatedtetrahedron",
                         SYM_TETRAHEDRAL);
}

Polyhedron TruncatedCube() {
  constexpr double inv_silver = 1.0 / (std::numbers::sqrt2 + 1.0);
  std::vector<vec3> vertices;

  for (uint8_t bits = 0b000; bits < 0b1000; bits++) {
    double s1 = (bits & 0b100) ? -1 : +1;
    double s2 = (bits & 0b010) ? -1 : +1;
    double s3 = (bits & 0b001) ? -1 : +1;

    vertices.emplace_back(s1 * inv_silver, s2, s3);
    vertices.emplace_back(s1, s2 * inv_silver, s3);
    vertices.emplace_back(s1, s2, s3 * inv_silver);
  }

  CHECK(vertices.size() == 24);
  return MakeConvexOrDie(std::move(vertices), "truncatedcube",
                         SYM_OCTAHEDRAL);
}

Polyhedron TruncatedOctahedron() {
  std::vector<vec3> vertices;

  constexpr double a = std::numbers::sqrt2;
  constexpr double b = std::numbers::sqrt2 * 0.5;

  for (uint8_t bits = 0b00; bits < 0b100; bits++) {
    double s1 = (bits & 0b10) ? -1 : +1;
    double s2 = (bits & 0b01) ? -1 : +1;

    vertices.emplace_back(s1 * a, s2 * b, 0.0);
    vertices.emplace_back(s1 * a, 0.0, s2 * b);
    vertices.emplace_back(0.0, s1 * a, s2 * b);

    vertices.emplace_back(s2 * b, s1 * a, 0.0);
    vertices.emplace_back(s2 * b, 0.0, s1 * a);
    vertices.emplace_back(0.0, s2 * b, s1 * a);
  }

  CHECK(vertices.size() == 24);
  return MakeConvexOrDie(std::move(vertices),
                         "truncatedoctahedron",
                         SYM_OCTAHEDRAL);
}

// The snub dodecahedron code is shared by the SnubDodecahedron
// (surprise!) and the PentagonalHexecontahedron, which I guess does
// not have a well-known simpler description without using duals.
//
// This represents the shape with edge length 2. It follows
// https://polytope.miraheze.org/wiki/Snub_dodecahedron
// But with all of the vertex coordinates multiplied by 2
// to simplify the expressions.
//
// Maybe a unit edge length would have been cleaner here after all,
// but I don't want to scale these shapes after solutions exist.
static std::vector<vec3> TwoEdgeLengthSnubDodecahedron() {
  constexpr double phi = std::numbers::phi;
  constexpr double phi_squared = phi * phi;

  constexpr double term = Sqrt(phi - 5.0 / 27.0);
  constexpr double xi =
    Cbrt(0.5 * (phi + term)) +
    Cbrt(0.5 * (phi - term));

  constexpr double xi_squared = xi * xi;
  constexpr double inv_xi = 1.0 / xi;

  std::vector<vec3> vertices;
  for (uint8_t bits = 0b000; bits < 0b1000; bits++) {
    double s1 = (bits & 0b100) ? -1 : +1;
    double s2 = (bits & 0b010) ? -1 : +1;
    double s3 = (bits & 0b001) ? -1 : +1;

    if ((std::popcount<uint8_t>(bits) & 1) == 1) {
      // Odd number of negative signs.

      AddEvenPermutations(
          s1 * phi * std::sqrt(phi * (xi - 1.0 - inv_xi)),
          s2 * xi * phi * std::sqrt(3.0 - xi_squared),
          s3 * phi * std::sqrt(xi * (xi + phi) + 1.0),
          &vertices);
      AddEvenPermutations(
          s1 * phi * std::sqrt(3.0 - xi_squared),
          s2 * xi * phi * std::sqrt(1.0 - xi + (1.0 + phi) / xi),
          s3 * phi * std::sqrt(xi * (xi + 1.0)),
          &vertices);
      AddEvenPermutations(
          s1 * xi_squared * phi * std::sqrt(phi * (xi - 1.0 - inv_xi)),
          s2 * phi * std::sqrt(xi + 1.0 - phi),
          s3 * std::sqrt(xi_squared * (1.0 + 2.0 * phi) - phi),
          &vertices);

    } else {
      // Even number of negative signs.

      AddEvenPermutations(
          s1 * xi_squared * phi * std::sqrt(3.0 - xi_squared),
          s2 * xi * phi * std::sqrt(phi * (xi - 1.0 - inv_xi)),
          s3 * phi_squared * inv_xi * std::sqrt(xi * (xi + phi) + 1.0),
          &vertices);

      AddEvenPermutations(
          s1 * std::sqrt(phi * (xi + 2.0) + 2.0),
          s2 * phi * std::sqrt(1.0 - xi + (1.0 + phi) / xi),
          s3 * xi * std::sqrt(xi * (1.0 + phi) - phi),
          &vertices);
    }
  }
  return vertices;
}

Polyhedron SnubDodecahedron() {
  std::vector<vec3> vertices = TwoEdgeLengthSnubDodecahedron();

  CHECK(vertices.size() == 60);
  return MakeConvexOrDie(std::move(vertices),
                         "snubdodecahedron",
                         SYM_ICOSAHEDRAL);
}

Polyhedron TriakisTetrahedron() {
  constexpr double ft = 5.0 / 3.0;
  std::vector<vec3> vertices = {
    vec3{ft, ft, ft},
    vec3{ft, -ft, -ft},
    vec3{-ft, ft, -ft},
    vec3{-ft, -ft, ft},
    vec3{-1, 1, 1},
    vec3{1, -1, 1},
    vec3{1, 1, -1},
    vec3{-1, -1, -1},
  };

  return MakeConvexOrDie(
      std::move(vertices), "triakistetrahedron",
      SYM_TETRAHEDRAL);
}

Polyhedron RhombicDodecahedron() {
  std::vector<vec3> vertices;

  // It contains a cube
  for (uint8_t bits = 0b000; bits < 0b1000; bits++) {
    double s1 = (bits & 0b100) ? -1 : +1;
    double s2 = (bits & 0b010) ? -1 : +1;
    double s3 = (bits & 0b001) ? -1 : +1;
    vertices.emplace_back(s1, s2, s3);
  }

  // With pyramids on
  for (double s : {-2.0, 2.0}) {
    vertices.emplace_back(s, 0.0, 0.0);
    vertices.emplace_back(0.0, s, 0.0);
    vertices.emplace_back(0.0, 0.0, s);
  }

  CHECK(vertices.size() == 14);

  return MakeConvexOrDie(
      std::move(vertices), "rhombicdodecahedron",
      SYM_OCTAHEDRAL);
}

Polyhedron TriakisOctahedron() {
  std::vector<vec3> vertices;

  constexpr double a = std::numbers::sqrt2 - 1.0;

  // It contains a cube
  for (uint8_t bits = 0b000; bits < 0b1000; bits++) {
    double s1 = (bits & 0b100) ? -1 : +1;
    double s2 = (bits & 0b010) ? -1 : +1;
    double s3 = (bits & 0b001) ? -1 : +1;
    vertices.emplace_back(s1 * a, s2 * a, s3 * a);
  }

  for (double s : {-1.0, 1.0}) {
    vertices.emplace_back(s, 0.0, 0.0);
    vertices.emplace_back(0.0, s, 0.0);
    vertices.emplace_back(0.0, 0.0, s);
  }

  CHECK(vertices.size() == 14);

  return MakeConvexOrDie(
      std::move(vertices), "triakisoctahedron",
      SYM_OCTAHEDRAL);
}

Polyhedron TetrakisHexahedron() {
  std::vector<vec3> vertices;

  // It contains a cube
  for (uint8_t bits = 0b000; bits < 0b1000; bits++) {
    double s1 = (bits & 0b100) ? -1 : +1;
    double s2 = (bits & 0b010) ? -1 : +1;
    double s3 = (bits & 0b001) ? -1 : +1;
    vertices.emplace_back(s1, s2, s3);
  }

  for (double s : {-1.5, 1.5}) {
    vertices.emplace_back(s, 0.0, 0.0);
    vertices.emplace_back(0.0, s, 0.0);
    vertices.emplace_back(0.0, 0.0, s);
  }

  CHECK(vertices.size() == 14);

  return MakeConvexOrDie(
      std::move(vertices), "tetrakishexahedron",
      SYM_OCTAHEDRAL);
}

Polyhedron DeltoidalIcositetrahedron() {
  std::vector<vec3> vertices;

  constexpr double a = std::numbers::sqrt2 * 0.5;
  constexpr double b = (2.0 * std::numbers::sqrt2 + 1.0) / 7.0;

  for (double s : {-1.0, 1.0}) {
    vertices.emplace_back(s, 0.0, 0.0);
    vertices.emplace_back(0.0, s, 0.0);
    vertices.emplace_back(0.0, 0.0, s);
  }

  for (uint8_t bits = 0b00; bits < 0b100; bits++) {
    double s1 = (bits & 0b10) ? -1 : +1;
    double s2 = (bits & 0b01) ? -1 : +1;
    vertices.emplace_back(0, s1 * a, s2 * a);
    vertices.emplace_back(s1 * a, 0, s2 * a);
    vertices.emplace_back(s1 * a, s2 * a, 0);
  }

  for (uint8_t bits = 0b000; bits < 0b1000; bits++) {
    double s1 = (bits & 0b100) ? -1 : +1;
    double s2 = (bits & 0b010) ? -1 : +1;
    double s3 = (bits & 0b001) ? -1 : +1;
    vertices.emplace_back(s1 * b, s2 * b, s3 * b);
  }

  CHECK(vertices.size() == 26);

  return MakeConvexOrDie(
      std::move(vertices), "deltoidalicositetrahedron",
      SYM_OCTAHEDRAL);
}

Polyhedron DisdyakisDodecahedron() {
  std::vector<vec3> vertices;

  constexpr double a = 1.0 / (1.0 + 2.0 * std::numbers::sqrt2);
  constexpr double b = 1.0 / (2.0 + 3.0 * std::numbers::sqrt2);
  constexpr double c = 1.0 / (3.0 + 3.0 * std::numbers::sqrt2);

  for (double s : {-a, a}) {
    vertices.emplace_back(s, 0.0, 0.0);
    vertices.emplace_back(0.0, s, 0.0);
    vertices.emplace_back(0.0, 0.0, s);
  }

  for (uint8_t bits = 0b00; bits < 0b100; bits++) {
    double s1 = (bits & 0b10) ? -1 : +1;
    double s2 = (bits & 0b01) ? -1 : +1;
    vertices.emplace_back(0, s1 * b, s2 * b);
    vertices.emplace_back(s1 * b, 0, s2 * b);
    vertices.emplace_back(s1 * b, s2 * b, 0);
  }

  // cube
  for (uint8_t bits = 0b000; bits < 0b1000; bits++) {
    double s1 = (bits & 0b100) ? -1 : +1;
    double s2 = (bits & 0b010) ? -1 : +1;
    double s3 = (bits & 0b001) ? -1 : +1;
    vertices.emplace_back(s1 * c, s2 * c, s3 * c);
  }

  CHECK(vertices.size() == 26);

  return MakeConvexOrDie(
      std::move(vertices), "disdyakisdodecahedron",
      SYM_OCTAHEDRAL);
}

Polyhedron DeltoidalHexecontahedron() {
  std::vector<vec3> vertices;

  // Following https://dmccooey.com/polyhedra/DeltoidalHexecontahedron.txt

  static constexpr double C0 = (5.0 - Sqrt(5.0)) / 4.0;
  static constexpr double C1 = (15.0 + Sqrt(5.0)) / 22.0;
  static constexpr double C2 = Sqrt(5.0) / 2.0;
  static constexpr double C3 = (5.0 + Sqrt(5.0)) / 6.0;
  static constexpr double C4 = (5.0 + 4.0 * Sqrt(5.0)) / 11.0;
  static constexpr double C5 = (5.0 + Sqrt(5.0)) / 4.0;
  static constexpr double C6 = (5.0 + 3.0 * Sqrt(5.0)) / 6.0;
  static constexpr double C7 = (25.0 + 9.0 * Sqrt(5.0)) / 22.0;
  static constexpr double C8 = Sqrt(5.0);

  static_assert(Abs(C0 - 0.690983005625052575897706582817) < 1.0e-10);
  static_assert(Abs(C1 - 0.783457635340899531654962439488) < 1.0e-10);
  static_assert(Abs(C2 - 1.11803398874989484820458683437 ) < 1.0e-10);
  static_assert(Abs(C3 - 1.20601132958329828273486227812 ) < 1.0e-10);
  static_assert(Abs(C4 - 1.26766108272719625323969951590 ) < 1.0e-10);
  static_assert(Abs(C5 - 1.80901699437494742410229341718 ) < 1.0e-10);
  static_assert(Abs(C6 - 1.95136732208322818153792016770 ) < 1.0e-10);
  static_assert(Abs(C7 - 2.05111871806809578489466195539 ) < 1.0e-10);
  static_assert(Abs(C8 - 2.23606797749978969640917366873 ) < 1.0e-10);

  vertices.emplace_back(0.0, 0.0,  C8);
  vertices.emplace_back(0.0, 0.0, -C8);
  vertices.emplace_back( C8, 0.0, 0.0);
  vertices.emplace_back(-C8, 0.0, 0.0);
  vertices.emplace_back(0.0,  C8, 0.0);
  vertices.emplace_back(0.0, -C8, 0.0);
  vertices.emplace_back(0.0,  C1,  C7);
  vertices.emplace_back(0.0,  C1, -C7);
  vertices.emplace_back(0.0, -C1,  C7);
  vertices.emplace_back(0.0, -C1, -C7);
  vertices.emplace_back( C7, 0.0,  C1);
  vertices.emplace_back( C7, 0.0, -C1);
  vertices.emplace_back(-C7, 0.0,  C1);
  vertices.emplace_back(-C7, 0.0, -C1);
  vertices.emplace_back( C1,  C7, 0.0);
  vertices.emplace_back( C1, -C7, 0.0);
  vertices.emplace_back(-C1,  C7, 0.0);
  vertices.emplace_back(-C1, -C7, 0.0);
  vertices.emplace_back( C3, 0.0,  C6);
  vertices.emplace_back( C3, 0.0, -C6);
  vertices.emplace_back(-C3, 0.0,  C6);
  vertices.emplace_back(-C3, 0.0, -C6);
  vertices.emplace_back( C6,  C3, 0.0);
  vertices.emplace_back( C6, -C3, 0.0);
  vertices.emplace_back(-C6,  C3, 0.0);
  vertices.emplace_back(-C6, -C3, 0.0);
  vertices.emplace_back(0.0,  C6,  C3);
  vertices.emplace_back(0.0,  C6, -C3);
  vertices.emplace_back(0.0, -C6,  C3);
  vertices.emplace_back(0.0, -C6, -C3);
  vertices.emplace_back( C0,  C2,  C5);
  vertices.emplace_back( C0,  C2, -C5);
  vertices.emplace_back( C0, -C2,  C5);
  vertices.emplace_back( C0, -C2, -C5);
  vertices.emplace_back(-C0,  C2,  C5);
  vertices.emplace_back(-C0,  C2, -C5);
  vertices.emplace_back(-C0, -C2,  C5);
  vertices.emplace_back(-C0, -C2, -C5);
  vertices.emplace_back( C5,  C0,  C2);
  vertices.emplace_back( C5,  C0, -C2);
  vertices.emplace_back( C5, -C0,  C2);
  vertices.emplace_back( C5, -C0, -C2);
  vertices.emplace_back(-C5,  C0,  C2);
  vertices.emplace_back(-C5,  C0, -C2);
  vertices.emplace_back(-C5, -C0,  C2);
  vertices.emplace_back(-C5, -C0, -C2);
  vertices.emplace_back( C2,  C5,  C0);
  vertices.emplace_back( C2,  C5, -C0);
  vertices.emplace_back( C2, -C5,  C0);
  vertices.emplace_back( C2, -C5, -C0);
  vertices.emplace_back(-C2,  C5,  C0);
  vertices.emplace_back(-C2,  C5, -C0);
  vertices.emplace_back(-C2, -C5,  C0);
  vertices.emplace_back(-C2, -C5, -C0);
  vertices.emplace_back( C4,  C4,  C4);
  vertices.emplace_back( C4,  C4, -C4);
  vertices.emplace_back( C4, -C4,  C4);
  vertices.emplace_back( C4, -C4, -C4);
  vertices.emplace_back(-C4,  C4,  C4);
  vertices.emplace_back(-C4,  C4, -C4);
  vertices.emplace_back(-C4, -C4,  C4);
  vertices.emplace_back(-C4, -C4, -C4);

  CHECK(vertices.size() == 62);
  return MakeConvexOrDie(
      std::move(vertices), "deltoidalhexecontahedron",
      SYM_ICOSAHEDRAL);
}

Polyhedron PentagonalIcositetrahedron() {
  std::vector<vec3> vertices;

  const double tribonacci =
    (1.0 + std::cbrt(19.0 + 3.0 * std::sqrt(33.0)) +
     std::cbrt(19.0 - 3.0 * std::sqrt(33.0))) / 3.0;

  const double tt = tribonacci * tribonacci;
  [[maybe_unused]] const double ttt = tt * tribonacci;

  // cube
  for (uint8_t bits = 0b000; bits < 0b1000; bits++) {
    double s1 = (bits & 0b100) ? -1 : +1;
    double s2 = (bits & 0b010) ? -1 : +1;
    double s3 = (bits & 0b001) ? -1 : +1;
    vertices.emplace_back(s1 * tt, s2 * tt, s3 * tt);
  }

  for (double s : {-1.0, 1.0}) {
    vertices.emplace_back(s * ttt, 0.0, 0.0);
    vertices.emplace_back(0.0, s * ttt, 0.0);
    vertices.emplace_back(0.0, 0.0, s * ttt);
  }

  for (uint8_t bits = 0b000; bits < 0b1000; bits++) {
    double s1 = (bits & 0b100) ? -1 : +1;
    double s2 = (bits & 0b010) ? -1 : +1;
    double s3 = (bits & 0b001) ? -1 : +1;

    if ((std::popcount<uint8_t>(bits) & 1) == 1) {
      // Odd number of negative signs
      AddOddPermutations(
          s1,
          s2 * (2.0 * tribonacci + 1.0),
          s3 * tt,
          &vertices);
    } else {
      // Even number of negative signs.
      AddEvenPermutations(
          s1,
          s2 * (2.0 * tribonacci + 1.0),
          s3 * tt,
          &vertices);
    }
  }

  CHECK(vertices.size() == 38);

  return MakeConvexOrDie(
      std::move(vertices), "pentagonalicositetrahedron",
      SYM_OCTAHEDRAL);
}


static void AddCyclicPermutations(double a, double b, double c,
                                std::vector<vec3> *vertices) {
  // For three elements, the cyclic permutations are the same
  // as the even permutations!
  return AddEvenPermutations(a, b, c, vertices);
}


Polyhedron RhombicTriacontahedron() {
  std::vector<vec3> vertices;

  static constexpr double phi = std::numbers::phi;
  static constexpr double inv_phi = 1.0 / std::numbers::phi;

  // cube
  for (uint8_t bits = 0b000; bits < 0b1000; bits++) {
    double s1 = (bits & 0b100) ? -1 : +1;
    double s2 = (bits & 0b010) ? -1 : +1;
    double s3 = (bits & 0b001) ? -1 : +1;
    vertices.emplace_back(s1, s2, s3);
  }

  for (uint8_t bits = 0b00; bits < 0b100; bits++) {
    double s1 = (bits & 0b10) ? -1 : +1;
    double s2 = (bits & 0b01) ? -1 : +1;
    AddCyclicPermutations(0.0, s1, s2 * phi,
                          &vertices);
    AddCyclicPermutations(0.0, s1 * phi, s2 * inv_phi,
                          &vertices);
  }

  CHECK(vertices.size() == 32);

  return MakeConvexOrDie(
      std::move(vertices), "rhombictriacontahedron",
      SYM_ICOSAHEDRAL);
}


Polyhedron TriakisIcosahedron() {
  std::vector<vec3> vertices;

  static constexpr double phi = std::numbers::phi;
  static constexpr double invphi = 1.0 / std::numbers::phi;
  static constexpr double sppo = Sqrt(phi * phi + 1.0);

  for (uint8_t bits = 0b00; bits < 0b100; bits++) {
    double s1 = (bits & 0b10) ? -1 : +1;
    double s2 = (bits & 0b01) ? -1 : +1;
    vertices.emplace_back(0.0, s1 / sppo, s2 * phi / sppo);
    vertices.emplace_back(s1 / sppo, s2 * phi / sppo, 0.0);
    vertices.emplace_back(s1 * phi / sppo, 0.0, s2 / sppo);
  }

  static constexpr double c = Sqrt(25.0 + 2.0 * Sqrt(5.0)) / 11.0;
  // cube
  for (uint8_t bits = 0b000; bits < 0b1000; bits++) {
    double s1 = (bits & 0b100) ? -1 : +1;
    double s2 = (bits & 0b010) ? -1 : +1;
    double s3 = (bits & 0b001) ? -1 : +1;
    vertices.emplace_back(s1 * c, s2 * c, s3 * c);
  }

  for (uint8_t bits = 0b00; bits < 0b100; bits++) {
    double s1 = (bits & 0b10) ? -1 : +1;
    double s2 = (bits & 0b01) ? -1 : +1;
    vertices.emplace_back(0.0, s1 * phi * c, s2 * invphi * c);
    vertices.emplace_back(s1 * invphi * c, 0.0, s2 * phi * c);
    vertices.emplace_back(s1 * phi * c, s2 * invphi * c, 0.0);
  }

  CHECK(vertices.size() == 32);
  return MakeConvexOrDie(
      std::move(vertices), "triakisicosahedron",
      SYM_ICOSAHEDRAL);
}

Polyhedron PentakisDodecahedron() {
  std::vector<vec3> vertices;

  static constexpr double phi = std::numbers::phi;
  static constexpr double invphi = 1.0 / std::numbers::phi;

  static constexpr double scale = (3.0 * phi + 12.0) / 19.0;

  // This is the convex hull of an icosahedron and dodecahedron,
  // but they need to be the right scale.

  // Icosahedron, scaled down.
  for (uint8_t bits = 0b00; bits < 0b100; bits++) {
    double s1 = (bits & 0b10) ? -1 : +1;
    double s2 = (bits & 0b01) ? -1 : +1;
    AddCyclicPermutations(0.0,
                          s1 * scale,
                          s2 * phi * scale,
                          &vertices);
  }

  // Dodecahedron
  for (uint8_t bits = 0b000; bits < 0b1000; bits++) {
    double s1 = (bits & 0b100) ? -1 : +1;
    double s2 = (bits & 0b010) ? -1 : +1;
    double s3 = (bits & 0b001) ? -1 : +1;
    vertices.emplace_back(s1, s2, s3);
  }
  for (uint8_t bits = 0b00; bits < 0b100; bits++) {
    double s1 = (bits & 0b10) ? -1 : +1;
    double s2 = (bits & 0b01) ? -1 : +1;
    AddCyclicPermutations(s1 * phi, s2 * invphi, 0.0,
                          &vertices);
  }

  CHECK(vertices.size() == 32);
  return MakeConvexOrDie(
      std::move(vertices), "pentakisdodecahedron",
      SYM_ICOSAHEDRAL);
}

Polyhedron DisdyakisTriacontahedron() {
  std::vector<vec3> vertices;

  constexpr double phi = std::numbers::phi;
  constexpr double sqrtp2 = Sqrt(phi + 2.0);
  constexpr double r = 5.0 / (3.0 * phi * sqrtp2);
  constexpr double s = ((7.0 * phi - 6.0) * sqrtp2) / 11.0;

  // They should be close to the quoted values.
  static_assert(Abs(r - 0.5415328270548438) < 1e-12);
  static_assert(Abs(s - 0.9210096876986302) < 1e-12);

  // cube
  for (uint8_t bits = 0b000; bits < 0b1000; bits++) {
    double s1 = (bits & 0b100) ? -1 : +1;
    double s2 = (bits & 0b010) ? -1 : +1;
    double s3 = (bits & 0b001) ? -1 : +1;
    vertices.emplace_back(s1 * r, s2 * r, s3 * r);
  }

  for (double sign : {-1.0, 1.0}) {
    vertices.emplace_back(sign * s, 0.0, 0.0);
    vertices.emplace_back(0.0, sign * s, 0.0);
    vertices.emplace_back(0.0, 0.0, sign * s);
  }

  for (uint8_t bits = 0b00; bits < 0b100; bits++) {
    double s1 = (bits & 0b10) ? -1 : +1;
    double s2 = (bits & 0b01) ? -1 : +1;
    AddCyclicPermutations(0.0,
                          s1 / sqrtp2,
                          s2 * phi / sqrtp2,
                          &vertices);

    AddCyclicPermutations(0.0, s1 * phi * r, s2 * r / phi,
                          &vertices);
  }

  for (uint8_t bits = 0b000; bits < 0b1000; bits++) {
    double s1 = (bits & 0b100) ? -1 : +1;
    double s2 = (bits & 0b010) ? -1 : +1;
    double s3 = (bits & 0b001) ? -1 : +1;
    AddCyclicPermutations(
        s1 * s * phi * 0.5,
        s2 * s * 0.5,
        s3 * s / (2.0 * phi),
        &vertices);
  }

  CHECK(vertices.size() == 62);
  return MakeConvexOrDie(
      std::move(vertices), "disdyakistriacontahedron",
      SYM_ICOSAHEDRAL);
}


Polyhedron PentagonalHexecontahedron() {
  std::vector<vec3> vertices;

  static constexpr double phi = std::numbers::phi;
  static constexpr double x_term = Sqrt(phi - 5.0 / 27.0);
  static constexpr double x =
    Cbrt((phi + x_term) * 0.5) + Cbrt((phi - x_term) * 0.5);

  // From
  // https://dmccooey.com/polyhedra/LpentagonalHexecontahedron.txt

  static constexpr double xx = x * x;

  static constexpr double C0 = phi * Sqrt(3.0 - xx) / 2.0;
  static constexpr double C1 =
      phi * Sqrt((x - 1.0 - (1.0 / x)) * phi) / (2.0 * x);
  static constexpr double C2 = phi * Sqrt((x - 1.0 - (1.0 / x)) * phi) / 2.0;
  static constexpr double C3 = xx * phi * Sqrt(3.0 - xx) / 2.0;
  static constexpr double C4 = phi * Sqrt(1.0 - x + (1.0 + phi) / x) / 2.0;
  static constexpr double C5 = Sqrt(x * (x + phi) + 1.0) / (2.0 * x);
  static constexpr double C6 = Sqrt((x + 2.0) * phi + 2.0) / (2.0 * x);
  static constexpr double C7 =
      Sqrt(-xx * (2.0 + phi) + x * (1.0 + 3.0 * phi) + 4) / 2.0;
  static constexpr double C8 = (1.0 + phi) * Sqrt(1.0 + (1.0 / x)) / (2.0 * x);
  static constexpr double C9 =
      Sqrt(2.0 + 3.0 * phi - 2.0 * x + (3.0 / x)) / 2.0;
  static constexpr double C10 =
      Sqrt(xx * (392.0 + 225.0 * phi) + x * (249.0 + 670.0 * phi) +
           (470.0 + 157.0 * phi)) / 62.0;
  static constexpr double C11 = phi * Sqrt(x * (x + phi) + 1.0) / (2.0 * x);
  static constexpr double C12 = phi * Sqrt(xx + x + 1.0 + phi) / (2.0 * x);
  static constexpr double C13 =
      phi * Sqrt(xx + 2.0 * x * phi + 2.0) / (2.0 * x);
  static constexpr double C14 = Sqrt(xx * (1.0 + 2.0 * phi) - phi) / 2.0;
  static constexpr double C15 = phi * Sqrt(xx + x) / 2.0;
  static constexpr double C16 =
      (phi * phi * phi) * Sqrt(x * (x + phi) + 1.0) / (2.0 * xx);
  static constexpr double C17 =
      Sqrt(xx * (617.0 + 842.0 * phi) + x * (919.0 + 1589.0 * phi) +
           (627.0 + 784.0 * phi)) / 62.0;
  static constexpr double C18 =
      (phi * phi) * Sqrt(x * (x + phi) + 1.0) / (2.0 * x);
  static constexpr double C19 = phi * Sqrt(x * (x + phi) + 1.0) / 2.0;

  // Check that the computed values are very close to their quoted
  // value.
  CHECK(std::abs(C0 - 0.192893711352359022108262546061) < 1e-10) << C0;
  CHECK(std::abs(C1 - 0.218483370127321224365534157111) < 1e-10) << C1;
  CHECK(std::abs(C2 - 0.374821658114562295266609516608) < 1e-10) << C2;
  CHECK(std::abs(C3 - 0.567715369466921317374872062669) < 1e-10) << C3;
  CHECK(std::abs(C4 - 0.728335176957191477360671629838) < 1e-10) << C4;
  CHECK(std::abs(C5 - 0.755467260516595579705585253517) < 1e-10) << C5;
  CHECK(std::abs(C6 - 0.824957552676275846265811111988) < 1e-10) << C6;
  CHECK(std::abs(C7 - 0.921228888309550499468934175898) < 1e-10) << C7;
  CHECK(std::abs(C8 - 0.959987701391583803994339068107) < 1e-10) << C8;
  CHECK(std::abs(C9 - 1.13706613386050418840961998424) < 1e-10) << C9;
  CHECK(std::abs(C10 - 1.16712343647533397917215468549) < 1e-10) << C10;
  CHECK(std::abs(C11 - 1.22237170490362309266282747264) < 1e-10) << C11;
  CHECK(std::abs(C12 - 1.27209628257581214613814794036) < 1e-10) << C12;
  CHECK(std::abs(C13 - 1.52770307085850512136921113078) < 1e-10) << C13;
  CHECK(std::abs(C14 - 1.64691794069037444140475745697) < 1e-10) << C14;
  CHECK(std::abs(C15 - 1.74618644098582634573474528789) < 1e-10) << C15;
  CHECK(std::abs(C16 - 1.86540131081769566577029161408) < 1e-10) << C16;
  CHECK(std::abs(C17 - 1.88844538928366915418351670356) < 1e-10) << C17;
  CHECK(std::abs(C18 - 1.97783896542021867236841272616) < 1e-10) << C18;
  CHECK(std::abs(C19 - 2.097053835252087992403959052348) < 1e-10) << C19;

  vertices.emplace_back( -C0,  -C1, -C19);
  vertices.emplace_back( -C0,   C1,  C19);
  vertices.emplace_back(  C0,   C1, -C19);
  vertices.emplace_back(  C0,  -C1,  C19);
  vertices.emplace_back(-C19,  -C0,  -C1);
  vertices.emplace_back(-C19,   C0,   C1);
  vertices.emplace_back( C19,   C0,  -C1);
  vertices.emplace_back( C19,  -C0,   C1);
  vertices.emplace_back( -C1, -C19,  -C0);
  vertices.emplace_back( -C1,  C19,   C0);
  vertices.emplace_back(  C1,  C19,  -C0);
  vertices.emplace_back(  C1, -C19,   C0);
  vertices.emplace_back( 0.0,  -C5, -C18);
  vertices.emplace_back( 0.0,  -C5,  C18);
  vertices.emplace_back( 0.0,   C5, -C18);
  vertices.emplace_back( 0.0,   C5,  C18);
  vertices.emplace_back(-C18,  0.0,  -C5);
  vertices.emplace_back(-C18,  0.0,   C5);
  vertices.emplace_back( C18,  0.0,  -C5);
  vertices.emplace_back( C18,  0.0,   C5);
  vertices.emplace_back( -C5, -C18,  0.0);
  vertices.emplace_back( -C5,  C18,  0.0);
  vertices.emplace_back(  C5, -C18,  0.0);
  vertices.emplace_back(  C5,  C18,  0.0);
  vertices.emplace_back(-C10,  0.0, -C17);
  vertices.emplace_back(-C10,  0.0,  C17);
  vertices.emplace_back( C10,  0.0, -C17);
  vertices.emplace_back( C10,  0.0,  C17);
  vertices.emplace_back(-C17, -C10,  0.0);
  vertices.emplace_back(-C17,  C10,  0.0);
  vertices.emplace_back( C17, -C10,  0.0);
  vertices.emplace_back( C17,  C10,  0.0);
  vertices.emplace_back( 0.0, -C17, -C10);
  vertices.emplace_back( 0.0, -C17,  C10);
  vertices.emplace_back( 0.0,  C17, -C10);
  vertices.emplace_back( 0.0,  C17,  C10);
  vertices.emplace_back( -C3,   C6, -C16);
  vertices.emplace_back( -C3,  -C6,  C16);
  vertices.emplace_back(  C3,  -C6, -C16);
  vertices.emplace_back(  C3,   C6,  C16);
  vertices.emplace_back(-C16,   C3,  -C6);
  vertices.emplace_back(-C16,  -C3,   C6);
  vertices.emplace_back( C16,  -C3,  -C6);
  vertices.emplace_back( C16,   C3,   C6);
  vertices.emplace_back( -C6,  C16,  -C3);
  vertices.emplace_back( -C6, -C16,   C3);
  vertices.emplace_back(  C6, -C16,  -C3);
  vertices.emplace_back(  C6,  C16,   C3);
  vertices.emplace_back( -C2,  -C9, -C15);
  vertices.emplace_back( -C2,   C9,  C15);
  vertices.emplace_back(  C2,   C9, -C15);
  vertices.emplace_back(  C2,  -C9,  C15);
  vertices.emplace_back(-C15,  -C2,  -C9);
  vertices.emplace_back(-C15,   C2,   C9);
  vertices.emplace_back( C15,   C2,  -C9);
  vertices.emplace_back( C15,  -C2,   C9);
  vertices.emplace_back( -C9, -C15,  -C2);
  vertices.emplace_back( -C9,  C15,   C2);
  vertices.emplace_back(  C9,  C15,  -C2);
  vertices.emplace_back(  C9, -C15,   C2);
  vertices.emplace_back( -C7,  -C8, -C14);
  vertices.emplace_back( -C7,   C8,  C14);
  vertices.emplace_back(  C7,   C8, -C14);
  vertices.emplace_back(  C7,  -C8,  C14);
  vertices.emplace_back(-C14,  -C7,  -C8);
  vertices.emplace_back(-C14,   C7,   C8);
  vertices.emplace_back( C14,   C7,  -C8);
  vertices.emplace_back( C14,  -C7,   C8);
  vertices.emplace_back( -C8, -C14,  -C7);
  vertices.emplace_back( -C8,  C14,   C7);
  vertices.emplace_back(  C8,  C14,  -C7);
  vertices.emplace_back(  C8, -C14,   C7);
  vertices.emplace_back( -C4,  C12, -C13);
  vertices.emplace_back( -C4, -C12,  C13);
  vertices.emplace_back(  C4, -C12, -C13);
  vertices.emplace_back(  C4,  C12,  C13);
  vertices.emplace_back(-C13,   C4, -C12);
  vertices.emplace_back(-C13,  -C4,  C12);
  vertices.emplace_back( C13,  -C4, -C12);
  vertices.emplace_back( C13,   C4,  C12);
  vertices.emplace_back(-C12,  C13,  -C4);
  vertices.emplace_back(-C12, -C13,   C4);
  vertices.emplace_back( C12, -C13,  -C4);
  vertices.emplace_back( C12,  C13,   C4);
  vertices.emplace_back(-C11, -C11, -C11);
  vertices.emplace_back(-C11, -C11,  C11);
  vertices.emplace_back(-C11,  C11, -C11);
  vertices.emplace_back(-C11,  C11,  C11);
  vertices.emplace_back( C11, -C11, -C11);
  vertices.emplace_back( C11, -C11,  C11);
  vertices.emplace_back( C11,  C11, -C11);
  vertices.emplace_back( C11,  C11,  C11);

  CHECK(vertices.size() == 92);
  return MakeConvexOrDie(
      std::move(vertices), "pentagonalhexecontahedron",
      SYM_ICOSAHEDRAL);
}

