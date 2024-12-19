
#include "polyhedra.h"

#include <limits>
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

Faces::Faces(int num_vertices, std::vector<std::vector<int>> v_in) :
  v(std::move(v_in)) {

  std::vector<std::unordered_set<int>> collated(num_vertices);
  for (const std::vector<int> &face : v) {
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
    CHECK(!neighbors[i].empty());
  }

  // And triangulate. Since the faces are convex, we can
  // just do this by creating triangle fans.
  for (const std::vector<int> &face : v) {
    CHECK(face.size() >= 3);
    int p0 = face[0];
    for (int i = 1; i + 1 < face.size(); i++) {
      triangulation.emplace_back(p0, face[i], face[i + 1]);
    }
  }
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
inline double PointLineDistance(
    // Line segment
    const vec2 &v0, const vec2 &v1,
    // Point to test
    const vec2 &pt) {

  const vec2 c = ClosestPointOnSegment(v0, v1, pt);
  const double dx = pt.x - c.x;
  const double dy = pt.y - c.y;
  return sqrt(dx * dx + dy * dy);
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

double DistanceToEdge(const vec2 &v0, const vec2 &v1, const vec2 &p) {
  return PointLineDistance(v0, v1, p);
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

double DistanceToHull(
    const std::vector<vec2> &points, const std::vector<int> &hull,
    const vec2 &pt) {

  std::optional<double> best_dist;
  for (int i = 0; i < hull.size(); i++) {
    const vec2 &v0 = points[hull[i]];
    const vec2 &v1 = points[hull[(i + 1) % hull.size()]];

    double dist = DistanceToEdge(v0, v1, pt);
    if (!best_dist.has_value() || dist < best_dist.value()) {
      best_dist = {dist};
    }
  }
  CHECK(best_dist.has_value());
  return best_dist.value();
}

double DistanceToMesh(const Mesh2D &mesh, const vec2 &pt) {
  std::optional<double> best_dist;
  for (const std::vector<int> &polygon : mesh.faces->v) {
    double dist = DistanceToHull(mesh.vertices, polygon, pt);
    if (!best_dist.has_value() || dist < best_dist.value()) {
      best_dist = {dist};
    }
  }
  CHECK(best_dist.has_value());
  return best_dist.value();
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
//  - Fixes some bugs relating to exactly equal or input points
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
    // infinite loops. Removing them does not affect the hull.
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

  CHECK(a != b);

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


bool PointInPolygon(const vec2 &point,
                    const std::vector<vec2> &vertices,
                    const std::vector<int> &polygon) {
  int winding_number = 0;
  for (int i = 0; i < polygon.size(); i++) {
    const vec2 &p0 = vertices[polygon[i]];
    const vec2 &p1 = vertices[polygon[(i + 1) % polygon.size()]];

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

// via https://en.wikipedia.org/wiki/Shoelace_formula
double AreaOfHull(const Mesh2D &mesh, const std::vector<int> &hull) {
  if (hull.size() < 3) return 0.0;

  double area = 0.0;
  // Iterate through the polygon vertices, using the shoelace formula.
  for (size_t i = 0; i < hull.size(); i++) {
    const vec2 &v0 = mesh.vertices[hull[i]];
    const vec2 &v1 = mesh.vertices[hull[(i + 1) % hull.size()]];
    area += v0.x * v1.y - v1.x * v0.y;
  }

  // Sign depends on the winding order, but we always want a positive
  // area.
  return std::abs(area * 0.5);
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

// Take all planes where all of the other vertices
// are on one side. (Basically, the 3D convex hull.)
// This is not fast but it should work for any convex polyhedron,
// so it's a clean way to generate a wide variety from just the
// vertices.
static Polyhedron ConvexPolyhedronFromVertices(
    std::vector<vec3> vertices, const char *name = "") {
  static constexpr int VERBOSE = 1;

  // All faces (as a set of vertices) we've already found. The
  // vertices in the face have not yet been ordered; they appear in
  // ascending sorted order.
  std::unordered_set<std::vector<int>, Hashing<std::vector<int>>>
    all_faces;

  // Given a plane defined by distinct points (v0, v1, v2), classify
  // all of the other points as either above, below, or on the plane.
  // If there are nonzero points both above and below, then this is
  // not a face; return nullopt. Otherwise, return the indices of the
  // vertices on the face, which will include at least i, j, and k.
  // The vertices are returned in sorted order (by index), which may
  // not be a proper winding for the polygon.
  auto GetFace = [&vertices](int i, int j, int k) ->
    std::optional<std::vector<int>> {
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

    CHECK(!(below && above));
    CHECK(below || above) << "This would only happen if we had "
        "a degenerate, volumeless polyhedron, which we do not.";

    coplanar.push_back(i);
    coplanar.push_back(j);
    coplanar.push_back(k);
    std::sort(coplanar.begin(), coplanar.end());
    return coplanar;
  };

  if (VERBOSE > 0) {
    printf("%s: There are %d vertices.\n",
           name, (int)vertices.size());
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

  if (VERBOSE > 0) {
    printf("%s: There are %d distinct faces.\n",
           name, (int)all_faces.size());
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

  Faces *faces = new Faces(vertices.size(), std::move(fs));
  return Polyhedron{
    .vertices = std::move(vertices),
    .faces = faces,
    .name = name,
  };
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

  return ConvexPolyhedronFromVertices(std::move(vertices), "snubcube");
}

Polyhedron Tetrahedron() {
  std::vector<vec3> vertices{
    vec3{1.0,   1.0,  1.0},
    vec3{1.0,  -1.0, -1.0},
    vec3{-1.0,  1.0, -1.0},
    vec3{-1.0, -1.0,  1.0},
  };

  return ConvexPolyhedronFromVertices(std::move(vertices), "tetrahedron");
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
  return ConvexPolyhedronFromVertices(std::move(vertices), "octahedron");
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

  return ConvexPolyhedronFromVertices(std::move(vertices), "icosahedron");
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
  return ConvexPolyhedronFromVertices(
      std::move(vertices), "cuboctahedron");
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
  return ConvexPolyhedronFromVertices(
      std::move(vertices), "rhombicuboctahedron");
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
  return ConvexPolyhedronFromVertices(
      std::move(vertices), "truncatedcuboctahedron");
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
  return ConvexPolyhedronFromVertices(
      std::move(vertices), "icosidodecahedron");
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
  return ConvexPolyhedronFromVertices(
      std::move(vertices), "truncateddodecahedron");
}

Polyhedron TruncatedIcosahedron() {
  std::vector<vec3> vertices;

  // Derive from the icosahedron.
  Polyhedron ico = Icosahedron();
  for (int i = 0; i < ico.vertices.size(); i++) {
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
  return ConvexPolyhedronFromVertices(
      std::move(vertices), "truncatedicosahedron");
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
  return ConvexPolyhedronFromVertices(
      std::move(vertices), "truncatedicosidodecahedron");
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
  return ConvexPolyhedronFromVertices(std::move(vertices),
                                      "rhombicosidodecahedron");
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
  return ConvexPolyhedronFromVertices(std::move(vertices),
                                      "truncatedtetrahedron");
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
  return ConvexPolyhedronFromVertices(std::move(vertices),
                                      "truncatedcube");
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
  return ConvexPolyhedronFromVertices(std::move(vertices),
                                      "truncatedoctahedron");
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
  return ConvexPolyhedronFromVertices(std::move(vertices),
                                      "snubdodecahedron");
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

  return ConvexPolyhedronFromVertices(
      std::move(vertices), "triakistetrahedron");
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

  return ConvexPolyhedronFromVertices(
      std::move(vertices), "rhombicdodecahedron");
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

  return ConvexPolyhedronFromVertices(
      std::move(vertices), "triakisoctahedron");
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

  return ConvexPolyhedronFromVertices(
      std::move(vertices), "tetrakishexahedron");
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

  return ConvexPolyhedronFromVertices(
    std::move(vertices), "deltoidalicositetrahedron");
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

  return ConvexPolyhedronFromVertices(
      std::move(vertices), "disdyakisdodecahedron");
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

  return ConvexPolyhedronFromVertices(
      std::move(vertices), "pentagonalicositetrahedron");
}


static void AddCyclicPermutations(double a, double b, double c,
                                std::vector<vec3> *vertices) {
  // For three elements, the cyclic permutatinos are the same
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

  return ConvexPolyhedronFromVertices(
      std::move(vertices), "rhombictriacontahedron");
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
  return ConvexPolyhedronFromVertices(
      std::move(vertices), "triakisicosahedron");
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
  return ConvexPolyhedronFromVertices(
      std::move(vertices), "pentakisdodecahedron");
}

Polyhedron DisdyakisTriacontahedron() {
  std::vector<vec3> vertices;

  constexpr double phi = std::numbers::phi;
  constexpr double sqrtp2 = Sqrt(phi + 2.0);
  constexpr double r = 5.0 / (3.0 * phi * sqrtp2);
  constexpr double s = ((7.0 * phi - 6.0) * sqrtp2) / 11.0;

  // cube
  for (uint8_t bits = 0b000; bits < 0b1000; bits++) {
    double s1 = (bits & 0b100) ? -1 : +1;
    double s2 = (bits & 0b010) ? -1 : +1;
    double s3 = (bits & 0b001) ? -1 : +1;
    vertices.emplace_back(s1 * r, s2 * r, s3 * r);
  }

  for (double s : {-1.0, 1.0}) {
    vertices.emplace_back(s, 0.0, 0.0);
    vertices.emplace_back(0.0, s, 0.0);
    vertices.emplace_back(0.0, 0.0, s);
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
  return ConvexPolyhedronFromVertices(
      std::move(vertices), "disdyakistriacontahedron");
}


Polyhedron PentagonalHexecontahedron() {
  std::vector<vec3> vertices;
#if 0

  static constexpr double phi = std::numbers::phi;
  static constexpr double invphi = 1.0 / phi;
  static constexpr double a = 1.0 / Sqrt(phi * phi + 1.0);

  // We need one of the roots of this polynomial, near
  // 0.95369785.
  static constexpr double r =
    Solve(0.9536,
          [](double x) {
            double xx = x * x;
            double x4 = xx * xx;
            double x6 = x4 * xx;
            double x8 = x4 * x4;
            double x10 = x6 * x4;
            double x12 = x6 * x6;
            return 700569.0 - 1795770.0 * xx +
              1502955.0 * x4 - 423900.0 * x6 +
              14175.0 * x8 - 2250.0 * x10 + 125.0 * x12;
          },
          [](double x) {
            double xx = x * x;
            double x4 = xx * xx;
            double x6 = x4 * xx;
            double x8 = x4 * x4;
            double x10 = x6 * x4;

            return 60.0 * x * (25.0 * x10 - 375.0 * x8 + 1890.0 * x6 -
                               42390.0 * x4 + 100197 * xx - 59859.0);
          });

  static constexpr double roverc3 = r / Cbrt(3.0);

  for (uint8_t bits = 0b00; bits < 0b100; bits++) {
    double s1 = (bits & 0b10) ? -1 : +1;
    double s2 = (bits & 0b01) ? -1 : +1;
    vertices.emplace_back(0.0, s1 * a, s2 * phi * a);
    vertices.emplace_back(s1 * a, s2 * phi * a, 0.0);
    vertices.emplace_back(s1 * phi * a, 0.0, s2 * a);
  }

  // Dodecahedron:
  for (uint8_t bits = 0b000; bits < 0b1000; bits++) {
    double s1 = (bits & 0b100) ? -1 : +1;
    double s2 = (bits & 0b010) ? -1 : +1;
    double s3 = (bits & 0b001) ? -1 : +1;
    vertices.emplace_back(s1 * roverc3, s2 * roverc3, s3 * roverc3);
  }

  for (uint8_t bits = 0b00; bits < 0b100; bits++) {
    double s1 = (bits & 0b10) ? -1 : +1;
    double s2 = (bits & 0b01) ? -1 : +1;
    vertices.emplace_back(0.0, s1 * phi * roverc3, s2 * invphi * roverc3);
    vertices.emplace_back(s1 * invphi * roverc3, 0.0, s2 * phi * roverc3);
    vertices.emplace_back(s1 * phi * roverc3, s2 * invphi * roverc3, 0.0);
  }

  // Can get this by solving a polynomial.
  static constexpr double circumradius =
    2.155837375115639701836629076693058277016851219;

  // Scale unit (circumradius) snub dodecahedron by R.
  // Here we have one with edge length 2.
  std::vector<vec3> snub_vertices = TwoEdgeLengthSnubDodecahedron();
  for (vec3 v : snub_vertices) {
    // First, transform to unit edge length.
    v *= 0.5;
    // This would have the circumradius above, so scale it down
    // to unit circumradius.
    v /= circumradius;
    printf("%s\n", VecString(v).c_str());
    // Now we want it to have circumradius r.
    v *= r;
    vertices.push_back(v);
  }

  // But it doesn't work! :(
  // This may be because the orientations are not consistent for
  // this formulation (I did get them from different places), or
  // perhaps I just have a bug. (Likely the known bug in SqrtSolve!)
#endif

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
  return ConvexPolyhedronFromVertices(
      std::move(vertices), "pentagonalhexecontahedron");
}
