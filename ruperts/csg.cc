
#include "csg.h"

#include <set>
#include <format>
#include <limits>
#include <string_view>
#include <tuple>
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

#include "polyhedra.h"
#include "ansi.h"
#include "base/logging.h"
#include "base/stringprintf.h"
#include "hashing.h"
#include "set-util.h"
#include "util.h"
#include "point-map.h"

#include "yocto_matht.h"

namespace {
// TODO: Use this throughout.
static inline vec2 Two(const vec3 &v) {
  return vec2{v.x, v.y};
}

// Project the point pt, which should like on the 2D segment (a, b)
// in the xy plane, such that it lies on the 3D segment (a, b).
static std::optional<vec3> PointToLine(const vec3 &a, const vec3 &b,
                                       const vec2 &pt2) {
  vec2 a2 = Two(a);
  vec2 b2 = Two(b);

  double numer = length(pt2 - a2);
  double denom = length(b2 - a2);

  // This would mean that (a, b) is basically perpendicular to xy.
  if (std::abs(denom) < 1.0e-10) {
    return std::nullopt;
  }

  double f = numer / denom;
  return {a + f * (b - a)};
}

// Project the point pt along z to the triangle (plane)
// defined by (a, b, c). The returned point has the same x,y position
// as pt, but a z coordinate that places it on that plane.
//
// This returns a result even if the point is not in the triangle.
// It returns nullopt when the plane is perpendicular to xy.
static std::optional<vec3> PointToPlane(
    const vec3 &a, const vec3 &b, const vec3 &c,
    const vec2 &pt) {
  vec2 a2 = {a.x, a.y};
  vec2 b2 = {b.x, b.y};
  vec2 c2 = {c.x, c.y};

  // Barycentric coordinates in 2D.
  vec2 v0 = b2 - a2;
  vec2 v1 = c2 - a2;
  vec2 v2 = pt - a2;

  double d00 = yocto::dot(v0, v0);
  double d01 = yocto::dot(v0, v1);
  double d11 = yocto::dot(v1, v1);
  double d20 = yocto::dot(v2, v0);
  double d21 = yocto::dot(v2, v1);

  double denom = d00 * d11 - d01 * d01;

  // If denom is zero (or close) then the triangle is degenerate in 2D.
  if (std::abs(denom) < 1.0e-10) {
    return std::nullopt;
  }

  double v = (d11 * d20 - d01 * d21) / denom;
  double w = (d00 * d21 - d01 * d20) / denom;
  double u = 1.0 - v - w;

  // The weights u, v, w can also be used to interpolate in 3D, since
  // these are linear interpolations and the projection from 3D to 2D
  // is also linear.
  double z = u * a.z + v * b.z + w * c.z;

  return {vec3{pt.x, pt.y, z}};
}

static std::optional<vec3> PointOnSegment(
    // First segment
    const vec3 &a, const vec3 &b,
    // Test point
    const vec3 &p) {
  vec3 ab = b - a;
  vec3 ap = p - a;

  vec3 cx = cross(ab, ap);
  // We need the cross product to be approximately
  // zero for them to be considered colinear.
  if (length(cx) > 0.00001) {
    return std::nullopt;
  }

  // parameter interpolating from a to b.
  double t = dot(ap, ab) / length_squared(ab);
  if (t < 0.0 || t > 1.0) return std::nullopt;
  return (a + t * ab);
}

static std::optional<vec2> LineIntersection(
    // First segment
    const vec2 &p0, const vec2 &p1,
    // Second segment
    const vec2 &p2, const vec2 &p3) {

  const vec2 s1 = p1 - p0;
  const vec2 s2 = p3 - p2;
  const vec2 m = p0 - p2;

  const double denom = s1.x * s2.y - s2.x * s1.y;
  // Note if denom is 0 or close to it, we will just
  // get an enormous (or maybe nan) s. This will not
  // fall into the interval [0, 1].

  const double s = (s1.x * m.y - s1.y * m.x) / denom;

  if (s >= 0.0 && s <= 1.0) {
    const double t = (s2.x * m.y - s2.y * m.x) / denom;

    if (t >= 0.0 && t <= 1.0) {
      return {vec2{p0 + (t * s1)}};
    }
  }
  return std::nullopt;
}

bool TriangleAndPolygonIntersect(
    const vec2 &a, const vec2 &b, const vec2&c,
    const std::vector<vec2> &polygon) {
  for (int i = 0; i < polygon.size(); i++) {
    const vec2 &v0 = polygon[i];
    const vec2 &v1 = polygon[(i + 1) % polygon.size()];

    if (LineIntersection(a, b, v0, v1).has_value() ||
        LineIntersection(b, c, v0, v1).has_value() ||
        LineIntersection(c, a, v0, v1).has_value()) {
      return true;
    }
  }
  return false;
}

// Input triangle must have a < b < c.
// If the triangle has the edge, returns the vertex that's not on that
// edge.
static std::optional<int>
TriangleWithEdge(const std::tuple<int, int, int> &tri,
                 int aa, int bb) {
  if (aa > bb) std::swap(aa, bb);
  const auto &[a, b, c] = tri;
  if (aa == a && bb == b) return {c};
  if (aa == a && bb == c) return {b};
  if (aa == b && bb == c) return {a};
  return std::nullopt;
}

static bool IsAVertex(const std::tuple<int, int, int> &tri,
                      int i) {
  const auto &[a, b, c] = tri;
  return a == i || b == i || c == i;
}

struct HoleMaker {
  // Indexed set of points.
  std::vector<vec3> points;
  // Point map so that we can get the index of a vertex
  // that's already been added. This also merges points
  // that are less than some epsilon from the first
  // one inserted in the vicinity.
  PointMap3<int> point_index;
  std::vector<std::tuple<int, int, int>> out_triangles;

  // Return the point's index if we already have it (or a point
  // very close to it).
  std::optional<int> GetPoint(const vec3 &pt) {
    return point_index.Get(pt);
  }

  int AddPoint(const vec3 &pt) {
    if (auto io = GetPoint(pt)) {
      return io.value();
    }

    int idx = points.size();
    points.push_back(pt);
    point_index.Add(pt, idx);
    return idx;
  }

  // Add a triangle as indices of its vertices. We insert them in
  // sorted order so that it's easier to find edges and so on.
  static void AddTriangleTo(std::vector<std::tuple<int, int, int>> *out,
                            int a, int b, int c) {
    std::vector<int> v;
    v.push_back(a);
    v.push_back(b);
    v.push_back(c);
    std::sort(v.begin(), v.end());
    CHECK(v.size() == 3);
    a = v[0];
    b = v[1];
    c = v[2];
    CHECK(a < b && b < c) << std::format("{} {} {}", a, b, c);
    out->emplace_back(a, b, c);
  }

  void SortPointIndicesByZ(std::vector<int> *indices) {
    std::sort(indices->begin(), indices->end(),
              [this](int a, int b) {
                CHECK(a >= 0 && b >= 0 &&
                      a < points.size() &&
                      b < points.size()) <<
                  std::format("a: {}, b: {}, points.size: {}",
                              a, b, points.size());
                return points[a].z < points[b].z;
              });
  }

  // Triangles that intersect or abut the hole polygon. These
  // vertices are indexes into points.
  std::vector<std::tuple<int, int, int>> work_triangles;

  // In any work triangle.
  bool HasEdge(int u, int v) {
    if (u == v) return false;
    // Triangles are stored with a < b < c.
    if (u > v) std::swap(u, v);
    for (const auto &[a, b, c] : work_triangles) {
      if (a == u && (b == v || c == v)) return true;
      if (b == u && c == v) return true;
    }
    return false;
  }

  const std::vector<vec2> input_polygon;
  // Hole, as point indices (top, bottom).
  std::vector<std::pair<int, int>> hole;

  HoleMaker(const Polyhedron &polyhedron,
            const std::vector<vec2> &polygon) : input_polygon(polygon) {

    for (int idx = 0; idx < polyhedron.vertices.size(); idx++) {
      const vec3 &v = polyhedron.vertices[idx];
      // We would just need to remap the triangles.
      CHECK(AddPoint(v) == idx) << "Points from the input polyhedron "
        "were merged. This can be handled, but isn't handled yet.";
    }

    // First, classify each triangle as entirely inside, entirely outside,
    // or intersecting. The intersecting triangles will be handled in
    // the next loop.
    for (const auto &[a, b, c] : polyhedron.faces->triangulation) {
      vec2 v0 = vec2(polyhedron.vertices[a].x, polyhedron.vertices[a].y);
      vec2 v1 = vec2(polyhedron.vertices[b].x, polyhedron.vertices[b].y);
      vec2 v2 = vec2(polyhedron.vertices[c].x, polyhedron.vertices[c].y);

      int count = 0;
      for (const vec2 &v : {v0, v1, v2}) {
        count += PointInPolygon(v, polygon) ? 1 : 0;
      }

      if (count == 3) {
        // If all are inside, then it cannot intersect the hole (convexity).
        // We just discard these.
      } else if (TriangleAndPolygonIntersect(v0, v1, v2, polygon)) {
        // Might need to be bisected below.
        CHECK(a != b && b != c && a != c);
        AddTriangleTo(&work_triangles, a, b, c);
      } else {
        // If the triangle is entirely outside, then we persist it untouched.
        CHECK(a != b && b != c && a != c);
        AddTriangleTo(&out_triangles, a, b, c);
      }
    }
  }

  // Project a 2D point through the mesh. It can intersect the interior
  // of a triangle, or an existing vertex, or an existing edge. In each
  // case, alter the mesh as appropriate so that the intersection is an
  // actual vertex. Returns the vector of vertex indices, sorted by
  // their z coordinate.

  std::vector<int> ProjectThroughMesh(const vec2 &pt) {
    std::vector<std::tuple<int, int, int>> new_triangles;
    new_triangles.reserve(work_triangles.size());

    // The intersection points. This is a set because when we hit
    // a vertex or edge, multiple triangles are typically implicated.
    std::set<int> new_points;

    // If an edge is already split, this gives the vertex index that
    // should be inserted along that edge on other triangles.
    std::unordered_map<std::pair<int, int>, int,
      Hashing<std::pair<int, int>>> already_split;

    // Returns (a, b, c, d) where a-b is the edge to be split, c
    // is the other vertex in the existing triangle, and d is the
    // new point to add.
    auto GetAlreadySplit = [&](int a, int b, int c) ->
      std::optional<std::tuple<int, int, int, int>> {
        CHECK(a < b && b < c && a < c);
        {
          auto it = already_split.find(std::make_pair(a, b));
          if (it != already_split.end()) return {{a, b, c, it->second}};
        }
        {
          auto it = already_split.find(std::make_pair(b, c));
          if (it != already_split.end()) return {{b, c, a, it->second}};
        }
        {
          auto it = already_split.find(std::make_pair(a, c));
          if (it != already_split.end()) return {{a, c, b, it->second}};
        }
        return std::nullopt;
      };

    for (const auto &tri : work_triangles) {
      const auto &[a, b, c] = tri;
      const vec3 &va = points[a];
      const vec3 &vb = points[b];
      const vec3 &vc = points[c];

      const vec2 &va2 = {va.x, va.y};
      const vec2 &vb2 = {vb.x, vb.y};
      const vec2 &vc2 = {vc.x, vc.y};

      // The location where the point would intersect this
      // triangle (but it may be outside it).
      auto p3o = PointToPlane(va, vb, vc, pt);

      if (!p3o.has_value()) {
        // This means that the triangle is perpendicular to the
        // xy plane. We can't create intersections with such
        // triangles, because they would be edges, not vertices.
        //
        // TODO: We could check here that the point is not on the
        // resulting edge (and error out).
        new_triangles.push_back(tri);
        continue;
      }

      const vec3 p3 = p3o.value();

      // If we already have this point, then that's going to be
      // the result (and we don't need to check anything below).
      if (std::optional<int> p = GetPoint(p3)) {
        // The typical case here would be that we intersected
        // the vertex of a triangle, but it's also possible that
        // we just coincidentally ended up at a vertex (the
        // point p3 is not necessarily in or on the triangle;
        // it's just on the plane). In any case that will be an
        // intersection with the surface, and we don't need to
        // do anything except record it.
        new_points.insert(p.value());
        new_triangles.push_back(tri);
        continue;
      }

      // Does this triangle have an edge that has already been
      // split? If so, we need to do the same split.
      if (const auto so = GetAlreadySplit(a, b, c)) {
        // a-b is the edge to split, c the existing other point;
        // d the point to insert.
        const auto &[a, b, c, d] = so.value();
        AddTriangleTo(&new_triangles, a, c, d);
        AddTriangleTo(&new_triangles, b, c, d);
        // (and discard the original triangle)
        // But that's all we need to do to deal with this triangle.
        continue;
      }

      // Record a new intersection on the edge a-b, with a<b,
      // and the other existing vertex c, at the point p3.
      auto IntersectsEdge =
        [this, &already_split, &new_triangles, &tri, &new_points](
            int a, int b, int c, const vec3 &p3) {
          CHECK(a < b);
          const int d = AddPoint(p3);
          // If it's actually one of the vertices, we're
          // already done.
          if (d == a || d == b) {
            new_points.insert(d);
            new_triangles.push_back(tri);
            return;
          }
          // This may be possible. Just do as above?
          CHECK(d != c) << "Degenerate triangle.";
          // Split the triangle.
          AddTriangleTo(&new_triangles, a, c, d);
          AddTriangleTo(&new_triangles, b, c, d);
          new_points.insert(d);
          CHECK(!already_split.contains({a, b}));
          already_split[{a, b}] = d;
          return;
        };

      // Does the point lie on an edge of the triangle?
      // Note: We recompute the intersection point, which is
      // logically the same but will be numerically closer
      // to the actual edge (doesn't depend on other vertex,
      // for example). But currently using the original
      // projected point here, since we already checked that
      // it is not a vertex.
      if (PointOnSegment(va, vb, p3).has_value()) {
        IntersectsEdge(a, b, c, p3);
        continue;
      } else if (PointOnSegment(vb, vc, p3).has_value()) {
        IntersectsEdge(b, c, a, p3);
        continue;
      } else if (PointOnSegment(va, vc, p3).has_value()) {
        IntersectsEdge(a, c, b, p3);
        continue;
      }

      // Now, the point is either outside the triangle completely,
      // or properly inside it.
      const vec2 p2 = {p3.x, p3.y};
      if (InTriangle(va2, vb2, vc2, p2)) {
        const int d = AddPoint(p3);
        new_points.insert(d);

        //
        //     a-------------b
        //      \`.       .'/
        //       \ `.  .'  /
        //        \   d   /
        //         \  |  /
        //          \ | /
        //           \|/
        //            c

        CHECK(a != b && a != d && a != c &&
              b != d && b != c &&
              d != c) << "We should have handled this with the "
          "edge and vertex tests above!";
        AddTriangleTo(&new_triangles, a, d, c);
        AddTriangleTo(&new_triangles, d, b, c);
        AddTriangleTo(&new_triangles, a, b, d);
        // And discard the existing triangle.
        continue;
      }

      // Otherwise, the common case that this point is just
      // not in the triangle at all. Preserve the triangle
      // as-is.
      new_triangles.push_back(tri);
    }

    work_triangles = std::move(new_triangles);
    std::vector<int> np(new_points.begin(), new_points.end());
    SortPointIndicesByZ(&np);
    return np;
  }

  // XXX deprecated?
  void Step2() {
    // Next, create a new hole polygon whose vertices are represented by
    // indices in the points. To do this, we'll add points when a
    // vertex is inside a triangle, or when it intersects a triangle
    // edge, or use an existing point when it lands on one. At the same
    // time, we'll split triangles accordingly.

    hole.reserve(input_polygon.size());
    for (const vec2 &v : input_polygon) {
      std::vector<int> ps = ProjectThroughMesh(v);
      CHECK(ps.size() == 2) << "Expecting exactly one intersection on "
        "the top and one on the bottom. Maybe this is not a proper "
        "hole, or maybe it is not in general position. (" << ps.size() <<
        ")";
      hole.emplace_back(ps[0], ps[1]);
    }

    #if 0
    printf("After splitting on points, work triangles:\n");
    for (const auto &[a, b, c] : work_triangles) {
      auto P = [&](int i) {
          printf("    %d = %s\n",
                 i, VecString(points[i]).c_str());
        };
      printf("  ----\n");
      P(a);
      P(b);
      P(c);
    }
    printf("  ----\n");
    #endif
  }

  void SaveMesh(std::string_view filename) {
    Mesh3D tmp;
    tmp.vertices = points;
    tmp.triangles = out_triangles;
    for (const auto &tri : work_triangles)
      tmp.triangles.push_back(tri);
    SaveAsSTL(tmp, filename, "makehole");
  }

  // On the edge from p to q, get the closest intersection
  // with an edge (or vertex). The point must be strictly
  // closer to q than p.
  std::optional<vec2> GetClosestIntersection(const vec2 &p,
                                             const vec2 &q) {
    const double dist_p_to_q = distance(p, q);

    // The closest point matching the criteria.
    std::optional<vec2> closest;
    double closest_dist = std::numeric_limits<double>::infinity();

    auto TryPoint = [&](const vec2 &v) {
        const double qdist = distance(v, q);
        // Must be strictly closer to q.
        if (qdist < dist_p_to_q) {
          double dist = length(v - p);
          if (!closest.has_value() || dist < closest_dist) {
            closest = {v};
          }
        }
      };

    auto TryEdge = [&](int u, int v) {
        CHECK(u < v);
        const vec2 &uv = {points[u].x, points[u].y};
        const vec2 &vv = {points[v].x, points[v].y};

        if (auto lo = LineIntersection(uv, vv, p, q)) {
          TryPoint(lo.value());
        }
      };

    // PERF: This could of course be faster with a spatial data
    // structure!
    for (const auto &[a, b, c] : work_triangles) {
      CHECK(a < b && b < c) << std::format("{} {} {}", a, b, c);
      TryPoint(Two(points[a]));
      TryPoint(Two(points[b]));
      TryPoint(Two(points[c]));
      TryEdge(a, b);
      TryEdge(b, c);
      TryEdge(a, c);
    }

    return closest;
  }

  void Split() {
    // Walk the polygon (in 2D) and project to vertices wherever
    // it has a vertex, or where there is an intersection with
    // an existing triangle. We expect two vertices each time:
    // One for the top and one for the bottom.

    CHECK(hole.empty());
    hole.reserve(input_polygon.size());

    // Project the point through the polyhedron (splitting it as
    // necessary), expecting two intersections.
    auto Sample = [this](const vec2 &v2) -> std::pair<int, int> {
        std::vector<int> ps = ProjectThroughMesh(v2);
        CHECK(ps.size() == 2) << "We expect every projected point to "
          "have both a top and bottom intersection, but got: " << ps.size();
        return {ps[0], ps[1]};
      };

    for (int idx = 0; idx < input_polygon.size(); idx++) {
      vec2 p = input_polygon[idx];
      const vec2 &q = input_polygon[(idx + 1) % input_polygon.size()];

      // Repeatedly find intersections between p and q.

      std::pair<int, int> pp = Sample(p);
      std::pair<int, int> qq = Sample(q);

      hole.push_back(pp);

      while (pp != qq) {
        auto io = GetClosestIntersection(p, q);
        if (!io.has_value()) {
          // No more intersections. Then we are done.
          break;
        }

        const vec2 &i2 = io.value();
        // i2 is a point between p and q.
        // TODO: Could assert this, since we require it for
        // termination.
        CHECK(i2 != p);

        auto rr = Sample(i2);
        CHECK(rr != pp) << "This might ok, but might also result "
          "in degenerate triangles?";

        pp = rr;
        p = i2;
      }

      // qq might already be in the hole.
      CHECK(!hole.empty());
      if (hole.back() != qq)
        hole.push_back(qq);
    }

  }

  void Step3() {
    // Now walk the edges of the hole and make sure that we have
    // vertices at every intersection.
    std::vector<std::pair<int, int>> new_hole;
    for (int idx = 0; idx < hole.size(); idx++) {
      int pt, pb, qt, qb;
      std::tie(pt, pb) = hole[idx];
      std::tie(qt, qb) = hole[(idx + 1) % hole.size()];

      CHECK(pt != qt && pb != qb) << "Fix this above";

      // Starting pair remains in the hole.
      new_hole.emplace_back(pt, pb);

      // Find intersections from p->q.
      while (pt != qt) {
        // The top edge and bottom edge are the same in 2D. So just get
        // them from the top. (Note that we may have snapped differently,
        // but we ignore this issue, perhaps to our peril?)
        const vec2 &pv = {points[pt].x, points[pt].y};
        const vec2 &qv = {points[qt].x, points[pt].y};

        // Get the closest intersection from pv->qv.
        // The first two indices are the edge, and the third is the
        // other vertex in the triangle. Edges are ordered a < b.
        std::optional<std::tuple<vec2, int, int, int>> closest;
        double closest_dist = std::numeric_limits<double>::infinity();
        auto Try = [&](int u, int v, int w) {
            // So that the output edge is ordered.
            CHECK(u < v);
            const vec2 &uv = {points[u].x, points[u].y};
            const vec2 &vv = {points[v].x, points[v].y};

            if (auto lo = LineIntersection(uv, vv, pv, qv)) {
              double dist = length(lo.value() - pv);
              if (!closest.has_value() || dist < closest_dist) {
                closest = {std::make_tuple(lo.value(), u, v, w)};
              }
            }
          };

        for (const auto &[a, b, c] : work_triangles) {
          CHECK(a < b && b < c) << std::format("{} {} {}", a, b, c);
          Try(a, b, c);
          Try(b, c, a);
          Try(a, c, b);
        }

        if (!closest.has_value()) {
          // If there was no intersection, then since p and q
          // are on vertices of triangles, they must already
          // be connected by an edge. So we just advance.
          CHECK(HasEdge(pt, qt));
          CHECK(HasEdge(pb, qb));

          pt = qt;
          pb = qb;
          break;
        }

        // Otherwise, split the triangle.

        const auto &[i2, edge_a, edge_b, c_] = closest.value();
        CHECK(edge_a < edge_b) << edge_a << " " << edge_b;

        // i2 is an intersection on the edge a-b. Find that point in 3D.
        const vec3 &va = points[edge_a];
        const vec3 &vb = points[edge_b];
        const std::optional<vec3> io = PointToLine(va, vb, i2);
        CHECK(io.has_value()) << "We intersected a perpendicular line?";
        const int i = AddPoint(io.value());

        printf("Edge %d-%d:\n"
               "  %s\n"
               "  %s\n", edge_a, edge_b,
               VecString(va).c_str(), VecString(vb).c_str());
        printf("Intersection point #%d, at %s\n",
               i, VecString(io.value()).c_str());

        // We expect to intersect a face on the other side as well.
        std::vector<int> new_points = {i};

        // Now add it to every triangle with an edge (a, b).
        {
          std::vector<std::tuple<int, int, int>> new_triangles;
          new_triangles.reserve(work_triangles.size());

          for (const auto &tri : work_triangles) {
            printf("Consider %d-%d-%d\n",
                   std::get<0>(tri),
                   std::get<1>(tri),
                   std::get<2>(tri));
            if (auto co = TriangleWithEdge(tri, edge_a, edge_b)) {
              const int c = co.value();

              // e.g.
              // a---i---b  //
              //  \  |  /   //
              //   \ | /    //
              //    \|/     //
              //     c      //
              CHECK(edge_a != i && edge_a != c && i != c);
              CHECK(edge_b != i && edge_b != c && i != c);
              AddTriangleTo(&new_triangles, edge_a, i, c);
              AddTriangleTo(&new_triangles, i, edge_b, c);
            } else if (IsAVertex(tri, i)) {

              // If the point snapped to a vertex of this triangle,
              // it is not inside it. No need to split.
              new_triangles.push_back(tri);

            } else {
              const auto &[a, b, c] = tri;
              const vec3 &va = points[a];
              const vec3 &vb = points[b];
              const vec3 &vc = points[c];
              const vec2 a2 = {va.x, va.y};
              const vec2 b2 = {vb.x, vb.y};
              const vec2 c2 = {vc.x, vc.y};

              if (InTriangle(a2, b2, c2, i2)) {

                printf("Split triangle %d-%d-%d which has vertices:\n"
                       "  %s\n"
                       "  %s\n"
                       "  %s\n"
                       "At point: %s\n",
                       a, b, c,
                       VecString(va).c_str(),
                       VecString(vb).c_str(),
                       VecString(vc).c_str(),
                       VecString(i2).c_str());

                // Split like in SplitTrianglesAtPoint.
                auto i3o = PointToPlane(va, vb, vc, i2);
                CHECK(i3o.has_value()) << "If the point is strictly "
                  "within the triangle, we should be able to project "
                  "it to 3D. But this could happen due to a "
                  "disagreement about 'epsilon'.";

                const int i = AddPoint(i3o.value());
                new_points.push_back(i);

                CHECK(a != i && i != c && a != c) <<
                  std::format("{} {} {}", a, i, c);
                AddTriangleTo(&new_triangles, a, i, c);
                CHECK(i != b && b != c && i != c);
                AddTriangleTo(&new_triangles, i, b, c);
                CHECK(a != b && b != i && a != i);
                AddTriangleTo(&new_triangles, a, b, i);

              } else {
                // Keep as-is.
                new_triangles.push_back(tri);
              }
            }
          }

          work_triangles = std::move(new_triangles);

          SortPointIndicesByZ(&new_points);
          CHECK(new_points.size() == 2) << "We should always enter "
            "in the top and exit on the bottom, creating two points.";

          pt = new_points[0];
          pb = new_points[1];
          new_hole.emplace_back(pt, pb);
        }
      }

      CHECK(pb == qb) << "This should be the case when the top segment "
        "becomes degenerate, but I guess it's possible that one snaps "
        "when the other doesn't.";

    }
  }

  // Remove triangles that have a vertex inside the hole.
  void RemoveHole() {
    std::vector<std::tuple<int, int, int>> new_triangles;
    new_triangles.reserve(work_triangles.size());

    std::unordered_set<int> hole_vertices;
    for (const auto &[t, b] : hole) {
      hole_vertices.insert(t);
      hole_vertices.insert(b);
    }

    int dropped = 0;
    for (const auto &tri : work_triangles) {
      const auto &[a, b, c] = tri;

      // If we did the previous splitting correctly, then
      // a triangle with any vertex strictly inside the
      // hole should be removed.
      bool inside = false;
      for (int v : {a, b, c}) {
        if (!hole_vertices.contains(v) &&
            PointInPolygon(Two(points[v]), input_polygon)) {
          inside = true;
          break;
        }
      }

      // Only keep it if it's on the outside.
      if (!inside) {
        new_triangles.push_back(tri);
      } else {
        dropped++;
      }
    }

    work_triangles = std::move(new_triangles);
    printf("Removed %d triangles in hole.\n", dropped);
  }

  void RepairHole() {

    // TODO: Use the hole to create internal faces. Delete triangles
    // that have a vertex inside the hole.

  }

  Mesh3D GetMesh() {
    // TODO: Build mesh3d and return.

    LOG(FATAL) << "Unimplemented";
    return Mesh3D{};
  }
};
}  // namespace

Mesh3D MakeHole(const Polyhedron &polyhedron,
                const std::vector<vec2> &polygon) {
  HoleMaker maker(polyhedron, polygon);
  maker.Split();
  maker.SaveMesh("makehole.stl");
  maker.RemoveHole();
  maker.SaveMesh("removehole.stl");

  return maker.GetMesh();
}
