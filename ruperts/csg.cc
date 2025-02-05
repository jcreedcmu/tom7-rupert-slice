
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

// Project the point pt, which should like on the 2D segment (a, b)
// in the xy plane, such that it lies on the 3D segment (a, b).
static std::optional<vec3> PointToLine(const vec3 &a, const vec3 &b,
                                       const vec2 &pt2) {
  vec2 a2 = {a.x, a.y};
  vec2 b2 = {b.x, b.y};

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

Mesh3D MakeHole(const Polyhedron &polyhedron,
                const std::vector<vec2> &polygon) {
  // Indexed set of points.
  std::vector<vec3> points;
  // Point map so that we can get the index of a vertex
  // that's already been added. This also merges points
  // that are less than some epsilon from the first
  // one inserted in the vicinity.
  PointMap3<int> point_index;
  std::vector<std::tuple<int, int, int>> out_triangles;
  auto AddPoint = [&](const vec3 &pt) {
      if (auto io = point_index.Get(pt)) {
        return io.value();
      }

      int idx = points.size();
      points.push_back(pt);
      point_index.Add(pt, idx);
      return idx;
    };

  for (int idx = 0; idx < polyhedron.vertices.size(); idx++) {
    const vec3 &v = polyhedron.vertices[idx];
    // We would just need to remap the triangles.
    CHECK(AddPoint(v) == idx) << "Points from the input polyhedron "
      "were merged. This can be handled, but isn't handled yet.";
  }

  // Add a triangle as indices of its vertices. We insert them in
  // sorted order so that it's easier to find edges and so on.
  auto AddTriangleTo = [&](std::vector<std::tuple<int, int, int>> *out,
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
    };

  // Triangles that intersect or abut the hole polygon. These
  // vertices are indexes into points.
  std::vector<std::tuple<int, int, int>> work_triangles;

  auto HasEdge = [&](int u, int v) {
      if (u == v) return false;
      // Triangles are stored with a < b < c.
      if (u > v) std::swap(u, v);
      for (const auto &[a, b, c] : work_triangles) {
        if (a == u && (b == v || c == v)) return true;
        if (b == u && c == v) return true;
      }
      return false;
    };

  auto SortPointIndicesByZ = [&points](std::vector<int> *indices) {
      std::sort(indices->begin(), indices->end(),
                [&points](int a, int b) {
                  CHECK(a >= 0 && b >= 0 &&
                        a < points.size() &&
                        b < points.size()) <<
                    std::format("a: {}, b: {}, points.size: {}",
                                a, b, points.size());
                  return points[a].z < points[b].z;
                });
    };

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

  // Next, create a new hole polygon whose vertices are represented by
  // indices in the points. To do this, we'll add points when a
  // vertex is inside a triangle, or when it intersects a triangle
  // edge. At the same time, we'll split triangles accordingly.
  //
  // Returns all the intersection points (as integer indices into points),
  // sorted from lowest z coordinate to highest.
  auto SplitTrianglesAtPoint = [&](const vec2 &pt) -> std::vector<int> {
      std::vector<std::tuple<int, int, int>> new_triangles;
      new_triangles.reserve(work_triangles.size());

      std::vector<int> new_points;

      for (const auto &[a, b, c] : work_triangles) {
        const vec3 &va = points[a];
        const vec3 &vb = points[b];
        const vec3 &vc = points[c];

        const vec2 &va2 = {va.x, va.y};
        const vec2 &vb2 = {vb.x, vb.y};
        const vec2 &vc2 = {vc.x, vc.y};

        // There should only be two triangles that the point lands
        // within (top and bottom). We might want to check that. This
        // also might be the right place to add an edge, or at least
        // record the correspondence.
        if (InTriangle(va2, vb2, vc2, pt)) {

          auto p3o = PointToPlane(va, vb, vc, pt);
          CHECK(p3o.has_value()) << "If the point is strictly within "
            "the triangle, we should be able to project it to 3D. "
            "But this could happen due to a disagreement about 'epsilon'.";

          const int p = AddPoint(p3o.value());
          new_points.push_back(p);

          //
          //     a-------------b
          //      \`.       .'/
          //       \ `.  .'  /
          //        \   p   /
          //         \  |  /
          //          \ | /
          //           \|/
          //            c

          CHECK(a != b && a != p && a != c &&
                b != p && b != c &&
                p != c);
          AddTriangleTo(&new_triangles, a, p, c);
          AddTriangleTo(&new_triangles, p, b, c);
          AddTriangleTo(&new_triangles, a, b, p);
        } else {
          // Not split.
          CHECK(a != b && b != c && a != c);
          AddTriangleTo(&new_triangles, a, b, c);
        }
      }

      work_triangles = std::move(new_triangles);
      SortPointIndicesByZ(&new_points);
      return new_points;
    };

  // First, no need to deal with general points for this. Split
  // triangles so that all the polygon points land on triangle
  // vertices. We get a hole as a series of point indices. There
  // is a top and bottom of the hole; this vector keeps them
  // in correspondence.
  std::vector<std::pair<int, int>> hole;
  hole.reserve(polygon.size());
  for (const vec2 &v : polygon) {
    // XXX should check that it's not the same as the previous
    // point
    std::vector<int> ps = SplitTrianglesAtPoint(v);
    CHECK(ps.size() == 2) << "Expecting exactly one intersection on "
      "the top and one on the bottom. Maybe this is not a proper "
      "hole, or maybe it is not in general position. (" << ps.size() <<
      ")";
    hole.emplace_back(ps[0], ps[1]);
  }

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

  {
    Mesh3D tmp;
    tmp.vertices = points;
    tmp.triangles = out_triangles;
    for (const auto &tri : work_triangles)
      tmp.triangles.push_back(tri);
    SaveAsSTL(tmp, "makehole.stl", "makehole");
  }


  // But edges of the polygon might now intersect edges of
  // triangles. Process each edge in the hole until this
  // is no longer the case.
  std::vector<std::pair<int, int>> new_hole;
  for (int idx = 0; idx < hole.size(); idx++) {
    int pt, pb, qt, qb;
    std::tie(pt, pb) = hole[idx];
    std::tie(qt, qb) = hole[(idx + 1) % hole.size()];

    CHECK(pt != qt && pb != qb) << "Fix this above";

    // We know both p and q are not strictly inside any triangles, and
    // moreover, each is a vertex of at least one triangle.
    new_hole.emplace_back(pt, pb);

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
              CHECK(i3o.has_value()) << "If the point is strictly within "
                "the triangle, we should be able to project it to 3D. "
                "But this could happen due to a disagreement about 'epsilon'.";

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

    // A polygon edge could intersect multiple triangles, or none!
    // For the loop, we want to be in a position where the point
    // is on a triangle edge.
  }

  // TODO: Use the hole to create internal faces. Delete triangles
  // that have a vertex inside the hole.

  // TODO: Build mesh3d and return.
}


