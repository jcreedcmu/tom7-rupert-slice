
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
#include "bounds.h"
#include "image.h"

#include "yocto_matht.h"

namespace {
// TODO: Use this throughout.
static inline vec2 Two(const vec3 &v) {
  return vec2{v.x, v.y};
}

// Draw triangles where all the vertices have z < 0.
static void DrawTop(const Mesh3D &mesh, std::string_view filename) {
  auto Filter = [&mesh](int i) {
      return mesh.vertices[i].z < 0.0;
    };

  Bounds bounds;
  for (int i = 0; i < mesh.vertices.size(); i++) {
    if (Filter(i)) {
      const vec3 &v = mesh.vertices[i];
      bounds.Bound(v.x, v.y);
    }
  }

  bounds.AddMarginFrac(0.10);

  // ImageRGBA img(1920, 1080);
  ImageRGBA img(1024, 768);
  img.Clear32(0x000000FF);

  Bounds::Scaler scaler =
    bounds.ScaleToFit(img.Width(), img.Height()).FlipY();

  auto ToScreen = [&](int i) {
      CHECK(i >= 0 && i < mesh.vertices.size());
      const vec3 &v = mesh.vertices[i];
      return std::make_pair((int)scaler.ScaleX(v.x),
                            (int)scaler.ScaleY(v.y));
    };

  // First draw the triangles.
  for (const auto &[a, b, c] : mesh.triangles) {
    if (Filter(a) && Filter(b) && Filter(c)) {
      const auto &[ax, ay] = ToScreen(a);
      const auto &[bx, by] = ToScreen(b);
      const auto &[cx, cy] = ToScreen(c);
      img.BlendThickLine32(ax, ay, bx, by, 3.0, 0xFFFFFF77);
      img.BlendThickLine32(bx, by, cx, cy, 3.0, 0xFFFFFF77);
      img.BlendThickLine32(cx, cy, ax, ay, 3.0, 0xFFFFFF77);
    }
  }

  // Now draw vertices.
  for (int i = 0; i < mesh.vertices.size(); i++) {
    if (Filter(i)) {
      const auto &[x, y] = ToScreen(i);
      img.BlendThickCircle32(x, y, 6.0f, 3.0f, 0xFF888877);
    }
  }

  // And labels.
  for (int i = 0; i < mesh.vertices.size(); i++) {
    if (Filter(i)) {
      const auto &[x, y] = ToScreen(i);
      img.BlendTextOutline32(x + 8, y + 8,
                             0x00000044,
                             0x88FF88FF,
                             std::format("{}", i));
    }
  }

  img.Save(filename);
  printf("Saved %s\n", std::string(filename).c_str());
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
    const bool VERBOSE = false;
    if (VERBOSE) {
      printf(ACYAN("proj") " at %s\n", VecString(pt).c_str());
    }
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

      if (VERBOSE) {
        printf("For triangle\n"
               "  %d. %s\n"
               "  %d. %s\n"
               "  %d. %s\n",
               a, VecString(va).c_str(),
               b, VecString(vb).c_str(),
               c, VecString(vc).c_str());
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
        if (VERBOSE) {
          printf("  " ABLUE("already split") " %d-%d (other %d). add %d\n",
                 a, b, c, d);
        }
        continue;
      }

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
        if (VERBOSE) {
          printf("  " AORANGE("perp") "\n");
        }
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
        if (VERBOSE) {
          printf("  " AGREEN("already have vertex") " %d\n",
                 p.value());
        }
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
          if (VERBOSE) {
            printf("  " APURPLE("new split") " %d-%d (other %d). add %d\n",
                   a, b, c, d);
          }
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
        if (VERBOSE) {
          printf("  " ARED("split inside") " %d-%d-%d +%d\n",
                 a, b, c, d);
        }
        continue;
      }

      // Otherwise, the common case that this point is just
      // not in the triangle at all. Preserve the triangle
      // as-is.
      if (VERBOSE) {
        printf("  " AGREY("nothing") "\n");
      }
      new_triangles.push_back(tri);
    }

    work_triangles = std::move(new_triangles);
    std::vector<int> np(new_points.begin(), new_points.end());
    SortPointIndicesByZ(&np);
    return np;
  }

  void SaveMesh(std::string_view filename) {
    Mesh3D tmp;
    tmp.vertices = points;
    tmp.triangles = out_triangles;
    for (const auto &tri : work_triangles)
      tmp.triangles.push_back(tri);
    SaveAsSTL(tmp, std::format("{}.stl", filename), "makehole");

    DrawTop(tmp, std::format("{}.png", filename));
  }

  // On the edge from p to q, get the closest intersection
  // with an edge (or vertex). The point must be strictly
  // closer to q than p.
  std::optional<vec2> GetClosestIntersection(const vec2 &p,
                                             const vec2 &q,
                                             bool verbose = false) {
    printf("From %s -> %s\n", VecString(p).c_str(),
           VecString(q).c_str());
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
            if (verbose) {
              printf("    " AYELLOW("(new best)") "\n");
            }
            closest = {v};
            closest_dist = dist;
          }
        }
      };

    auto TryEdge = [&](int u, int v) {
        CHECK(u < v);
        const vec2 &uv = {points[u].x, points[u].y};
        const vec2 &vv = {points[v].x, points[v].y};
        if (verbose) {
          printf("  Try edge %d-%d:\n", u, v);
        }
        if (auto lo = LineIntersection(uv, vv, p, q)) {
          if (verbose) {
            printf("   Intersection at %s\n",
                   VecString(lo.value()).c_str());
          }
          TryPoint(lo.value());
        }
      };

    // PERF: This could of course be faster with a spatial data
    // structure!
    for (const auto &[a, b, c] : work_triangles) {
      CHECK(a < b && b < c) << std::format("{} {} {}", a, b, c);

      // We might want to try very close points. But it doesn't
      // make sense to check points that aren't actually on the
      // way there!
      // TryPoint(Two(points[a]));
      // TryPoint(Two(points[b]));
      // TryPoint(Two(points[c]));
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

    int filename_index = 0;
    // Project the point through the polyhedron (splitting it as
    // necessary), expecting two intersections.
    auto Sample = [this](const vec2 &v2) -> std::pair<int, int> {
        std::vector<int> ps = ProjectThroughMesh(v2);
        CHECK(ps.size() == 2) << "We expect every projected point to "
          "have both a top and bottom intersection, but got: " << ps.size();
        return {ps[0], ps[1]};
      };

    for (int idx = 0; idx < input_polygon.size(); idx++) {
      printf(ABGCOLOR(0, 255, 255,
                      ADARKGREY("    --- in poly %d ---    ")) "\n",
             idx);
      vec2 p = input_polygon[idx];
      const vec2 &q = input_polygon[(idx + 1) % input_polygon.size()];

      // Repeatedly find intersections between p and q.

      std::pair<int, int> pp = Sample(p);
      std::pair<int, int> qq = Sample(q);
      printf("Top cut: %d->%d\n", pp.first, qq.first);

      hole.push_back(pp);

      while (pp != qq) {
        SaveMesh(std::format("split{}", filename_index));
        filename_index++;

        // Split0 lgtm.

        // Split1 looks probably good now.

        printf("Top vertex " AWHITE("%d") " to " AWHITE("%d") "\n",
               pp.first, qq.first);
        auto io = GetClosestIntersection(p, q, pp.first == 14);
        if (!io.has_value()) {
          printf("No more intersections.\n");
          // No more intersections. Then we are done.
          // MaybeAddEdge(pp.first, qq.first);
          // MaybeAddEdge(pp.second, qq.second);
          break;
        }

        const vec2 &i2 = io.value();
        // i2 is a point between p and q.
        // TODO: Could assert this, since we require it for
        // termination.
        CHECK(i2 != p);

        // It could snap to the same point, though.
        auto rr = Sample(i2);
        printf("Intersection at %s. top = %d\n", VecString(i2).c_str(),
               rr.first);
        if (rr != pp) {
          hole.push_back(rr);
          pp = rr;
        }

        p = i2;
      }

      // qq might already be in the hole.
      CHECK(!hole.empty());
      if (hole.back() != qq)
        hole.push_back(qq);
    }
  }

  void CheckEdgeLoop(const std::vector<int> &loop) {
    static constexpr bool VERBOSE = false;
    if (VERBOSE) {
      printf("Loop:\n");
    }
    int missing = 0;
    for (int i = 0; i < loop.size(); i++) {
      const int a = loop[i];
      const int b = loop[(i + 1) % loop.size()];
      if (VERBOSE) {
        printf("  %d. #%d -> #%d   %s %s\n",
               i, a, b,
               VecString(points[a]).c_str(),
               VecString(points[b]).c_str());
      }
      if (a == b) {
        if (VERBOSE) {
          printf("    (skip)\n");
        }
        continue;
      }
      if (!HasEdge(a, b)) {
        if (VERBOSE) {
          printf("    (missing)\n");
        }
        missing++;
      }
    }
    if (VERBOSE) {
      printf("Missing %d edges.\n", missing);
    }
    CHECK(missing == 0);
  }

  // Now we have a vertex at every point and intersection on the
  // top and bottom holes. This should also result in an edge
  // between adjacent vertices.
  void CheckEdgeLoops() {
    std::vector<int> top, bot;
    for (const auto &[t, b] : hole) {
      if (top.empty() || t != top.back()) top.push_back(t);
      if (bot.empty() || b != bot.back()) bot.push_back(b);
    }

    CheckEdgeLoop(top);
    CheckEdgeLoop(bot);
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
      // If a triangle has all its vertices on its hole,
      // then it should also be removed (it is part of a
      // face that spans the entire hole).
      bool on = true;
      for (int v : {a, b, c}) {
        bool on_hole = hole_vertices.contains(v);
        if (!on_hole) on = false;
        if (!on_hole &&
            PointInPolygon(Two(points[v]), input_polygon)) {
          inside = true;
        }
      }

      // Only keep it if it's on the outside.
      if (on || inside) {
        dropped++;
      } else {
        new_triangles.push_back(tri);
      }
    }

    work_triangles = std::move(new_triangles);
    printf("Removed %d triangles in hole.\n", dropped);
  }

  void RepairHole() {

    // Create internal holes as a triangle strip.
    for (int idx = 0; idx < hole.size(); idx++) {
      const auto &[pt, pb] = hole[idx];
      const auto &[qt, qb] = hole[(idx + 1) % hole.size()];
      if (pt == qt && pb == qb) {
        // Nothing to do if the two edges are the same.
      } else if (pt == qt) {
        // Top points are equal, so we just need one
        // triangle.
        AddTriangleTo(&work_triangles, pt, pb, qb);
      } else if (pb == qb) {
        // Same, on the bottom.
        AddTriangleTo(&work_triangles, pb, pt, qt);
      } else {
        // A quad, made of two triangles.

        //   pt------qt
        //   |`.      |
        //   |  `.    |
        //   |    `.  |
        //   pb------qb

        AddTriangleTo(&work_triangles, pt, qt, qb);
        AddTriangleTo(&work_triangles, pt, qb, pb);
      }
    }
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
  maker.SaveMesh("split");
  maker.CheckEdgeLoops();
  maker.RemoveHole();
  maker.SaveMesh("removehole");
  maker.RepairHole();
  maker.SaveMesh("repairhole");

  return maker.GetMesh();
}
