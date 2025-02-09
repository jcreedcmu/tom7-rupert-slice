
#include "csg.h"

#include <set>
#include <format>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "bignum/big.h"
#include "bignum/big-overloads.h"
#include "big-polyhedra.h"
#include "ansi.h"
#include "base/logging.h"
#include "base/stringprintf.h"
#include "hashing.h"
#include "bounds.h"
#include "image.h"
#include "dirty.h"
#include "mesh.h"
#include "rendering.h"
#include "polyhedra.h"

static constexpr bool DEBUG = false;
static constexpr bool VERBOSE = false;

namespace {
// TODO: Use this throughout.
static inline BigVec2 Two(const BigVec3 &v) {
  return BigVec2{v.x, v.y};
}

inline bool CrossIsZero(const BigVec3 &a, const BigVec3 &b) {
  // PERF can just do the comparisons
  return
    BigRat::Sign(a.y * b.z - a.z * b.y) == 0 &&
    BigRat::Sign(a.z * b.x - a.x * b.z) == 0 &&
    BigRat::Sign(a.x * b.y - a.y * b.x) == 0;
}


// Draw triangles where all the vertices have z < 0.
static void DrawTop(const Mesh3D &mesh, std::string_view filename) {
  auto Filter = [&mesh](int i) {
      (void)mesh;
      // return mesh.vertices[i].z >= 0.0;
      return true;
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
  ImageRGBA img(1024, 1024);
  // ImageRGBA img(5120, 5120);
  static constexpr float LINE_WIDTH = 1.0f;
  img.Clear32(0x000000FF);

  Dirty dirty(img.Width(), img.Height());

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
      img.BlendThickLine32(ax, ay, bx, by, LINE_WIDTH, 0xFFFFFF77);
      img.BlendThickLine32(bx, by, cx, cy, LINE_WIDTH, 0xFFFFFF77);
      img.BlendThickLine32(cx, cy, ax, ay, LINE_WIDTH, 0xFFFFFF77);
    }
  }

  // Now draw vertices.
  for (int i = 0; i < mesh.vertices.size(); i++) {
    if (Filter(i)) {
      const auto &[x, y] = ToScreen(i);
      float r = 6.0f;
      int pad = r + 3;
      img.BlendThickCircle32(x, y, 6.0f, 3.0f,
                             Rendering::Color(i) & 0xFFFFFF77);
      dirty.MarkUsed(x - pad, y - pad, pad * 2, pad * 2);
    }
  }

  // And labels.
  for (int i = 0; i < mesh.vertices.size(); i++) {
    if (Filter(i)) {
      const auto &[ox, oy] = ToScreen(i);
      int w = ImageRGBA::TEXT_WIDTH * 2 + 4;
      int h = ImageRGBA::TEXT_HEIGHT + 4;
      const auto &[x, y] =
        dirty.PlaceNearby(ox, oy, w, h, 32);
      dirty.MarkUsed(x, y, w, h);

      img.BlendTextOutline32(x, y,
                             0x00000044,
                             // 0x88FF88FF,
                             Rendering::Color(i),
                             std::format("{}", i));
    }
  }

  img.Save(filename);
  printf("Saved %s\n", std::string(filename).c_str());
}

// Project the point pt along z to the triangle (plane)
// defined by (a, b, c). The returned point has the same x,y position
// as pt, but a z coordinate that places it on that plane.
//
// This returns a result even if the point is not in the triangle.
// It returns nullopt when the plane is perpendicular to xy.
static std::optional<BigVec3> PointToPlane(
    const BigVec3 &a, const BigVec3 &b, const BigVec3 &c,
    const BigVec2 &pt) {
  BigVec2 a2 = Two(a);
  BigVec2 b2 = Two(b);
  BigVec2 c2 = Two(c);

  // Barycentric coordinates in 2D.
  BigVec2 v0 = b2 - a2;
  BigVec2 v1 = c2 - a2;
  BigVec2 v2 = pt - a2;

  BigRat d00 = dot(v0, v0);
  BigRat d01 = dot(v0, v1);
  BigRat d11 = dot(v1, v1);
  BigRat d20 = dot(v2, v0);
  BigRat d21 = dot(v2, v1);

  BigRat denom = d00 * d11 - d01 * d01;

  // If denom is zero (or close) then the triangle is degenerate in 2D.
  if (denom == BigRat(0)) return std::nullopt;

  BigRat v = (d11 * d20 - d01 * d21) / denom;
  BigRat w = (d00 * d21 - d01 * d20) / denom;
  BigRat u = BigRat(1) - v - w;

  // The weights u, v, w can also be used to interpolate in 3D, since
  // these are linear interpolations and the projection from 3D to 2D
  // is also linear.
  BigRat z = u * a.z + v * b.z + w * c.z;

  return {BigVec3{pt.x, pt.y, z}};
}

static std::optional<BigVec3> PointOnSegment(
    // First segment
    const BigVec3 &a, const BigVec3 &b,
    // Test point
    const BigVec3 &p) {
  BigVec3 ab = b - a;
  BigVec3 ap = p - a;

  BigVec3 cx = cross(ab, ap);
  if (length_squared(cx) != BigRat(0)) {
    return std::nullopt;
  }

  // parameter interpolating from a to b.
  BigRat t = dot(ap, ab) / length_squared(ab);
  if (t < BigRat(0) || t > BigRat(1)) return std::nullopt;

  BigVec3 isect = a + t * ab;
  return {isect};
}

static std::optional<BigVec2> LineIntersection(
    // First segment
    const BigVec2 &p0, const BigVec2 &p1,
    // Second segment
    const BigVec2 &p2, const BigVec2 &p3) {

  const BigVec2 s1 = p1 - p0;
  const BigVec2 s2 = p3 - p2;
  const BigVec2 m = p0 - p2;

  const BigRat denom = s1.x * s2.y - s2.x * s1.y;
  if (denom == BigRat(0)) return std::nullopt;

  const BigRat s = (s1.x * m.y - s1.y * m.x) / denom;

  if (s >= BigRat(0) && s <= BigRat(1)) {
    const BigRat t = (s2.x * m.y - s2.y * m.x) / denom;

    if (t >= BigRat(0) && t <= BigRat(1)) {
      return {BigVec2{p0 + t * s1}};
    }
  }
  return std::nullopt;
}

bool TriangleAndPolygonIntersect(
    const BigVec2 &a, const BigVec2 &b, const BigVec2&c,
    const std::vector<BigVec2> &polygon) {
  for (int i = 0; i < polygon.size(); i++) {
    const BigVec2 &v0 = polygon[i];
    const BigVec2 &v1 = polygon[(i + 1) % polygon.size()];

    if (LineIntersection(a, b, v0, v1).has_value() ||
        LineIntersection(b, c, v0, v1).has_value() ||
        LineIntersection(c, a, v0, v1).has_value()) {
      return true;
    }
  }
  return false;
}

// Analogous to PointMap3, but with exact tests.
template<class Value>
struct BigPointMap3 {
  // Note that this does *not* deduplicate points! It's just
  // the number of elements that have been inserted.
  size_t Size() const {
    return pts.size();
  }

  bool Contains(const BigVec3 &p) const {
    return Get(p).has_value();
  }

  std::optional<Value> Get(const BigVec3 &p) const {
    for (const auto &[q, v] : pts) {
      if (p == q) {
        return {v};
      }
    }

    return std::nullopt;
  }

  void Add(const BigVec3 &q, const Value &v) {
    pts.emplace_back(q, v);
  }

  std::vector<BigVec3> Points() const {
    std::vector<BigVec3> ret;
    ret.reserve(pts.size());
    for (const auto &[p, v] : pts) ret.push_back(p);
    return ret;
  }

 private:
  // perf use kd-tree, or even hash map?
  std::vector<std::pair<BigVec3, Value>> pts;
};

struct BigHoleMaker {
  // Indexed set of points.
  std::vector<BigVec3> points;
  // Point map so that we can get the index of a vertex
  // that's already been added. Exact.
  BigPointMap3<int> point_index;
  std::vector<std::tuple<int, int, int>> out_triangles;

  // We create a path consisting of a series of vertices to
  // describe the polygonal hole on each side. But if we
  // later split an edge, we need to know this so that we
  // can go through the inserted split point.
  // In an edge split a-c-b, maps (a, b) to c, with a<b.
  // Note that a-c and c-b may be further split.
  std::unordered_map<std::pair<int, int>, int,
    Hashing<std::pair<int, int>>> split_edges;

  std::optional<int> GetSplitEdge(int a, int b) {
    if (b < a) std::swap(a, b);
    auto it = split_edges.find({a, b});
    if (it == split_edges.end()) return std::nullopt;
    CHECK(it->second != a && it->second != b);
    return {it->second};
  }

  // Return the point's index if we already have it (or a point
  // very close to it).
  std::optional<int> GetPoint(const BigVec3 &pt) {
    return point_index.Get(pt);
  }

  int AddPoint(const BigVec3 &pt) {
    if (auto io = GetPoint(pt)) {
      return io.value();
    }

    int idx = points.size();
    points.push_back(pt);
    point_index.Add(pt, idx);
    if (VERBOSE) {
      printf("Added new point " AYELLOW("%d")
             " at %s.\n", idx, VecString(pt).c_str());
    }
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

  const std::vector<BigVec2> input_polygon;
  // Hole, as point indices (top, bottom).
  std::vector<std::pair<int, int>> hole;

  BigHoleMaker(const Polyhedron &polyhedron,
               const std::vector<BigVec2> &polygon) :
    input_polygon(polygon) {

    std::vector<BigVec3> bigverts;
    bigverts.reserve(polyhedron.vertices.size());
    for (int idx = 0; idx < polyhedron.vertices.size(); idx++) {
      const vec3 &v = polyhedron.vertices[idx];
      bigverts.emplace_back(
          BigRat::FromDouble(v.x),
          BigRat::FromDouble(v.y),
          BigRat::FromDouble(v.z));
    }

    for (int idx = 0; idx < polyhedron.vertices.size(); idx++) {
      const BigVec3 &bv = bigverts[idx];
      // We would just need to remap the triangles.
      CHECK(AddPoint(bv) == idx) << "Points from the input polyhedron "
        "were merged. This can be handled, but isn't handled yet.";
    }

    // First, classify each triangle as entirely inside, entirely outside,
    // or intersecting. The intersecting triangles will be handled in
    // the next loop.
    for (const auto &[a, b, c] : polyhedron.faces->triangulation) {
      BigVec2 v0 = Two(bigverts[a]);
      BigVec2 v1 = Two(bigverts[b]);
      BigVec2 v2 = Two(bigverts[c]);

      int count = 0;
      for (const BigVec2 &v : {v0, v1, v2}) {
        count += PointInPolygon(v, polygon) ? 1 : 0;
      }

      if (count == 3) {
        // If all are inside, then it cannot intersect the hole (convexity).
        // We just discard these.
        if (VERBOSE) {
          printf("[%d] Triangle %d-%d-%d is inside the hole.\n",
                 count, a, b, c);
        }

        // TODO: Not correct to ignore triangles outside, since we might
        // still need to split them. If the hole has vertices in the
        // triangle it still needs to be considered. We could do this
        // by fixing the behavior of TriangleAndPolygonIntersect.
      } else if (true || TriangleAndPolygonIntersect(v0, v1, v2, polygon)) {
        // Might need to be bisected below.
        CHECK(a != b && b != c && a != c);
        AddTriangleTo(&work_triangles, a, b, c);
        if (VERBOSE) {
          printf("[%d] Triangle %d-%d-%d intersects the hole.\n",
                 count, a, b, c);
        }

      } else {
        // If the triangle is entirely outside, then we persist it untouched.
        CHECK(a != b && b != c && a != c);
        AddTriangleTo(&out_triangles, a, b, c);
        if (VERBOSE) {
          printf("[%d] Triangle %d-%d-%d is entirely outside the hole.\n",
                 count, a, b, c);
        }
      }
    }
  }

  // Project a 2D point through the mesh. It can intersect the interior
  // of a triangle, or an existing vertex, or an existing edge. In each
  // case, alter the mesh as appropriate so that the intersection is an
  // actual vertex. Returns the vector of vertex indices, sorted by
  // their z coordinate.

  std::vector<int> ProjectThroughMesh(const BigVec2 &pt) {
    if (VERBOSE) {
      printf(ACYAN("proj") " at %s (%d triangles)\n",
             VecString(pt).c_str(),
             (int)work_triangles.size());
    }

    // static constexpr bool VERBOSE = false;

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
      const BigVec3 &va = points[a];
      const BigVec3 &vb = points[b];
      const BigVec3 &vc = points[c];

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

      const BigVec2 &va2 = {va.x, va.y};
      const BigVec2 &vb2 = {vb.x, vb.y};
      const BigVec2 &vc2 = {vc.x, vc.y};

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

      const BigVec3 p3 = p3o.value();

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
            int a, int b, int c, const BigVec3 &p3) {
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
          if (VERBOSE) {
            printf("Split %d-%d, inserting %d\n", a, b, d);
          }
          already_split[{a, b}] = d;
          CHECK(a < b);
          split_edges[{a, b}] = d;
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
      const BigVec2 p2 = {p3.x, p3.y};
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
          printf("  " AYELLOW("split inside") " %d-%d-%d +%d\n",
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
    for (const BigVec3 &v : points) {
      tmp.vertices.push_back(SmallVec(v));
    }

    tmp.triangles = out_triangles;
    for (const auto &tri : work_triangles)
      tmp.triangles.push_back(tri);
    SaveAsSTL(tmp, std::format("{}.stl", filename), "makehole");

    DrawTop(tmp, std::format("{}.png", filename));
  }

  // On the edge from p to q, get the closest intersection
  // with an edge (or vertex). The point must be strictly
  // closer to q than p. The intersection may not be on
  // an edge involving points in ignore_pts.
  std::optional<BigVec2> GetClosestIntersection(
      const BigVec2 &p,
      const BigVec2 &q,
      const std::unordered_set<int> &ignore_pts,
      bool verbose = false) {
    if (verbose) {
      printf("From %s -> %s\n", VecString(p).c_str(),
             VecString(q).c_str());
    }
    const BigRat sqdist_p_to_q = distance_squared(p, q);

    // The closest point matching the criteria.
    std::optional<BigVec2> closest;
    BigRat closest_sqdist = BigRat(0);

    auto TryPoint = [&](const BigVec2 &v) {
        // Must be strictly closer to q.
        const BigRat q_sqdist = distance_squared(v, q);
        if (q_sqdist < sqdist_p_to_q) {
          BigRat sqdist = length_squared(v - p);
          if (!closest.has_value() || sqdist < closest_sqdist) {
            if (verbose) {
              printf("    " AYELLOW("(new best)") "\n");
            }
            closest = {v};
            closest_sqdist = sqdist;
          }
        }
      };

    auto TryEdge = [&](int u, int v) {
        if (ignore_pts.contains(u) ||
            ignore_pts.contains(v)) return;
        CHECK(u < v);
        const BigVec2 uv = Two(points[u]);
        const BigVec2 vv = Two(points[v]);
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
    auto Sample = [this](const BigVec2 &v2) -> std::pair<int, int> {
        std::vector<int> ps = ProjectThroughMesh(v2);
        CHECK(ps.size() == 2) << "We expect every projected point to "
          "have both a top and bottom intersection, but got: " << ps.size()
        << "\nProjecting point: " << VecString(v2)
        << "\n" << Error();
        return {ps[0], ps[1]};
      };

    for (int idx = 0; idx < input_polygon.size(); idx++) {
      if (VERBOSE) {
        printf(ABGCOLOR(0, 255, 255,
                        ADARKGREY("    --- in poly %d ---    ")) "\n",
               idx);
      }
      BigVec2 p = input_polygon[idx];
      const BigVec2 &q = input_polygon[(idx + 1) % input_polygon.size()];

      // Repeatedly find intersections between p and q.

      std::pair<int, int> pp = Sample(p);
      std::pair<int, int> qq = Sample(q);
      if (VERBOSE) {
        printf("Top cut: %d->%d\n", pp.first, qq.first);
      }

      hole.push_back(pp);

      while (pp != qq) {
        if (DEBUG) {
          SaveMesh(std::format("split{}", filename_index));
        }
        filename_index++;


        // Split0 lgtm.

        // Split1 looks probably good now.

        if (VERBOSE) {
          printf("Cut top vertex " AWHITE("%d") " to " AWHITE("%d") "\n",
                 pp.first, qq.first);
        }

        // The face we are currently carving. If the mesh is already
        // well-formed, we can't intersect with an edge that's already
        // connected to one of these vertices (and spurious intersections)
        // cause problems.
        std::unordered_set<int> ignore_pts = {
          pp.first, pp.second,
          qq.first, qq.second};

        auto io = GetClosestIntersection(p, q, ignore_pts);

        if (!io.has_value()) {
          if (VERBOSE) {
            printf("No more intersections.\n");
          }
          // No more intersections. Then we are done.
          // MaybeAddEdge(pp.first, qq.first);
          // MaybeAddEdge(pp.second, qq.second);
          break;
        }

        const BigVec2 &i2 = io.value();
        // i2 is a point between p and q.
        // TODO: Could assert this, since we require it for
        // termination.
        CHECK(i2 != p);
        if (VERBOSE) {
          printf("Took intersection at %s.\n", VecString(i2).c_str());
        }

        // It could snap to the same point, though.
        auto rr = Sample(i2);
        if (VERBOSE) {
          printf("Snapped top to %d\n", rr.first);
        }
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
    if (VERBOSE) {
      printf("Triangles:\n");
      for (const auto &[a, b, c] : work_triangles) {
        printf("  %d-%d-%d\n", a, b, c);
      }
    }
    if (VERBOSE) {
      printf("Splits:\n");
      for (const auto &[ab, c] : split_edges) {
        printf("  %d-%d: %d\n", ab.first, ab.second, c);
      }
    }
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
    CHECK(missing == 0) << Error();
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

  // Account for splits on the edge loops.
  void FixEdgeLoops() {
    if (VERBOSE) {
      printf("Triangles:\n");
      for (const auto &[a, b, c] : work_triangles) {
        printf("  %d-%d-%d\n", a, b, c);
      }
    }
    if (VERBOSE) {
      printf("Splits:\n");
      for (const auto &[ab, c] : split_edges) {
        printf("  %d-%d: %d\n", ab.first, ab.second, c);
      }
    }
    if (VERBOSE) {
      printf("Loop:\n");
      for (const auto &[t, b] : hole) {
        printf("  %d / %d\n", t, b);
      }
    }

    std::vector<std::pair<int, int>> out;
    int added = 0;
    for (int i = 0; i < hole.size(); i++) {
      int pt, pb, qt, qb;
      std::tie(pt, pb) = hole[i];
      std::tie(qt, qb) = hole[(i + 1) % hole.size()];

      if (VERBOSE) {
        printf("Raw %d/%d -> %d/%d\n", pt, pb, qt, qb);
      }

      // Skip exact duplicate points.
      if (pt == pb &&
          qt == qb) continue;

      for (;;) {
        bool t = pt == qt || HasEdge(pt, qt);
        bool b = pb == qb || HasEdge(pb, qb);
        if (VERBOSE) {
          printf("Inner %d/%d -> %d/%d %s/%s\n",
                 pt, pb, qt, qb,
                 t ? "ok" : "_", b ? "ok" : "_"
                 );
        }

        if (t && b) {
          out.emplace_back(pt, pb);
          goto next;
        } else if (!t && !b) {
          std::optional<int> ot = GetSplitEdge(pt, qt);
          std::optional<int> ob = GetSplitEdge(pb, qb);
          CHECK(ot.has_value()) << std::format("{}-{} ?", pt, qt);
          CHECK(ob.has_value()) << std::format("{}-{} ?", pb, qb);
          out.emplace_back(pt, pb);
          pt = ot.value();
          pb = ob.value();
          added += 2;
        } else if (!t) {
          // We expect to normally stay in sync, but if only
          // one was split, we can handle this by duplicating
          // points on one side.
          std::optional<int> ot = GetSplitEdge(pt, qt);
          CHECK(ot.has_value()) << std::format("{}-{} ?", pt, qt);
          out.emplace_back(pt, pb);
          pt = ot.value();
          added++;
        } else {
          CHECK(!b);
          std::optional<int> ob = GetSplitEdge(pb, qb);
          CHECK(ob.has_value()) << std::format("{}-{} ?", pb, qb);
          out.emplace_back(pt, pb);
          pb = ob.value();
          added++;
        }
      }

    next:;
    }

    if (added > 0) {
      printf("Added %d intermediate points\n", added);
    }
    hole = std::move(out);
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
    if (VERBOSE) {
      printf("Removed %d triangles in hole.\n", dropped);
    }
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
    // TODO: Improve mesh.
    // TODO: Garbage collect.
    Mesh3D ret;
    for (const BigVec3 &v : points) {
      ret.vertices.push_back(SmallVec(v));
    }
    ret.triangles = out_triangles;
    for (const auto &tri : work_triangles)
      ret.triangles.push_back(tri);

    return ret;
  }

  std::string Error() const {
    std::string out;
    AppendFormat(&out, "Input poly:\n");
    for (const BigVec2 &v : input_polygon) {
      AppendFormat(&out, "  {{{},{}}}\n",
                   v.x.ToString(), v.y.ToString());
    }
    return out;
  }

  // TODO: This may work, but it doesn't find any simplifications
  // in my examples. I think I need something fancier.
  void SimplifyColinear() const {
    printf("Simplify colinear. In: %d triangles\n",
           (int)work_triangles.size());

    // If we have two colinear edges a-b and b-c,
    // like this:
    //
    //       d      //
    //      /|\     //
    //     / | \    //
    //    /  |  \   //
    //   a---b---c  //
    //    \  |  /   //
    //     \ | /    //
    //      \|/     //
    //       e      //
    //
    // Then a-b-e and b-c-e are coplanar, and
    // a-d-b and d-c-b are coplanar. (But these
    // pairs need not be). We can collapse a-b-c to
    // a single edge:
    //
    //       d      //
    //      / \     //
    //     /   \    //
    //    /     \   //
    //   a-------c  //
    //    \     /   //
    //     \   /    //
    //      \ /     //
    //       e      //

    for (;;) {
      std::unordered_map<int, std::set<int>> adjacent;
      auto AddEdge = [&](int u, int v) {
          adjacent[u].insert(v);
          adjacent[v].insert(u);
        };
      for (const auto &[a, b, c] : work_triangles) {
        AddEdge(a, b);
        AddEdge(b, c);
        AddEdge(a, c);
      }

      // These are vertices that are being deleted. We don't
      // consider their neighbors for simplification.
      std::unordered_set<int> to_delete;

      std::vector<std::tuple<int, int, int>> new_triangles;
      new_triangles.reserve(work_triangles.size());

      // Now try to find a point we can delete.
      for (const auto &[b, neighbors] : adjacent) {
        printf("%d has %d neighbors:", b, (int)neighbors.size());
        for (int i : neighbors) {
          printf(" %d", i);
        }
        printf("\n");
        if (neighbors.size() == 4) {
          // If we already marked one of the neighbors
          // for deletion, we need to wait for a later
          // round to act on this.
          if ([&]{
              for (int n : neighbors)
                if (to_delete.contains(n))
                  return true;
              return false;
            }()) continue;

          std::vector<int> vs(neighbors.begin(), neighbors.end());
          CHECK(vs.size() == 4);

          const int i1 = vs[0];
          const int i2 = vs[1];
          const int i3 = vs[2];
          const int i4 = vs[3];

          printf("Consider %d with neighbors %d, %d, %d, %d\n",
                 b, i1, i2, i3, i4);

          const BigVec3 &v0 = points[b];
          const BigVec3 &v1 = points[i1];
          const BigVec3 &v2 = points[i2];
          const BigVec3 &v3 = points[i3];
          const BigVec3 &v4 = points[i4];

          auto Colinear = [&](const BigVec3 &a,
                              const BigVec3 &c) {
              BigVec3 ab = v0 - a;
              BigVec3 bc = c - v0;
              return CrossIsZero(ab, bc);
            };

          // As in the diagram above.
          auto Simplify = [&](int a, int c,
                              int d, int e) {

              to_delete.insert(b);

              // The two new triangles that result from
              // deleting b.
              AddTriangleTo(&new_triangles, a, d, c);
              AddTriangleTo(&new_triangles, a, c, e);
            };

          if (Colinear(v1, v2)) {
            Simplify(i1, i2, i3, i4);
          } else if (Colinear(v1, v3)) {
            Simplify(i1, i3, i2, i4);
          } else if (Colinear(v2, v3)) {
            Simplify(i2, i3, i1, i4);
          } else if (Colinear(v2, v4)) {
            Simplify(i2, i4, i1, i3);
          } else if (Colinear(v3, v4)) {
            Simplify(i3, i4, i1, i2);
          }
        }
      }

      // Now copy any triangles that don't involve deleted points.
      for (const auto &[a, b, c] : work_triangles) {
        if (!to_delete.contains(a) &&
            !to_delete.contains(b) &&
            !to_delete.contains(c)) {
          AddTriangleTo(&new_triangles, a, b, c);
        }
      }

      new_triangles = std::move(work_triangles);

      if (to_delete.empty()) break;
    }

    printf("Simplify colinear. Out: %d triangles\n",
           (int)work_triangles.size());
  }
};
}  // namespace

Mesh3D BigMakeHole(const Polyhedron &polyhedron,
                   const std::vector<vec2> &polygon) {
  std::vector<BigVec2> bigpolygon;
  for (const vec2 &v : polygon)
    bigpolygon.emplace_back(BigRat::FromDouble(v.x),
                            BigRat::FromDouble(v.y));

  BigHoleMaker maker(polyhedron, bigpolygon);
  maker.Split();
  if (DEBUG) maker.SaveMesh("split");
  maker.FixEdgeLoops();
  maker.CheckEdgeLoops();
  maker.RemoveHole();
  if (DEBUG) maker.SaveMesh("removehole");
  maker.RepairHole();
  if (DEBUG) maker.SaveMesh("repairhole");
  maker.SimplifyColinear();
  if (DEBUG) maker.SaveMesh("simplifycolinear");

  maker.SaveMesh("final");

  return maker.GetMesh();
}
