
#ifndef _RUPERTS_POLYHEDRA_H
#define _RUPERTS_POLYHEDRA_H

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "arcfour.h"
#include "base/logging.h"
#include "randutil.h"

#include "yocto_matht.h"

using vec2 = yocto::vec<double, 2>;
using vec3 = yocto::vec<double, 3>;
using vec4 = yocto::vec<double, 4>;
using mat4 = yocto::mat<double, 4>;
using quat4 = yocto::quat<double, 4>;
using frame3 = yocto::frame<double, 3>;

// We never change the connectivity of the objects
// in question, so we can avoid copying the faces.
//
struct Faces {
  // Each face is represented as the list of vertex (indices) that
  // circumscribe it. The vertices may appear in clockwise or
  // counter-clockwise order.
  std::vector<std::vector<int>> v;
  // For each vertex id, its immediate neighbors, in ascending order.
  std::vector<std::vector<int>> neighbors;
  // An arbitrary triangulation of the polyhedron (each "face" is a
  // triangle, but now faces can be coplanar).
  std::vector<std::tuple<int, int, int>> triangulation;

  // Number of vertices we expect.
  int NumVertices() const { return (int)neighbors.size(); }

  // Computes the neighbors from the faces. This assumes that
  // very vertex is on at least one face, which will be true
  // for well-formed polyhedra.
  Faces(int num_vertices, std::vector<std::vector<int>> v);
};

struct Polyhedron {
  // A polyhedron is nominally centered around (0,0).
  // This vector contains the positions of the vertices
  // in the polyhedron. The indices of the vertices are
  // significant.
  std::vector<vec3> vertices;
  // Not owned. Note that the routines below allocate
  // a new Faces object that you have to manage.
  const Faces *faces = nullptr;
  // The optional name of the polyhedron.
  const char *name = "";
};

// A polyhedron projected to 2D. This generally creates
// overlapping polygons, possibly also with coincident
// vertices.
struct Mesh2D {
  // As above.
  std::vector<vec2> vertices;
  // Not owned.
  const Faces *faces = nullptr;
};

inline vec4 VecFromQuat(const quat4 &q) {
  return vec4{.x = q.x, .y = q.y, .z = q.z, .w = q.w};
}

inline quat4 QuatFromVec(const vec4 &q) {
  return quat4{.x = q.x, .y = q.y, .z = q.z, .w = q.w};
}

std::string VecString(const vec3 &v);
std::string VecString(const vec2 &v);
std::string QuatString(const quat4 &q);
std::string FrameString(const frame3 &f);
std::string FormatNum(uint64_t n);

// For an oriented edge from v0 to v1, return the signed
// distance to that edge. Negative distance means to the left.
// Note: This cannot be used to find the signed distance to
// a polygon, because of ambiguity when the closest point is
// one of the vertices.
inline double SignedDistanceToEdge(const vec2 &v0, const vec2 &v1,
                                   const vec2 &p) {
  vec2 edge = v1 - v0;

  vec2 p_edge = p - v0;
  double cx = yocto::cross(edge, p_edge);
  // Signed. Negative means on the left.
  double dist = cx / yocto::length(edge);

  // Consider the endpoints. These are unsigned distances.
  double d0 = yocto::length(p_edge);
  double d1 = yocto::length(p - v1);

  if (d0 < std::abs(dist)) dist = d0 * yocto::sign(cx);
  if (d1 < std::abs(dist)) dist = d1 * yocto::sign(cx);

  return dist;
}

// Unsigned distance to the edge.
double DistanceToEdge(const vec2 &v0, const vec2 &v1, const vec2 &p);

// Rotate the polyhedron. They share the same faces pointer.
inline Polyhedron Rotate(const Polyhedron &p, const frame3 &frame) {
  Polyhedron ret = p;
  for (vec3 &v : ret.vertices) {
    v = yocto::transform_point(frame, v);
  }
  return ret;
}

inline Polyhedron Rotate(const Polyhedron &p, const quat4 &quat) {
  return Rotate(p, yocto::rotation_frame(quat));
}

inline Mesh2D Translate(const Mesh2D &m, const vec2 &t) {
  Mesh2D ret = m;
  for (vec2 &v : ret.vertices) {
    v += t;
  }
  return ret;
}

bool IsConvex(const std::vector<vec2> &points,
              const std::vector<int> &polygon);

// Maximum distance between any two points.
double Diameter(const Polyhedron &p);

// Create the shadow of the polyhedron on the x-y plane.
// The mesh's faces object aliases the input polyhedron's.
Mesh2D Shadow(const Polyhedron &p);

double DistanceToHull(
    const std::vector<vec2> &points, const std::vector<int> &hull,
    const vec2 &pt);

double DistanceToMesh(const Mesh2D &mesh, const vec2 &pt);

// Returns the convex hull (a single polygon) as indices into the
// vertex list (e.g. Mesh2D::vertices). This is the intuitive "gift
// wrapping" algorithm; much faster approaches exist, but we have a
// small number of vertices and this is in the outer loop.
//
// TODO: Now we have some need for a faster convex hull. I think
// we can at least make this algorithm faster by using the connectivity
// of the convex polyhedron we've projected. (Make sure to consider
// the case that an entire polygonal face is projected to a line,
// however!)
//
// This function doesn't handle collinear or coincident points well,
// so it's not really recommended.
std::vector<int> GiftWrapConvexHull(const std::vector<vec2> &vertices);

// Compute the convex hull, using Graham's scan. O(n lg n).
// This is typically slower than QuickHull.
std::vector<int> GrahamScan(const std::vector<vec2> &vertices);

// Compute the convex hull, using the QuickHull algorithm. This
// is not faster for the problem sizes here (~100 vertices) but
// is O(n lg n) time asymptotically.
std::vector<int> QuickHull(const std::vector<vec2> &v);

// The area of the convex hull; should also work for any simple
// polygon.
double AreaOfHull(const Mesh2D &mesh, const std::vector<int> &hull);

// Faces of a polyhedron must be planar. This computes the
// total error across all faces. If it is far from zero,
// something is wrong, but exact zero is not expected due
// to floating point imprecision.
double PlanarityError(const Polyhedron &p);

inline quat4 RandomQuaternion(ArcFour *rc) {
  const auto &[x, y, z, w] = RandomUnit4D(rc);
  return quat4{.x = x, .y = y, .z = z, .w = w};
}

inline vec2 Project(const vec3 &point, const mat4 &proj) {
  vec4 pp = proj * vec4{.x = point.x, .y = point.y, .z = point.z, .w = 1.0};
  return vec2{.x = pp.x / pp.w, .y = pp.y / pp.w};
}

// Point-in-polygon test using the winding number algorithm
bool PointInPolygon(const vec2 &point,
                    const std::vector<vec2> &vertices,
                    const std::vector<int> &polygon);

inline bool InMesh(const Mesh2D &mesh, const vec2 &pt) {
  for (const std::vector<int> &face : mesh.faces->v)
    if (PointInPolygon(pt, mesh.vertices, face))
      return true;

  return false;
}

inline bool InHull(const Mesh2D &mesh, const std::vector<int> &hull,
                   const vec2 &pt) {
  return PointInPolygon(pt, mesh.vertices, hull);
}

// Sample two faces from the polyhedron that are not parallel to each
// other.
std::pair<int, int> TwoNonParallelFaces(ArcFour *rc, const Polyhedron &poly);

// Signed distance to the triangle from the point p. Vertex order
// does not matter. Negative sign means the interior of the triangle.
double TriangleSignedDistance(vec2 p0, vec2 p1, vec2 p2, vec2 p);

// Generate some polyhedra. Note that each call new-ly allocates a
// Faces object, which is then owned by the caller. Many of these are
// not fast because they do some kind of search (e.g. a
// polynomial-time convex hull calculation) to find the connectivity.
// You should reuse the base shape and share the faces object rather
// than keep generating new ones.

// Platonic
Polyhedron Tetrahedron();
Polyhedron Cube();
Polyhedron Dodecahedron();
Polyhedron Icosahedron();
Polyhedron Octahedron();

// Archimedean
Polyhedron TruncatedTetrahedron();
Polyhedron Cuboctahedron();
Polyhedron TruncatedCube();
Polyhedron TruncatedOctahedron();
Polyhedron Rhombicuboctahedron();
Polyhedron TruncatedCuboctahedron();
Polyhedron SnubCube();
Polyhedron Icosidodecahedron();
Polyhedron TruncatedDodecahedron();
Polyhedron TruncatedIcosahedron();
Polyhedron Rhombicosidodecahedron();
Polyhedron TruncatedIcosidodecahedron();
Polyhedron SnubDodecahedron();

// Catalan
Polyhedron TriakisTetrahedron();
Polyhedron RhombicDodecahedron();
Polyhedron TriakisOctahedron();
Polyhedron TetrakisHexahedron();
Polyhedron DeltoidalIcositetrahedron();
Polyhedron DisdyakisDodecahedron();
Polyhedron DeltoidalHexecontahedron();
Polyhedron PentagonalIcositetrahedron();
Polyhedron RhombicTriacontahedron();
Polyhedron TriakisIcosahedron();
Polyhedron PentakisDodecahedron();
Polyhedron DisdyakisTriacontahedron();
Polyhedron PentagonalHexecontahedron();

#endif
