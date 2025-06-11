
#ifndef _RUPERTS_POLYHEDRA_H
#define _RUPERTS_POLYHEDRA_H

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include "arcfour.h"
#include "base/logging.h"
#include "randutil.h"
#include "mesh.h"
#include "yocto_matht.h"

using vec2 = yocto::vec<double, 2>;
using vec3 = yocto::vec<double, 3>;
using vec4 = yocto::vec<double, 4>;
using mat4 = yocto::mat<double, 4>;
using quat4 = yocto::quat<double, 4>;
using frame3 = yocto::frame<double, 3>;
using frame2 = yocto::frame<double, 2>;

inline constexpr uint8_t SYM_UNKNOWN = 0b0;
inline constexpr uint8_t SYM_TETRAHEDRAL = 0b1;
inline constexpr uint8_t SYM_OCTAHEDRAL = 0b10;
inline constexpr uint8_t SYM_ICOSAHEDRAL = 0b100;
using SymmetryGroup = uint8_t;

// We never change the connectivity of the objects
// in question, so we can avoid copying the faces.
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

  // Computes the neighbors and triangulation from the faces. This
  // assumes that every vertex is on at least one face, which will be
  // true for well-formed polyhedra.
  Faces(int num_vertices, std::vector<std::vector<int>> v);

  Faces() {}

  // Like the constructor, but returns nullptr if something goes wrong
  // (like there are vertices that are not on faces).
  static Faces *Create(int num_vertices, std::vector<std::vector<int>> v);

 private:
  bool Init(int num_vertices, std::vector<std::vector<int>> v);
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
  std::string name;
  // The optional symmetry group(s).
  SymmetryGroup symmetry = SYM_UNKNOWN;
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
std::string Points2DString(const std::vector<vec2> &v);
std::string QuatString(const quat4 &q);
std::string FrameString(const frame3 &f);
std::string MatString(const mat4 &m);
std::string FormatNum(uint64_t n);

inline bool TriangleIsDegenerate(const vec3 &v0,
                                 const vec3 &v1,
                                 const vec3 &v2) {
  vec3 v0v1 = v1 - v0;
  vec3 v0v2 = v2 - v0;
  vec3 cross_product = yocto::cross(v0v1, v0v2);
  // This is a about 2e-16, which is pretty generous.
  return yocto::length_squared(cross_product) <
    std::numeric_limits<double>::epsilon();
}

// Returns a frame representing rotation by angle around the origin.
inline frame2 rotation_frame2(double angle) {
  auto s = std::sin(angle);
  auto c = std::cos(angle);
  return {{c, s}, {-s, c}, {0.0, 0.0}};
}

// Euclidean distance (non-negative) to the line segment from
// the point. This may be one of the endpoints.
double PointLineDistance(
    // Line segment
    const vec2 &v0, const vec2 &v1,
    // Point to test
    const vec2 &pt);

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

// Precomputation for testing points in a polygon. This
// should be faster if you need to call PointInPolygon
// many times for the same polygon.
double SignedAreaOfConvexPoly(const std::vector<vec2> &points);
bool IsConvexAndScreenClockwise(const std::vector<vec2> &poly);
#define POLYTESTER_USE_BB 0
struct PolyTester2D {
  static constexpr bool SELF_CHECK = false;

  // The polygon must be convex, screen clockwise, and must include
  // the origin. These conditions are not checked.
  PolyTester2D(const std::vector<vec2> &poly) : poly(poly) {
    if (SELF_CHECK) {
      CHECK(SignedAreaOfConvexPoly(poly) > 0.0);
      CHECK(IsConvexAndScreenClockwise(poly));
    }

    // TODO: Precompute.
    edges.reserve(poly.size());
    edge_sqlens.reserve(poly.size());

    for (int i = 0; i < poly.size(); i++) {
      const vec2 &v0 = poly[i];
      const vec2 &v1 = poly[(i + 1) % poly.size()];
      const vec2 edge = v1 - v0;
      const double sqlen = length_squared(edge);
      edges.push_back(edge);
      edge_sqlens.push_back(sqlen);
      #if POLYTESTER_USE_BB
      min_x = std::min(min_x, v0.x);
      max_x = std::max(max_x, v0.x);
      min_y = std::min(min_y, v0.y);
      max_y = std::max(max_y, v0.y);
      #endif
    }
  }

  // Returns nullopt if the point is inside. Otherwise, minimum squared
  // distance to the polygon.
  std::optional<double> SquaredDistanceOutside(const vec2 &pt) const;

  bool IsInside(const vec2 &pt) const {
    return !SquaredDistanceOutside(pt).has_value();
  }

 private:
  double SquaredDistanceToPoly(const vec2 &pt) const;
  bool PointInPolygon(const vec2 &point) const;

  const std::vector<vec2> &poly;
  // parallel to the vertices. Represents the edge from the vertex
  // to the next one.
  std::vector<vec2> edges;
  std::vector<double> edge_sqlens;

  #if POLYTESTER_USE_BB
  // Bounding box.
  double min_x = std::numeric_limits<double>::infinity();
  double max_x = -std::numeric_limits<double>::infinity();
  double min_y = std::numeric_limits<double>::infinity();
  double max_y = -std::numeric_limits<double>::infinity();
  #endif
};


// Rotate (and translate, if the frame contains a translation) the polyhedron.
// They share the same faces pointer.
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


// Reflect the polyhedron across the XY plane.
// Shares faces with the argument (but they will be inside-out).
Polyhedron ReflectXY(const Polyhedron &p);

inline Mesh2D Translate(const Mesh2D &m, const vec2 &t) {
  Mesh2D ret = m;
  for (vec2 &v : ret.vertices) {
    v += t;
  }
  return ret;
}

bool IsHullConvex(const std::vector<vec2> &points,
                  const std::vector<int> &polygon);

bool IsPolyConvex(const std::vector<vec2> &poly);

// Screen clockwise = cartesian CCW.
bool IsConvexAndScreenClockwise(const std::vector<vec2> &poly);

// Maximum distance between any two points.
// Note: This is non-standard.
double Diameter(const Polyhedron &p);

// Normalize so that the most distant point from the origin is
// distance 1. This really only makes sense if the polyhedron
// contains the origin (and ideally is centered around it).
Polyhedron NormalizeRadius(const Polyhedron &p);

// Move the polyhedron so that the origin is at the center of
// mass of its vertices.
Polyhedron Recenter(const Polyhedron &p);

// Create the shadow of the polyhedron on the x-y plane.
// The mesh's faces object aliases the input polyhedron's.
Mesh2D Shadow(const Polyhedron &p);

// Non-negative distance to hull.
double DistanceToHull(
    const std::vector<vec2> &points, const std::vector<int> &hull,
    const vec2 &pt);

double SquaredDistanceToPoly(
    const std::vector<vec2> &poly, const vec2 &pt);

// Return the closest point (could be on an edge or a vertex) on
// the hull, and its distance.
std::pair<vec2, double> ClosestPointOnHull(
    const std::vector<vec2> &points, const std::vector<int> &hull,
    const vec2 &pt);

double DistanceToMesh(const Mesh2D &mesh, const vec2 &pt);

// Return the minimum distance between the two hulls; assumes that
// the inner is completely inside the outer.
double HullClearance(const std::vector<vec2> &outer_points,
                     const std::vector<int> &outer_hull,
                     const std::vector<vec2> &inner_points,
                     const std::vector<int> &inner_hull);

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
// Positive if screen clockwise (cartesian ccw) winding order;
// negative for screen ccw (cartesian cw).
double SignedAreaOfHull(const Mesh2D &mesh, const std::vector<int> &hull);
// Same, but with the points directly.
double SignedAreaOfConvexPoly(const std::vector<vec2> &points);

// In the inner loop, we compute a convex hull for a covex polyhedron
// centered at the origin, and then test whether many points are
// inside that hull. If the point's not close to the hull, this can
// be done much faster by just testing whether it's a known distance
// from the hull. This represents the circumscribed circle of the hull,
// centered at the origin.
//
// Note that this requires for correctness (but does not check) that
// the hull contains the origin.
struct HullCircumscribedCircle {
  HullCircumscribedCircle(const std::vector<vec2> &vertices,
                          const std::vector<int> &hull) {
    CHECK(hull.size() != 0);
    max_sqdist = yocto::length_squared(vertices[hull[0]]);
    for (int idx = 1; idx < hull.size(); idx++) {
      const vec2 &v = vertices[hull[idx]];
      max_sqdist = std::max(max_sqdist, yocto::length_squared(v));
    }
  }

  bool DefinitelyOutside(const vec2 &pt) const {
    return yocto::length_squared(pt) > max_sqdist;
  }

  double max_sqdist = 0.0;
};

// Same idea, but with an inscribed circle. This touches edges, not
// vertices.
//
// Note that this requires for correctness (but does not check) that
// the hull contains the origin.
struct HullInscribedCircle {
  HullInscribedCircle(const std::vector<vec2> &vertices,
                      const std::vector<int> &hull) {
    CHECK(hull.size() >= 3) << Points2DString(vertices);

    for (int i = 0; i < hull.size(); ++i) {
      const vec2 &v0 = vertices[hull[i]];
      const vec2 &v1 = vertices[hull[(i + 1) % hull.size()]];

      // Express as v0 + t * edge.
      vec2 edge = v1 - v0;
      // The vector to the closest point will be orthogonal to the edge.
      double t = -yocto::dot(v0, edge) / yocto::length_squared(edge);
      t = std::clamp(t, 0.0, 1.0);
      vec2 closest_point = v0 + t * edge;

      min_sqdist = std::min(min_sqdist, yocto::length_squared(closest_point));
    }
  }

  bool DefinitelyInside(const vec2 &pt) const {
    return yocto::length_squared(pt) < min_sqdist;
  }

  double min_sqdist = std::numeric_limits<double>::max();
};

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

// Point-in-polygon test using the winding number algorithm.
// Takes a vertex buffer and indices into that set.
bool PointInPolygon(const vec2 &point,
                    const std::vector<vec2> &vertices,
                    const std::vector<int> &polygon);

// Takes the polygon directly as vertices.
bool PointInPolygon(const vec2 &point,
                    const std::vector<vec2> &polygon);

// Is pt strictly within the triangle a-b-c? Exact. Works with both
// winding orders.
bool InTriangle(const vec2 &a, const vec2 &b, const vec2 &c,
                const vec2 &pt);

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

// Various tests for exactly zero, usually for assertions.

inline bool AllZero(const std::vector<vec2> &vec) {
  for (const vec2 &v : vec)
    if (v.x != 0.0 || v.y != 0.0)
      return false;
  return true;
}

inline bool AllZero(const vec3 &vec) {
  return vec.x == 0.0 && vec.y == 0.0 && vec.z == 0.0;
}

inline bool AllZero(const quat4 &q) {
  return q.x == 0.0 && q.y == 0.0 && q.z == 0.0 && q.w == 0.0;
}

inline bool AllZero(const frame3 &f) {
  return AllZero(f.x) && AllZero(f.y) && AllZero(f.z) && AllZero(f.o);
}

// Orient the mesh and save as STL.
void SaveAsSTL(const Polyhedron &poly, std::string_view filename);

// Generate little tetrahedra at the points, for debugging.
void DebugPointCloudAsSTL(const std::vector<vec3> &vertices,
                          std::string_view filename);

// Save polyhedron as JSON.
void SaveAsJSON(const Polyhedron &poly, std::string_view filename);

// Save solution as JSON.
void SaveAsJSON(const frame3 &outer_frame,
                const frame3 &inner_frame,
                std::string_view filename);


// Unpack a rigid frame into a rotation (as a normalized quaternion)
// and a translation vector. If the matrix is not actually a rigid
// transform, then the result may not be meaningful.
std::pair<quat4, vec3> UnpackFrame(const frame3 &f);

// For normalized vectors a and b (interpreted as orientations on
// the sphere), compute a rotation from a to b.
quat4 RotationFromAToB(const vec3 &a, const vec3 &b);

// See big-polyhedra.
vec3 ViewPosFromNonUnitQuat(const quat4 &q);

// Sample two faces from the polyhedron that are not parallel to each
// other.
std::pair<int, int> TwoNonParallelFaces(ArcFour *rc, const Polyhedron &poly);

// Signed distance to the triangle from the point p. Vertex order
// does not matter. Negative sign means the interior of the triangle.
double TriangleSignedDistance(vec2 p0, vec2 p1, vec2 p2, vec2 p);

// Standard loss function for solution procedure. This recomputes the
// entire problem; if some of the parameters are fixed (e.g. the outer
// transformation) then you should not use this!
//
// Requires that outer the polyhedron contain the origin after applying
// the frame. This is typical (P/A/C solids all contain it; usually we
// do not translate the outer poly). If you aren't sure, use LossFunction.
double LossFunctionContainsOrigin(const Polyhedron &poly,
                                  const frame3 &outer_frame,
                                  const frame3 &inner_frame);

// As previous, but with two potentially different polyhedra.
double HeteroLossFunctionContainsOrigin(const Polyhedron &outer_poly,
                                        const Polyhedron &inner_poly,
                                        const frame3 &outer_frame,
                                        const frame3 &inner_frame);

// Same, but does not assume the outer polyhedron contains the origin.
double LossFunction(const Polyhedron &poly,
                    const frame3 &outer_frame,
                    const frame3 &inner_frame);

// Same, but also compute a gradient when we have a solution (which
// is slow).
double FullLossContainsOrigin(const Polyhedron &poly,
                              const frame3 &outer_frame,
                              const frame3 &inner_frame);

// Get the ratio inner_area / outer_area, which is a reasonable metric
// for how good the solution is. Will be in [0, 1). Returns nullopt
// if the solution is not valid.
std::optional<double> GetRatio(const Polyhedron &poly,
                               const frame3 &outer_frame,
                               const frame3 &inner_frame);

// Get the minimum distance between the two hulls. Here a larger
// distance is better.
std::optional<double> GetClearance(const Polyhedron &poly,
                                   const frame3 &outer_frame,
                                   const frame3 &inner_frame);

TriangularMesh3D PolyToTriangularMesh(const Polyhedron &poly);

// Creates an approximate sphere by "triforce" subdivision of the
// icosahedron with the given depth (exponential).
TriangularMesh3D ApproximateSphere(int depth);

// A regular n-gon, extruded to the given depth. Centered at
// the origin. num_points is n, the number of points on the
// top and bottom faces; this will have 2n vertices.
Polyhedron NPrism(int64_t num_points, double depth);

// Like NPrism, but an anti-prism has triangular side faces, and the
// bottom and top faces have their vertices interleaved.
Polyhedron NAntiPrism(int64_t num_points, double depth);

// Takes ownership of the vertices, which should be a convex hull.
// Creates faces as all planes where all the other points are on one
// side. This is not fast; it's intended for a small number of
// vertices. The Faces pointer in the returned Polyhedron is owned
// by the caller. Can fail if the points are not actually a convex
// hull, returning nullopt.
std::optional<Polyhedron> PolyhedronFromConvexVertices(
    std::vector<vec3> vertices, std::string_view name = "");

// First compute the convex hull and discard points outside it.
// Then, the same as above. This only fails for degenerate point sets.
std::optional<Polyhedron> PolyhedronFromVertices(
    std::vector<vec3> vertices, std::string_view name = "");

// Return a newly constructed polyhedron (from below) by its name,
// or abort.
Polyhedron PolyhedronByName(std::string_view name);
// Nickname for abbreviated display. Not canonical; may change.
std::string PolyhedronShortName(std::string_view name);
// Get the proper name from the nickname.
std::string PolyhedronIdFromNickname(std::string_view nickname);
// Get the human name (e.g. with spaces between words).
std::string PolyhedronHumanName(std::string_view name);

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
