
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "base/logging.h"
#include "bignum/big-overloads.h"
#include "bignum/big.h"
#include "polyhedra.h"
#include "hashing.h"

// TODO: To cc-lib bignum

struct BigVec3 {
  BigVec3(BigRat x, BigRat y, BigRat z) :
    x(std::move(x)), y(std::move(y)), z(std::move(z)) {}
  BigVec3() {}
  BigRat x = BigRat(0), y = BigRat(0), z = BigRat(0);
};

struct BigVec2 {
  BigVec2(BigRat x, BigRat y) :
    x(std::move(x)), y(std::move(y)) {}
  BigVec2() {}
  BigRat x = BigRat(0), y = BigRat(0);
};

// Quaternion as xi + yj + zk + w
struct BigQuat {
  BigQuat(BigRat x, BigRat y, BigRat z, BigRat w) :
    x(std::move(x)), y(std::move(y)), z(std::move(z)), w(std::move(w)) {}
  BigQuat() {}
  BigRat x = BigRat(0), y = BigRat(0), z = BigRat(0), w = BigRat(1);
};

BigQuat MakeBigQuat(const quat4 &smallquat);
BigQuat ApproxBigQuat(const quat4 &smallquat, int64_t max_denom);

struct BigPoly {
  std::vector<BigVec3> vertices;
  const Faces *faces = nullptr;
  std::string name;
};

struct BigMesh2D {
  std::vector<BigVec2> vertices;
  const Faces *faces = nullptr;
};

// Scale the vector so that it has integer coordinates. The
// result is a canonical representation of the direction.
BigVec3 ScaleToMakeIntegral(const BigVec3 &a);

inline BigVec3 operator +(const BigVec3 &a, const BigVec3 &b) {
  return BigVec3(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline BigVec3 operator -(const BigVec3 &a, const BigVec3 &b) {
  return BigVec3(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline BigVec2 operator +(const BigVec2 &a, const BigVec2 &b) {
  return BigVec2(a.x + b.x, a.y + b.y);
}

inline BigVec2 operator -(const BigVec2 &a, const BigVec2 &b) {
  return BigVec2(a.x - b.x, a.y - b.y);
}

inline BigVec3 operator -(const BigVec3 &a) {
  return BigVec3(-a.x, -a.y, -a.z);
}

inline BigVec2 operator *(const BigVec2 &v, const BigRat &s) {
  return BigVec2(v.x * s, v.y * s);
}

inline BigVec2 operator *(const BigRat &s, const BigVec2 &v) {
  return BigVec2(v.x * s, v.y * s);
}

// Exact equality.
inline bool operator ==(const BigVec2 &a, const BigVec2 &b) {
  return a.x == b.x && a.y == b.y;
}

inline bool operator ==(const BigVec3 &a, const BigVec3 &b) {
  return a.x == b.x && a.y == b.y && a.z == b.z;
}

inline BigRat dot(const BigVec2 &a, const BigVec2 &b) {
  return a.x * b.x + a.y * b.y;
}

inline BigRat dot(const BigVec3 &a, const BigVec3 &b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline BigRat cross(const BigVec2 &a, const BigVec2 &b) {
  return a.x * b.y - a.y * b.x;
}

inline BigVec3 cross(const BigVec3 &a, const BigVec3 &b) {
  return BigVec3{
    a.y * b.z - a.z * b.y,
    a.z * b.x - a.x * b.z,
    a.x * b.y - a.y * b.x
  };
}

inline quat4 SmallQuat(const BigQuat &q) {
  return quat4{q.x.ToDouble(), q.y.ToDouble(), q.z.ToDouble(), q.w.ToDouble()};
}

inline BigRat length_squared(const BigVec2 &a) {
  return dot(a, a);
}

inline BigRat length_squared(const BigVec3 &a) {
  return dot(a, a);
}

inline BigVec3 operator *(const BigRat &s, const BigVec3 &v) {
  return BigVec3{s * v.x, s * v.y, s * v.z};
}

template<>
struct Hashing<BigVec3> {
  std::size_t operator()(const BigVec3 &v) const {
    uint64_t x = BigRat::HashCode(v.x);
    uint64_t y = BigRat::HashCode(v.y);
    uint64_t z = BigRat::HashCode(v.z);
    return ((x * 3) + y) * 7 + z;
  }
};

// With color
std::string VecString(const BigVec2 &v);
std::string VecString(const BigVec3 &v);
std::string QuatString(const BigQuat &q);
// For serialization to disk, etc.
std::string PlainVecString(const BigVec2 &v);
std::string PlainQuatString(const BigQuat &q);

BigQuat Normalize(const BigQuat &q, int digits);

inline BigQuat UnitInverse(const BigQuat &q) {
  return BigQuat(-q.x, -q.y, -q.z, q.w);
}

inline BigQuat operator*(const BigQuat &a, const BigQuat &b) {
  return BigQuat{
    a.x * b.w + a.w * b.x + a.y * b.w - a.z * b.y,
    a.y * b.w + a.w * b.y + a.z * b.x - a.x * b.z,
    a.z * b.w + a.w * b.z + a.x * b.y - a.y * b.x,
    a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
  };
}

inline vec3 SmallVec(const BigVec3 &v) {
  return vec3(v.x.ToDouble(), v.y.ToDouble(), v.z.ToDouble());
}

inline vec2 SmallVec(const BigVec2 &v) {
  return vec2(v.x.ToDouble(), v.y.ToDouble());
}

inline quat4 quat_conjugate(const quat4 &q) {
  return {-q.x, -q.y, -q.z, q.w};
}

// Rigid frames stored as a column-major affine transform matrix.
struct BigFrame {
  BigVec3 x = {BigRat(1), BigRat(0), BigRat(0)};
  BigVec3 y = {BigRat(0), BigRat(1), BigRat(0)};
  BigVec3 z = {BigRat(0), BigRat(0), BigRat(1)};
  BigVec3 o = {BigRat(0), BigRat(0), BigRat(0)};

  BigVec3& operator[](int i) {
    switch (i) {
    case 0: return x;
    case 1: return y;
    case 2: return z;
    case 3: return o;
    default:
      LOG(FATAL) << "Bad";
      return x;
    }
  }

  const BigVec3 &operator[](int i) const {
    switch (i) {
    case 0: return x;
    case 1: return y;
    case 2: return z;
    case 3: return o;
    default:
      LOG(FATAL) << "Bad";
      return x;
    }
  }
};

// Compute the rigid frame that rotates according to the unit
// quaternion v.
BigFrame RotationFrame(const BigQuat &v);

// Same, but not requiring a unit quaternion.
BigFrame NonUnitRotationFrame(const BigQuat &v);

inline BigVec3 operator*(const BigVec3 &a, BigRat b) {
  return BigVec3(a.x * b, a.y * b, a.z * b);
}

inline BigVec3 TransformPoint(const BigFrame &f, const BigVec3 &v) {
  return f.x * v.x + f.y * v.y + f.z * v.z + f.o;
}

inline BigVec2 TransformAndProjectPoint(const BigFrame &f, const BigVec3 &v) {
  // scale vector, but discard the z coordinate
  auto Times3To2 = [](const BigVec3 &u, const BigRat &r) {
    return BigVec2(u.x * r, u.y * r);
  };

  BigVec2 fx = Times3To2(f.x, v.x);
  BigVec2 fy = Times3To2(f.y, v.y);
  BigVec2 fz = Times3To2(f.z, v.z);
  // PERF this is always zero for our problems
  BigVec2 o = BigVec2(f.o.x, f.o.y);

  return fx + fy + fz + o;
}

// Small Fixed-size matrices stored in column major format.
struct BigMat3 {
  // left column
  BigVec3 x = {BigRat(1), BigRat(0), BigRat(0)};
  // middle column
  BigVec3 y = {BigRat(0), BigRat(1), BigRat(0)};
  // right column
  BigVec3 z = {BigRat(0), BigRat(0), BigRat(1)};

  BigVec3 &operator[](int i) {
    switch (i) {
    case 0: return x;
    case 1: return y;
    case 2: return z;
    default:
      LOG(FATAL) << "Index out of bounds.";
    }
  }
  const BigVec3& operator[](int i) const {
    switch (i) {
    case 0: return x;
    case 1: return y;
    case 2: return z;
    default:
      LOG(FATAL) << "Index out of bounds.";
    }
  }
};

// XXX These don't work how I'd expect. Fix or delete.
BigVec3 RotatePoint(const BigQuat &q, const BigVec3 &v);
BigPoly Rotate(const BigQuat &q, const BigPoly &poly);

// Rotate the entire polyhedron using the frame. Exact.
BigPoly Rotate(const BigFrame &f, const BigPoly &poly);
// Project to 2D along the z axis.
BigMesh2D Shadow(const BigPoly &poly);
// Translate the entire mesh by the vector t.
BigMesh2D Translate(const BigVec2 &t, const BigMesh2D &m);
// Rotate and project at once, which is faster because we don't
// need to compute the discarded z coordinates.
BigMesh2D RotateAndProject(const BigFrame &f, const BigPoly &poly);

Polyhedron SmallPoly(const BigPoly &big);
Mesh2D SmallMesh(const BigMesh2D &big);

BigPoly MakeBigPolyFromVertices(std::vector<BigVec3> vertices);

// Some polyhedra with arbitrary precision.
BigPoly BigRidode(int digits);
BigPoly BigDhexe(int digits);
BigPoly BigPhexe(int digits);
BigPoly BigScube(int digits);
BigPoly BigSdode(int digits);
BigPoly BigTriac(int digits);
// Some are exact, but we take an (ignored) digits argument anyway.
BigPoly BigCube(int digits);
BigPoly BigTetra(int digits);

// Returns true if the solution (doubles) is actually valid (using
// rational arithmetic with the specified precision).
bool ValidateSolution(const BigPoly &poly,
                      const frame3 &outer,
                      const frame3 &inner,
                      int digits);

// Point-in-polygon test using the winding number algorithm.
// Takes a vertex buffer and indices into that set.
bool PointInPolygon(const BigVec2 &point,
                    const std::vector<BigVec2> &vertices,
                    const std::vector<int> &polygon);

// Takes the polygon directly as vertices.
bool PointInPolygon(const BigVec2 &point,
                    const std::vector<BigVec2> &polygon);

// Is pt strictly within the triangle a-b-c? Exact. Works with both
// winding orders.
bool InTriangle(const BigVec2 &a, const BigVec2 &b, const BigVec2 &c,
                const BigVec2 &pt);

// Is the point strictly within any triangle in the mesh? Note that this
// excludes points that are inside the hull but only lie exactly on
// edges, like the center of a triangulated square. This is often not
// what you want! TODO: Fix!
bool InMesh(const BigMesh2D &mesh, const BigVec2 &pt);

// Check if the point is strictly within *any* triangle induced by the
// point set. Returns such a triangle if so. This also can exclude a
// point that lies exactly and exclusively on internal edges (e.g. the
// center of a triangulated square), although most of the time there
// will then be another triangle that includes it.
std::optional<std::tuple<int, int, int>>
InMeshExhaustive(const BigMesh2D &mesh, const BigVec2 &pt);

inline BigRat distance_squared(const BigVec2 &a, const BigVec2 &b) {
  BigVec2 edge(a.x - b.x, a.y - b.y);
  return length_squared(edge);
}

// Get the index of the closest mesh point to the target point.
int GetClosestPoint(const BigMesh2D &mesh, const BigVec2 &pt);

// Get the vertex index of the closest vertex (among those in the
// hull) to the target point. Also returns the *squared* distance.
std::pair<int, BigRat> GetClosestPoint(const std::vector<BigVec2> &vertices,
                                       const std::vector<int> &hull,
                                       const BigVec2 &pt);

// Get the convex hull. This is intended to be exact.
std::vector<int> BigQuickHull(const std::vector<BigVec2> &vertices);
// Fast(er), but approximate since it uses the double-based
// hull computation.
std::vector<int> BigHull(const std::vector<BigVec2> &bigvs);

// Check if the point is strictly within the convex hull. Either hull
// orientation works, but it must be a convex polygon (not just a
// point set).
bool InHull(const std::vector<BigVec2> &vertices,
            const std::vector<int> &hull,
            const BigVec2 &pt);

BigRat SquaredDistanceToClosestPointOnSegment(
    // Line segment
    const BigVec2 &v0,
    const BigVec2 &v1,
    // Point to test
    const BigVec2 &pt);

BigRat SquaredDistanceToHull(const std::vector<BigVec2> &vertices,
                             const std::vector<int> &hull,
                             const BigVec2 &pt);
