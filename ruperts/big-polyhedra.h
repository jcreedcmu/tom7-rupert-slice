
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

// TODO: To cc-lib bignum

struct BigVec3 {
  BigVec3(BigRat x, BigRat y, BigRat z) :
    x(std::move(x)), y(std::move(y)), z(std::move(z)) {}
  BigRat x = BigRat(0), y = BigRat(0), z = BigRat(0);
};

struct BigVec2 {
  BigVec2(BigRat x, BigRat y) :
    x(std::move(x)), y(std::move(y)) {}
  BigRat x = BigRat(0), y = BigRat(0);
};

// Quaternion as xi + yj + zk + w
struct BigQuat {
  BigQuat(BigRat x, BigRat y, BigRat z, BigRat w) :
    x(std::move(x)), y(std::move(y)), z(std::move(z)), w(std::move(w)) {}
  BigRat x = BigRat(0), y = BigRat(0), z = BigRat(0), w = BigRat(1);
};

struct BigPoly {
  std::vector<BigVec3> vertices;
  const Faces *faces = nullptr;
};

struct BigMesh2D {
  std::vector<BigVec2> vertices;
  const Faces *faces = nullptr;
};

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

// Exact equality.
inline bool operator ==(const BigVec2 &a, const BigVec2 &b) {
  return a.x == b.x && a.y == b.y;
}

inline BigRat cross(const BigVec2 &a, const BigVec2 &b) {
  return a.x * b.y - a.y * b.x;
}

inline quat4 SmallQuat(const BigQuat &q) {
  return quat4{q.x.ToDouble(), q.y.ToDouble(), q.z.ToDouble(), q.w.ToDouble()};
}

std::string VecString(const BigVec2 &v);
std::string QuatString(const BigQuat &q);

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

inline static vec3 SmallVec(const BigVec3 &v) {
  return vec3(v.x.ToDouble(), v.y.ToDouble(), v.z.ToDouble());
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

inline BigVec3 operator*(const BigVec3& a, BigRat b) {
  return BigVec3(a.x * b, a.y * b, a.z * b);
}

inline BigVec3 TransformPoint(const BigFrame &f, const BigVec3 &v) {
  return f.x * v.x + f.y * v.y + f.z * v.z + f.o;
}

// XXX These don't work how I'd expect. Fix or delete.
BigVec3 RotatePoint(const BigQuat &q, const BigVec3 &v);
BigPoly Rotate(const BigQuat &q, const BigPoly &poly);

// Rotate the entire polyhedron using the frame. Exact.
BigPoly Rotate(const BigFrame &f, const BigPoly &poly);
// Project to 2D along the z axis.
BigMesh2D Shadow(const BigPoly &poly);
// Translate the entire mesh by the vector t.
BigMesh2D Translate(const BigVec2 &t, const BigMesh2D &m);

std::vector<int> BigHull(const std::vector<BigVec2> &bigvs);

Polyhedron SmallPoly(const BigPoly &big);
Mesh2D SmallMesh(const BigMesh2D &big);

BigPoly MakeBigPolyFromVertices(std::vector<BigVec3> vertices);

// Some polyhedra with arbitrary precision.
BigPoly BigRidode(int digits);

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

inline BigRat LengthSquared(const BigVec2 &a) {
  return a.x * a.x + a.y + a.y;
}

inline BigRat DistanceSquared(const BigVec2 &a, const BigVec2 &b) {
  BigVec2 edge(a.x - b.x, a.y - b.y);
  return LengthSquared(edge);
}

int GetClosestPoint(const BigMesh2D &mesh, const BigVec2 &pt);
