
#ifndef _RUPERTS_Z3_H
#define _RUPERTS_Z3_H

#include <cstdint>
#include <format>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <unordered_set>

#include "base/stringprintf.h"
#include "bignum/big.h"
#include "bignum/polynomial.h"
#include "polyhedra.h"
#include "big-polyhedra.h"

struct SymbolTable {
  std::string Fresh(std::string_view hint);
 private:
  int64_t counter = 0;
  std::unordered_set<std::string> used;
};

inline Polynomial Constant(int c) {
  return "t"_p - Polynomial(c);
}

struct P3 {
  P3(int x, int y, int z) : x(Constant(x)), y(Constant(y)), z(Constant(z)) {}
  Polynomial x, y, z;
};

struct SymbolicPolyhedron {
  std::vector<P3> vertices;
  const Faces *faces = nullptr;
  std::string name;
};

struct Z3Bool {
  explicit Z3Bool(std::string s) : s(std::move(s)) {}
  std::string s;
};

struct Z3Real {
  explicit Z3Real(int i) : Z3Real((int64_t)i) {}
  explicit Z3Real(int64_t i) : s(std::format("{}.0", i)) {}
  explicit Z3Real(std::string s) : s(std::move(s)) {}
  explicit Z3Real(const BigInt &i) :
    s(std::format("{}.0", i.ToString())) {}
  explicit Z3Real(const BigRat &r) {
    const auto &[n, d] = r.Parts();
    if (d == 1) {
      s = std::format("{}.0", n.ToString());
    } else {
      s = std::format("(/ {}.0 {}.0)",
                      n.ToString(),
                      d.ToString());
    }
  }
  Z3Real(const Z3Real &other) = default;
  Z3Real(Z3Real &&other) = default;
  Z3Real &operator =(const Z3Real &other) = default;
  Z3Real &operator =(Z3Real &&other) = default;
  std::string s;
 private:
  // avoid constructing from doubles, since there are many
  // values that cannot be represented exactly.
  Z3Real(double d) {}
};

struct Z3Vec3 {
  Z3Vec3(Z3Real x, Z3Real y, Z3Real z) :
    x(std::move(x)), y(std::move(y)), z(std::move(z)) {}
  Z3Real x;
  Z3Real y;
  Z3Real z;
};

struct Z3Vec2 {
  Z3Vec2(Z3Real x, Z3Real y) : x(std::move(x)), y(std::move(y)) {}
  Z3Real x;
  Z3Real y;
};

struct Z3Quat {
  Z3Quat(Z3Real x, Z3Real y, Z3Real z, Z3Real w) :
    x(std::move(x)), y(std::move(y)), z(std::move(z)), w(std::move(w)) {}
  Z3Real x;
  Z3Real y;
  Z3Real z;
  Z3Real w;
};

struct Z3Frame {
  Z3Frame(Z3Vec3 x, Z3Vec3 y, Z3Vec3 z) :
    x(std::move(x)), y(std::move(y)), z(std::move(z)) {}
  // Column-major, following Yocto.
  Z3Vec3 x;
  Z3Vec3 y;
  Z3Vec3 z;
};

inline std::string Fresh(std::string_view hint = "") {
  static SymbolTable *table = new SymbolTable;
  if (hint.empty()) hint = "o";
  return table->Fresh(hint);
}

SymbolicPolyhedron SymbolicCube();
SymbolicPolyhedron SymbolicTetrahedron();
SymbolicPolyhedron FromBigPoly(const BigPoly &poly);

inline Z3Real operator+(const Z3Real &a, const Z3Real &b) {
  return Z3Real(std::format("(+ {} {})", a.s, b.s));
}

inline Z3Real operator-(const Z3Real &a, const Z3Real &b) {
  return Z3Real(std::format("(- {} {})", a.s, b.s));
}

inline Z3Real operator*(const Z3Real &a, const Z3Real &b) {
  return Z3Real(std::format("(* {} {})", a.s, b.s));
}

inline Z3Real operator/(const Z3Real &a, const Z3Real &b) {
  return Z3Real(std::format("(/ {} {})", a.s, b.s));
}

inline Z3Vec3 operator*(const Z3Vec3 &a, const Z3Real &b) {
  return Z3Vec3(a.x * b, a.y * b, a.z * b);
}

inline Z3Vec3 operator+(const Z3Vec3 &a, const Z3Vec3 &b) {
  return Z3Vec3(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline Z3Vec2 operator*(const Z3Vec2 &a, const Z3Real &b) {
  return Z3Vec2(a.x * b, a.y * b);
}

inline Z3Vec2 operator+(const Z3Vec2 &a, const Z3Vec2 &b) {
  return Z3Vec2(a.x + b.x, a.y + b.y);
}

inline Z3Vec2 operator-(const Z3Vec2 &a, const Z3Vec2 &b) {
  return Z3Vec2(a.x - b.x, a.y - b.y);
}

inline Z3Real cross(const Z3Vec2 &a, const Z3Vec2 &b) {
  return a.x * b.y - a.y * b.x;
}

inline Z3Bool operator&&(const Z3Bool &a, const Z3Bool &b) {
  return Z3Bool(std::format("(and {} {})", a.s, b.s));
}

inline Z3Bool operator||(const Z3Bool &a, const Z3Bool &b) {
  return Z3Bool(std::format("(or {} {})", a.s, b.s));
}

Z3Real Sum(const std::vector<Z3Real> &v);
Z3Vec2 Sum(const std::vector<Z3Vec2> &v);
Z3Bool Or(const std::vector<Z3Bool> &v);

// XXX return Z3Real
std::string PolynomialToZ3(std::string_view var, const Polynomial &p);

inline void EmitVertex(std::string_view var,
                       const Polynomial &p,
                       std::string *out) {
  AppendFormat(out, "(declare-const {} Real)\n", var);
  AppendFormat(out, "(assert (= {} 0.0))\n", PolynomialToZ3(var, p));
}

inline std::vector<Z3Vec3> EmitPolyhedron(const SymbolicPolyhedron &poly,
                                          std::string_view p,
                                          std::string *out) {
  std::vector<Z3Vec3> vs;
  for (int i = 0; i < poly.vertices.size(); i++) {
    std::string vx = Fresh(std::format("{}_{}_x", p, i));
    std::string vy = Fresh(std::format("{}_{}_y", p, i));
    std::string vz = Fresh(std::format("{}_{}_z", p, i));
    EmitVertex(vx, poly.vertices[i].x, out);
    EmitVertex(vy, poly.vertices[i].y, out);
    EmitVertex(vz, poly.vertices[i].z, out);
    vs.emplace_back(Z3Real(vx), Z3Real(vy), Z3Real(vz));
  }

  return vs;
}

inline Z3Real NewReal(std::string *out, std::string_view name_hint = "") {
  std::string r = Fresh(name_hint);
  AppendFormat(out, "(declare-const {} Real)\n", r);
  return Z3Real(r);
}

inline Z3Vec3 NewVec3(std::string *out, std::string_view name_hint = "") {
  Z3Real x = NewReal(out, std::format("{}x", name_hint));
  Z3Real y = NewReal(out, std::format("{}y", name_hint));
  Z3Real z = NewReal(out, std::format("{}z", name_hint));
  return Z3Vec3(x, y, z);
}

inline Z3Real NameReal(std::string *out, const Z3Real &r,
                       std::string_view name_hint = "") {
  std::string v = Fresh(name_hint);
  AppendFormat(out, "(define-fun {} () Real {})\n", v, r.s);
  return Z3Real(v);
}

inline Z3Vec3 NameVec3(std::string *out, const Z3Vec3 &v,
                       std::string_view name_hint = "v") {
  Z3Real x = NameReal(out, v.x, std::format("{}_x", name_hint));
  Z3Real y = NameReal(out, v.y, std::format("{}_y", name_hint));
  Z3Real z = NameReal(out, v.z, std::format("{}_z", name_hint));
  return Z3Vec3(Z3Real(x), Z3Real(y), Z3Real(z));
}

inline Z3Vec2 NameVec2(std::string *out, const Z3Vec2 &v,
                       std::string_view name_hint = "v") {
  Z3Real x = NameReal(out, v.x, std::format("{}_x", name_hint));
  Z3Real y = NameReal(out, v.y, std::format("{}_y", name_hint));
  return Z3Vec2(Z3Real(x), Z3Real(y));
}

inline Z3Quat NewQuat(std::string *out, std::string_view v) {
  Z3Real qx = NewReal(out, std::format("q{}_x", v));
  Z3Real qy = NewReal(out, std::format("q{}_y", v));
  Z3Real qz = NewReal(out, std::format("q{}_z", v));
  Z3Real qw = NewReal(out, std::format("q{}_w", v));
  return Z3Quat(qx, qy, qz, qw);
}

inline Z3Frame NewFrame(std::string *out, std::string_view v) {
  Z3Vec3 x = NewVec3(out, std::format("{}x", v));
  Z3Vec3 y = NewVec3(out, std::format("{}y", v));
  Z3Vec3 z = NewVec3(out, std::format("{}z", v));
  return Z3Frame(x, y, z);
}

// Assert that the frame is a rotation frame (rigid rotation).
void AssertRotationFrame(std::string *out, const Z3Frame &frame);

// Test whether the sign of the two reals is the same, and nonzero.
Z3Bool SameNonzeroSign(std::string *out,
                       const Z3Real &a, const Z3Real &b);

// True if the point 'pt' is strictly inside the triangle a-b-c. If
// the triangle is degenerate, this will be false.
Z3Bool EmitInTriangle(std::string *out,
                      const Z3Vec2 &a, const Z3Vec2 &b, const Z3Vec2 &c,
                      const Z3Vec2 &pt);

#endif
