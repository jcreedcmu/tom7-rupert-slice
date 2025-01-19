
#include "ansi.h"

#include <cstdint>
#include <cstdio>
#include <format>
#include <print>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <unordered_set>

#include "base/stringprintf.h"
#include "bignum/big-overloads.h"
#include "bignum/big.h"
#include "bignum/polynomial.h"
#include "polyhedra.h"
#include "util.h"

struct SymbolTable {
  int64_t counter = 0;
  std::unordered_set<std::string> used;
  std::string Fresh(std::string_view hint) {
    std::string actual = std::string(hint);
    while (used.contains(actual)) {
      actual = std::format("{}_v{}", hint, counter);
      counter++;
    }
    used.insert(actual);
    return actual;
  }
};

static std::string Fresh(std::string_view hint = "") {
  static SymbolTable *table = new SymbolTable;
  if (hint.empty()) hint = "o";
  return table->Fresh(hint);
}

// For symbolic reasoning we want an exact representation of
// coordinates. One thing that will generally work here is
// to express each coordinate as the root of a polynomial in
// a single variable t.


Polynomial Constant(int c) {
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

SymbolicPolyhedron SymbolicCube() {
  std::vector<P3> vertices;
  auto AddVertex = [&vertices](int x, int y, int z) {
      int idx = (int)vertices.size();
      vertices.push_back(P3(x, y, z));
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
  return SymbolicPolyhedron{
    .vertices = std::move(vertices),
    .faces = faces,
    .name = "cube",
  };
}


struct Z3Real {
  explicit Z3Real(int i) : Z3Real((int64_t)i) {}
  explicit Z3Real(int64_t i) : s(std::format("{}.0", i)) {}
  explicit Z3Real(std::string s) : s(std::move(s)) {}
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

Z3Real operator+(const Z3Real &a, const Z3Real &b) {
  return Z3Real(std::format("(+ {} {})", a.s, b.s));
}

Z3Real operator-(const Z3Real &a, const Z3Real &b) {
  return Z3Real(std::format("(- {} {})", a.s, b.s));
}

Z3Real operator*(const Z3Real &a, const Z3Real &b) {
  return Z3Real(std::format("(* {} {})", a.s, b.s));
}

Z3Real operator/(const Z3Real &a, const Z3Real &b) {
  return Z3Real(std::format("(/ {} {})", a.s, b.s));
}

Z3Vec3 operator*(const Z3Vec3 &a, const Z3Real &b) {
  return Z3Vec3(a.x * b, a.y * b, a.z * b);
}

Z3Vec3 operator+(const Z3Vec3 &a, const Z3Vec3 &b) {
  return Z3Vec3(a.x + b.x, a.y + b.y, a.z + b.z);
}

Z3Vec2 operator*(const Z3Vec2 &a, const Z3Real &b) {
  return Z3Vec2(a.x * b, a.y * b);
}

Z3Vec2 operator+(const Z3Vec2 &a, const Z3Vec2 &b) {
  return Z3Vec2(a.x + b.x, a.y + b.y);
}

std::string ToZ3(std::string_view var, const Polynomial &p) {
  if (p.sum.empty()) return "0.0";
  // Otherwise, a sum...
  std::string out = "(+";

  auto Exp = [](std::string_view v, int e) -> std::string {
      if (e == 0) {
        return "1.0";
      }

      if (e == 1) {
        return std::string(v);
      }

      std::string out = "(*";
      for (int i = 0; i < e; i++) {
        AppendFormat(&out, " {}", std::string(v));
      }
      return out += ")";
    };

  auto TermString = [&](const Term &t) -> std::string {
      if (t.product.empty()) return "1.0";
      CHECK(t.product.size() == 1 &&
            t.product.begin()->first == "t") << "Must be a polynomial in a "
        "single variable t.";
      return Exp(var, t.product.begin()->second);
    };

  for (const auto &[t, coeff] : p.sum) {
    AppendFormat(&out, " (* {} {})", coeff.ToString(), TermString(t));
  }

  out += ")";
  return out;
}

void EmitVertex(std::string_view var,
                const Polynomial &p,
                std::string *out) {
  AppendFormat(out, "(declare-const {} Real)\n", var);
  AppendFormat(out, "(assert (= {} 0.0))\n", ToZ3(var, p));
}

std::vector<Z3Vec3> EmitPolyhedron(const SymbolicPolyhedron &poly,
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

static Z3Real NewReal(std::string *out, std::string_view name_hint = "") {
  std::string r = Fresh(name_hint);
  AppendFormat(out, "(declare-const {} Real)\n", r);
  return Z3Real(r);
}

static Z3Real NameReal(std::string *out, const Z3Real &r,
                       std::string_view name_hint = "") {
  std::string v = Fresh(name_hint);
  AppendFormat(out, "(define-fun {} () Real {})\n", v, r.s);
  return Z3Real(v);
}

static Z3Vec3 NameVec3(std::string *out, const Z3Vec3 &v,
                       std::string_view name_hint = "v") {
  Z3Real x = NameReal(out, v.x, std::format("{}_x", name_hint));
  Z3Real y = NameReal(out, v.y, std::format("{}_y", name_hint));
  Z3Real z = NameReal(out, v.z, std::format("{}_z", name_hint));
  return Z3Vec3(Z3Real(x), Z3Real(y), Z3Real(z));
}

static Z3Vec2 NameVec2(std::string *out, const Z3Vec2 &v,
                       std::string_view name_hint = "v") {
  Z3Real x = NameReal(out, v.x, std::format("{}_x", name_hint));
  Z3Real y = NameReal(out, v.y, std::format("{}_y", name_hint));
  return Z3Vec2(Z3Real(x), Z3Real(y));
}

static Z3Quat NewQuat(std::string *out, std::string_view v) {
  Z3Real qx = NewReal(out, std::format("q{}_x", v));
  Z3Real qy = NewReal(out, std::format("q{}_y", v));
  Z3Real qz = NewReal(out, std::format("q{}_z", v));
  Z3Real qw = NewReal(out, std::format("q{}_w", v));
  return Z3Quat(qx, qy, qz, qw);
}

static std::vector<Z3Vec2> EmitShadow(std::string *out,
                                      const std::vector<Z3Vec3> vin,
                                      const Z3Quat &q,
                                      const Z3Vec2 &trans,
                                      std::string_view name_hint) {
  Z3Real xx = q.x * q.x;
  Z3Real yy = q.y * q.y;
  Z3Real zz = q.z * q.z;
  Z3Real ww = q.w * q.w;

  // This normalization term is |v|^-2, which is
  // 1/(sqrt(x^2 + y^2 + z^2 + w^2))^2, so we can avoid
  // the square root! It's always multipled by 2 in the
  // terms below, so we also do that here.
  Z3Real two_s = NameReal(out,
                          Z3Real(2) / (xx + yy + zz + ww),
                          std::format("{}_two_s", name_hint));

  Z3Real xy = q.x * q.y;
  Z3Real zx = q.z * q.x;
  Z3Real zw = q.z * q.w;
  Z3Real yw = q.y * q.w;
  Z3Real yz = q.y * q.z;
  Z3Real xw = q.x * q.w;

  /*
    1 - 2s(y^2 + z^2)    ,   2s (x y - z w)      ,   2s (x z + y w)
    2s (x y + z w)       ,   1 - 2s(x^2 + z^2)   ,   2s (y z - x w)
    2s (x z - y w)       ,   2s (y z + x w)      ,   1 - 2s(x^2 + y^2)
  */

  // Following yocto, the matrix consists of three columns called "x",
  // "y", "z", each with "x", "y", "z" coordinates.

  // We give these names since they appear many times (for each vertex!)
  std::string m = std::format("{}m", name_hint);
  // left column
  Z3Vec3 mx =
    NameVec3(out, Z3Vec3(Z3Real(1) - two_s * (yy + zz),
                         two_s * (xy + zw),
                         two_s * (zx - yw)), std::format("{}x", m));
  // middle
  Z3Vec3 my =
    NameVec3(out, Z3Vec3(two_s * (xy - zw),
                         Z3Real(1) - two_s * (xx + zz),
                         two_s * (yz + xw)), std::format("{}y", m));
  // right
  Z3Vec3 mz =
    NameVec3(out, Z3Vec3(two_s * (zx + yw),
                         two_s * (yz - xw),
                         Z3Real(1) - two_s * (xx + yy)),
             std::format("{}z", m));

  // Now multiply each vertex.
 std::vector<Z3Vec2> ret;
 ret.reserve(vin.size());
 for (int i = 0; i < vin.size(); i++) {
   const Z3Vec3 &v = vin[i];

   Z3Vec3 x = mx * v.x;
   Z3Vec3 y = my * v.y;
   Z3Vec3 z = mz * v.z;

   // Just discarding the z coordinates.
   Z3Vec2 proj(x.x + y.x + z.x + trans.x,
               x.y + y.y + z.y + trans.y);

   Z3Vec2 p = NameVec2(out, proj, std::format("s{}{}", name_hint, i));
   ret.push_back(p);
 }

 return ret;
}


void EmitProblem(const SymbolicPolyhedron &outer,
                 const SymbolicPolyhedron &inner,
                 std::string_view filename) {
  std::string out;

  // Polyhedra in their base orientations.
  std::vector<Z3Vec3> outer_verts = EmitPolyhedron(outer, "o", &out);
  std::vector<Z3Vec3> inner_verts = EmitPolyhedron(inner, "i", &out);

  // Parameters.
  Z3Quat qo = NewQuat(&out, "o");
  Z3Quat qi = NewQuat(&out, "i");
  Z3Vec2 trans(NewReal(&out, "tx"), NewReal(&out, "ty"));

  Z3Vec2 id(Z3Real("0.0"), Z3Real("0.0"));

  // Polyhedra in their rotated orientations.
  std::vector<Z3Vec2> outer_shadow =
    EmitShadow(&out, outer_verts, qo, id, "o");
  std::vector<Z3Vec2> inner_shadow =
    EmitShadow(&out, inner_verts, qi, trans, "i");

  // XXX containment constraints

  AppendFormat(&out, "\n");
  AppendFormat(&out, "(check-sat)\n");
  AppendFormat(&out, "(get-model)\n");

  Util::WriteFile(filename, out);
  std::print("Wrote {}\n", filename);
}


static void Emit() {
  SymbolicPolyhedron cube = SymbolicCube();
  EmitProblem(cube, cube, "cube.z3");
}

int main(int argc, char **argv) {
  ANSI::Init();

  Emit();

  printf("OK\n");
  return 0;
}
