
#include "ansi.h"

#include <cstdio>
#include <format>
#include <print>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "base/stringprintf.h"
#include "bignum/big-overloads.h"
#include "bignum/big.h"
#include "bignum/polynomial.h"
#include "polyhedra.h"
#include "util.h"

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

void EmitPolyhedron(const SymbolicPolyhedron &poly,
                    std::string_view p,
                    std::string *out) {
  for (int i = 0; i < poly.vertices.size(); i++) {
    std::string vx = std::format("{}_{}_x", p, i);
    std::string vy = std::format("{}_{}_y", p, i);
    std::string vz = std::format("{}_{}_z", p, i);
    EmitVertex(vx, poly.vertices[i].x, out);
    EmitVertex(vy, poly.vertices[i].y, out);
    EmitVertex(vz, poly.vertices[i].z, out);
  }
}

std::string AppendQuatParams(std::string_view v, std::string *out) {
  AppendFormat(out,
               "(declare-const q{}_x Real)\n"
               "(declare-const q{}_y Real)\n"
               "(declare-const q{}_z Real)\n"
               "(declare-const q{}_w Real)\n",
               v, v, v, v);
  return std::format("q{}", v);
}

void EmitRotation(std::string_view vin,
                  std::string_view q,
                  std::string_view vout,
                  // temporary name for matrix components
                  std::string_view m,
                  std::string *out) {

  // quaternion components
  auto Times2 = [q](std::string_view a, std::string_view b) {
      return std::format("(* {}_{} {}_{})", q, a, q, b);
    };

  std::string xx = Times2("x", "x");
  std::string yy = Times2("y", "y");
  std::string zz = Times2("z", "z");
  std::string ww = Times2("w", "w");

  // This normalization term is |v|^-2, which is
  // 1/(sqrt(x^2 + y^2 + z^2 + w^2))^2, so we can avoid
  // the square root! It's always multipled by 2 in the
  // terms below, so we also do that here.
  std::string two_s = std::format("(/ 2.0 (+ {} {} {} {}))",
                                  xx, yy, zz, ww);

  std::string xy = Times2("x", "y");
  std::string zx = Times2("z", "x");
  std::string zw = Times2("z", "w");
  std::string yw = Times2("y", "w");
  std::string yz = Times2("y", "z");
  std::string xw = Times2("x", "w");

  // Note that the yocto matrix looks a bit different from this. On the
  // diagonal, it's doing the trick that since v^2 = 1, we can write
  // 1 + s(y^2 + z^2) as (x^2 + y^2 + z^2 + w^2) - 2(y^2 + z^2) and
  // then cancel terms.
  //
  // There are also some sign differences. I think this is because
  // of differences in the handedness of the coordinate system.
  // This should not matter as long as we're consistent, but I need
  // to verify.

  // m00 m01 m02
  // m10 m11 m12
  // m20 m21 m22

  /*
    1 - 2s(y^2 + z^2)    ,   2s (x y - z w)      ,   2s (x z + y w)
    2s (x y + z w)       ,   1 - 2s(x^2 + z^2)   ,   2s (y z - x w)
    2s (x z - y w)       ,   2s (y z + x w)      ,   1 - 2s(x^2 + y^2)
  */

  /*

    {{w * w + x * x - y * y - z * z, (x * y + z * w) * 2, (z * x - y * w) * 2},
      {(x * y - z * w) * 2, w * w - x * x + y * y - z * z, (y * z + x * w) * 2},
      {(z * x + y * w) * 2, (y * z - x * w) * 2, w * w - x * x - y * y + z * z},
      {0, 0, 0}};

   */

  // Following yocto, the matrix consists of three columns called "x",
  // "y", "z", each with "x", "y", "z" coordinates.
  auto M = [m](int r, int c) {
      return std::format("m{}{}{}", m, c, r);
    };

  // Let's do this in column-major order so it's easiest to compare
  // to the yocto code (and big-polyhedra), but of course the order
  // does not matter here.
  // Left column
  AppendFormat(out, "(define-fun {} () Real (- 1.0 (* {} (+ {} {}))))\n",
               M(0, 0), two_s, yy, zz);
  AppendFormat(out, "(define-fun {} () Real (* {} (+ {} {})))\n",
               M(1, 0), two_s, xy, zw);
  AppendFormat(out, "(define-fun {} () Real (* {} (- {} {})))\n",
               M(2, 0), two_s, zx, yw);
  // Middle column
  AppendFormat(out, "(define-fun {} () Real (* {} (- {} {})))\n",
               M(0, 1), two_s, xy, zw);
  AppendFormat(out, "(define-fun {} () Real (- 1.0 (* {} (+ {} {}))))\n",
               M(1, 1), two_s, xx, zz);
  AppendFormat(out, "(define-fun {} () Real (* {} (+ {} {})))\n",
               M(2, 1), two_s, yz, xw);
  // Right column
  AppendFormat(out, "(define-fun {} () Real (* {} (+ {} {})))\n",
               M(0, 2), two_s, zx, yw);
  AppendFormat(out, "(define-fun {} () Real (* {} (- {} {})))\n",
               M(1, 2), two_s, yz, xw);
  AppendFormat(out, "(define-fun {} () Real (- 1.0 (* {} (+ {} {}))))\n",
               M(2, 2), two_s, xx, yy);


  // we have tmpa : vec2 = m.x * v.x
  //         tmpb : vec2 = m.y * v.y
  //         tmpc : vec2 = m.z * v.z

  std::string tmpa = std::format("t{}_a", m);
  std::string tmpb = std::format("t{}_b", m);
  std::string tmpc = std::format("t{}_c", m);

  auto Times3To2 = [](std::string_view out,
                      int col,
                      std::string_view e) {
      AppendFormat(out, "(define-fun
    };

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

}


void EmitProblem(const SymbolicPolyhedron &outer,
                 const SymbolicPolyhedron &inner,
                 std::string_view filename) {
  std::string out;

  // Polyhedra in their base orientations.
  EmitPolyhedron(outer, "o", &out);
  EmitPolyhedron(inner, "i", &out);

  // Parameters.
  std::string qo = AppendQuatParams("o", &out);
  std::string qi = AppendQuatParams("i", &out);
  AppendFormat(&out,
               "(declare-const t_x Real)\n"
               "(declare-const t_y Real)\n");

  // Polyhedra in their rotated orientations.
  EmitRotation("o", qo, "ro", "mo", &out);
  EmitRotation("i", qi, "ri", "mi", &out);


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
