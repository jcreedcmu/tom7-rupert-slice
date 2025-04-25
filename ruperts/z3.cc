
#include "z3.h"

#include <format>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <mutex>

#include "base/stringprintf.h"
#include "bignum/polynomial.h"
#include "polyhedra.h"
#include "threadutil.h"

std::string SymbolTable::Fresh(std::string_view hint) {
  std::string actual = std::string(hint);
  MutexLock ml(&mu);
  while (used.contains(actual)) {
    actual = std::format("{}_v{}", hint, counter);
    counter++;
  }
  used.insert(actual);
  return actual;
}


std::string PolynomialToZ3(std::string_view var, const Polynomial &p) {
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

SymbolicPolyhedron SymbolicTetrahedron() {
  std::vector<P3> vertices;
  auto AddVertex = [&vertices](int x, int y, int z) {
      int idx = (int)vertices.size();
      vertices.push_back(P3(x, y, z));
      return idx;
    };

  int a = AddVertex(1, 1, 1);
  int b = AddVertex(1, -1, -1);
  int c = AddVertex(-1, 1, -1);
  int d = AddVertex(-1, -1, 1);

  std::vector<std::vector<int>> fs;
  fs.reserve(3);

  fs.push_back({a, c, b});
  fs.push_back({a, d, c});
  fs.push_back({c, d, b});
  fs.push_back({a, b, d});

  Faces *faces = new Faces(4, std::move(fs));
  return SymbolicPolyhedron{
    .vertices = std::move(vertices),
    .faces = faces,
    .name = "tetrahedron",
  };
}

template<class T, class F>
static std::string JoinOp(std::string_view op,
                          std::string_view unit,
                          const std::vector<T> &v,
                          const F &f) {
  if (v.size() == 1) return f(v[0]);
  if (v.empty()) return std::string(unit);
  std::string ret = std::format("({}", op);
  for (const T &b : v) {
    AppendFormat(&ret, " {}", f(b));
  }
  return ret + ")";
}

Z3Real Dot(const Z3Vec3 &a, const Z3Vec3 &b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

Z3Vec3 Cross(const Z3Vec3 &a, const Z3Vec3 &b) {
  return Z3Vec3{
    a.y * b.z - a.z * b.y,
    a.z * b.x - a.x * b.z,
    a.x * b.y - a.y * b.x,
  };
}

Z3Real Sum(const std::vector<Z3Real> &v) {
  return Z3Real(
      JoinOp("+", "0.0", v, [](const Z3Real &b) { return b.s; }));
}

Z3Vec2 Sum(const std::vector<Z3Vec2> &v) {
  std::string x = JoinOp("+", "0.0", v, [](const Z3Vec2 &b) { return b.x.s; });
  std::string y = JoinOp("+", "0.0", v, [](const Z3Vec2 &b) { return b.y.s; });
  return Z3Vec2(Z3Real(x), Z3Real(y));
}

Z3Bool Or(const std::vector<Z3Bool> &v) {
  return Z3Bool(
      JoinOp("or", "true", v, [](const Z3Bool &b) { return b.s; }));
}

void AssertRotationFrame(std::string *out, const Z3Frame &frame) {
  // f.x: f.y: f.x:
  //  x.x  y.x  z.x    a b c
  //  x.y  y.y  z.y    d e f
  //  x.z  y.z  z.z    g h i

  const Z3Real &a = frame.x.x;
  const Z3Real &b = frame.y.x;
  const Z3Real &c = frame.z.x;
  const Z3Real &d = frame.x.y;
  const Z3Real &e = frame.y.y;
  const Z3Real &f = frame.z.y;
  const Z3Real &g = frame.x.z;
  const Z3Real &h = frame.y.z;
  const Z3Real &i = frame.z.z;

  // These are implied by orthogonality, but maybe helpful.
  AppendFormat(out, ";; every element of the matrix is in [-1, 1].\n");
  for (const auto &m : {a, b, c, d, e, f, g, h, i}) {
    AppendFormat(out, "(assert (>= {} -1.0))\n", m.s);
    AppendFormat(out, "(assert (<= {} 1.0))\n", m.s);
  }

  Z3Real aei = a * e * i;
  Z3Real bfg = b * f * g;
  Z3Real cdh = c * d * h;
  Z3Real ceg = c * e * g;
  Z3Real bdi = b * d * i;
  Z3Real afh = a * f * h;

  Z3Real det = aei + bfg + cdh - ceg - bdi - afh;

  // PERF: A determinant of -1 would be a reflection (and rotation),
  // which is actually fine for our problem since the polyhedra are
  // symmetric. So we could relax this if we wanted. If the matrix
  // is orthogonal (below) then its determinant will be 1 or -1.
  // Determinant is +1.
  AppendFormat(out, "\n; determinant is +1.\n");
  AppendFormat(out, "(assert (= 1.0 {}))\n", det.s);

  // Orthogonality: f * f^T = id. f * f^T is:
  //   a^2 + b^2 + c^2     ad + be + cf      ag + bh + ci
  //   da + eb + fc        d^2 + e^2 + f^2   dg + eh + fi
  //   ga + hb + ic        gd + he + if      g^2 + h^2 + i^2
  // The matrix will be symmetric, so we just check the upper
  // triangular part.

  AppendFormat(out, "\n; matrix is orthogonal.\n");
  // Rows are unit length.
  AppendFormat(out, "(assert (= 1.0 {}))\n", (a * a + b * b + c * c).s);
  AppendFormat(out, "(assert (= 1.0 {}))\n", (d * d + e * e + f * f).s);
  AppendFormat(out, "(assert (= 1.0 {}))\n", (g * g + h * h + i * i).s);
  // Cols are unit length. Redundant, but maybe helpful?
  AppendFormat(out, "(assert (= 1.0 {}))\n", (a * a + d * d + g * g).s);
  AppendFormat(out, "(assert (= 1.0 {}))\n", (b * b + e * e + h * h).s);
  AppendFormat(out, "(assert (= 1.0 {}))\n", (c * c + f * f + i * i).s);

  // Rows are orthogonal to one another (dot product zero).
  AppendFormat(out, "(assert (= 0.0 {}))\n", (a * d + b * e + c * f).s);
  AppendFormat(out, "(assert (= 0.0 {}))\n", (a * g + b * h + c * i).s);
  AppendFormat(out, "(assert (= 0.0 {}))\n", (d * g + e * h + f * i).s);
  // Same for columns. (Again, redundant.)
  AppendFormat(out, "(assert (= 0.0 {}))\n", (a * b + d * e + g * h).s);
  AppendFormat(out, "(assert (= 0.0 {}))\n", (a * c + d * f + g * i).s);
  AppendFormat(out, "(assert (= 0.0 {}))\n", (b * c + e * f + h * i).s);
}

Z3Bool SameNonzeroSign(std::string *out,
                       const Z3Real &a, const Z3Real &b) {
  Z3Real aval = NameReal(out, a, "sgna");
  Z3Real bval = NameReal(out, b, "sgnb");
  return Z3Bool(
      std::format("(or "
                  "(and (< {} 0.0) (< {} 0.0)) "
                  "(and (> {} 0.0) (> {} 0.0)))",
                  aval.s, bval.s, aval.s, bval.s));
}

Z3Bool EmitInTriangle(std::string *out,
                      const Z3Vec2 &a, const Z3Vec2 &b, const Z3Vec2 &c,
                      const Z3Vec2 &pt) {

  // The idea behind this test is that for each edge, we check
  // to see if the test point is on the same side as a reference
  // point, which is the third point of the triangle.
  auto SameSide = [out](const Z3Vec2 &u, const Z3Vec2 &v,
                        const Z3Vec2 &p1, const Z3Vec2 &p2) -> Z3Bool {
      Z3Vec2 edge = v - u;
      Z3Real c1 = Cross(edge, p1 - u);
      Z3Real c2 = Cross(edge, p2 - u);

      // Excluding the edge itself.
      return SameNonzeroSign(out, c1, c2);
    };

  return SameSide(a, b, c, pt) &&
    SameSide(b, c, a, pt) &&
    SameSide(c, a, b, pt);
}

// See the implementation in big-polyhedra.cc.
Z3Frame NonUnitRotationFrame(std::string *out, const Z3Quat &v) {
  Z3Real two(2);

  Z3Real xx = v.x * v.x;
  Z3Real yy = v.y * v.y;
  Z3Real zz = v.z * v.z;
  Z3Real ww = v.w * v.w;

  Z3Real two_s = DeclareReal(out, Z3Real(2) / (xx + yy + zz + ww), "scale");

  Z3Real xy = v.x * v.y;
  Z3Real zx = v.z * v.x;
  Z3Real zw = v.z * v.w;
  Z3Real yw = v.y * v.w;
  Z3Real yz = v.y * v.z;
  Z3Real xw = v.x * v.w;

  Z3Real one(1);

  return Z3Frame(
    // Left
    Z3Vec3{
      one - two_s * (yy + zz),
      two_s * (xy + zw),
      two_s * (zx - yw)
    },
    // Middle
    Z3Vec3{
      two_s * (xy - zw),
      one - two_s * (xx + zz),
      two_s * (yz + xw)
    },
    // Right
    Z3Vec3{
      two_s * (zx + yw),
      two_s * (yz - xw),
      one - two_s * (xx + yy)
    });
}

inline Z3Frame InverseRigid(const Z3Frame &frame) {
  Z3Vec3 x = Z3Vec3(frame.x.x, frame.y.x, frame.z.x);
  Z3Vec3 y = Z3Vec3(frame.x.y, frame.y.y, frame.z.y);
  Z3Vec3 z = Z3Vec3(frame.x.z, frame.y.z, frame.z.z);
  return Z3Frame(x, y, z);
}



// PERF! Expand this and discard the components we don't use.
[[maybe_unused]]
static Z3Vec3 ReferenceViewPosFromNonUnitQuat(std::string *out,
                                              const Z3Quat &q) {
  Z3Frame frame = NonUnitRotationFrame(out, q);
  // We just apply the inverse rotation to (0, 0, 1).
  Z3Frame iframe = InverseRigid(frame);
  return TransformPoint(iframe, Z3Vec3(Z3Real(0), Z3Real(0), Z3Real(1)));
}


// Same as Reference version, but skipping work we don't need.
Z3Vec3 ViewPosFromNonUnitQuat(std::string *out, const Z3Quat &q) {
  Z3Real one(1);
  Z3Real two(2);

  Z3Real xx = q.x * q.x;
  Z3Real yy = q.y * q.y;
  Z3Real zz = q.z * q.z;
  Z3Real ww = q.w * q.w;

  Z3Real two_s = two / (xx + yy + zz + ww);

  Z3Real zx = q.z * q.x;
  Z3Real yw = q.y * q.w;
  Z3Real yz = q.y * q.z;
  Z3Real xw = q.x * q.w;

  return DeclareVec3(
      out,
      Z3Vec3(two_s * (zx - yw),
             two_s * (yz + xw),
             one - two_s * (xx + yy)),
      "view");
}
