
// Generate Z3 formula.

#include <cstdint>
#include <cstdio>
#include <format>
#include <print>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <unordered_set>

#include "ansi.h"
#include "base/stringprintf.h"
#include "bignum/polynomial.h"
#include "polyhedra.h"
#include "symmetry-groups.h"
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

enum class Containment {
  TRIANGULATION,
  COMBINATION,
};


inline constexpr Containment containment = Containment::COMBINATION;

enum class Parameterization {
  QUATS,
  MATRICES,
  BOUNDED_MATRICES,
};

inline constexpr Parameterization parameterization =
  Parameterization::BOUNDED_MATRICES;
// Parameterization::QUATS;

inline constexpr SymmetryGroup symmetry_group =
  SymmetryGroup::OCTAHEDRAL;

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

struct Z3Bool {
  explicit Z3Bool(std::string s) : s(std::move(s)) {}
  std::string s;
};

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

Z3Vec2 operator-(const Z3Vec2 &a, const Z3Vec2 &b) {
  return Z3Vec2(a.x - b.x, a.y - b.y);
}

Z3Real cross(const Z3Vec2 &a, const Z3Vec2 &b) {
  return a.x * b.y - a.y * b.x;
}

static Z3Bool operator&&(const Z3Bool &a, const Z3Bool &b) {
  return Z3Bool(std::format("(and {} {})", a.s, b.s));
}

static Z3Bool operator||(const Z3Bool &a, const Z3Bool &b) {
  return Z3Bool(std::format("(or {} {})", a.s, b.s));
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

static Z3Real Sum(const std::vector<Z3Real> &v) {
  return Z3Real(
      JoinOp("+", "0.0", v, [](const Z3Real &b) { return b.s; }));
}

static Z3Vec2 Sum(const std::vector<Z3Vec2> &v) {
  std::string x = JoinOp("+", "0.0", v, [](const Z3Vec2 &b) { return b.x.s; });
  std::string y = JoinOp("+", "0.0", v, [](const Z3Vec2 &b) { return b.y.s; });
  return Z3Vec2(Z3Real(x), Z3Real(y));
}

static Z3Bool Or(const std::vector<Z3Bool> &v) {
  return Z3Bool(
      JoinOp("or", "true", v, [](const Z3Bool &b) { return b.s; }));
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

static Z3Vec3 NewVec3(std::string *out, std::string_view name_hint = "") {
  Z3Real x = NewReal(out, std::format("{}x", name_hint));
  Z3Real y = NewReal(out, std::format("{}y", name_hint));
  Z3Real z = NewReal(out, std::format("{}z", name_hint));
  return Z3Vec3(x, y, z);
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

static Z3Frame NewFrame(std::string *out, std::string_view v) {
  // Z3Vec3 x = NameVec3(out, NewVec3(out, std::format("{}x", v)));
  // Z3Vec3 y = NameVec3(out, NewVec3(out, std::format("{}y", v)));
  // Z3Vec3 z = NameVec3(out, NewVec3(out, std::format("{}z", v)));
  Z3Vec3 x = NewVec3(out, std::format("{}x", v));
  Z3Vec3 y = NewVec3(out, std::format("{}y", v));
  Z3Vec3 z = NewVec3(out, std::format("{}z", v));

  return Z3Frame(x, y, z);
}

static void AssertRotationFrame(std::string *out,
                                const Z3Frame &frame) {
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

static std::vector<Z3Vec2> EmitShadowFromFrame(std::string *out,
                                               const std::vector<Z3Vec3> &vin,
                                               const Z3Frame &frame,
                                               const Z3Vec2 &trans,
                                               std::string_view name_hint) {
  // Now multiply each vertex.
  std::vector<Z3Vec2> ret;
  ret.reserve(vin.size());
  for (int i = 0; i < vin.size(); i++) {
    const Z3Vec3 &v = vin[i];

    Z3Vec3 x = frame.x * v.x;
    Z3Vec3 y = frame.y * v.y;
    Z3Vec3 z = frame.z * v.z;

    // Just discarding the z coordinates.
    Z3Vec2 proj(x.x + y.x + z.x + trans.x,
                x.y + y.y + z.y + trans.y);

    Z3Vec2 p = NameVec2(out, proj, std::format("s{}{}", name_hint, i));
    ret.push_back(p);
  }

  return ret;
}

static std::vector<Z3Vec2> EmitShadowFromQuat(std::string *out,
                                              const std::vector<Z3Vec3> &vin,
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

  Z3Frame frame(mx, my, mz);

  return EmitShadowFromFrame(out, vin, frame, trans, name_hint);
}

static Z3Bool SameNonzeroSign(std::string *out,
                              const Z3Real &a, const Z3Real &b) {
  Z3Real aval = NameReal(out, a, "sgna");
  Z3Real bval = NameReal(out, b, "sgnb");
  return Z3Bool(
      std::format("(or "
                  "(and (< {} 0.0) (< {} 0.0)) "
                  "(and (> {} 0.0) (> {} 0.0)))",
                  aval.s, bval.s, aval.s, bval.s));
}

static Z3Bool EmitInTriangle(std::string *out,
                             const Z3Vec2 &a, const Z3Vec2 &b, const Z3Vec2 &c,
                             const Z3Vec2 &pt) {
  // The idea behind this test is that for each edge, we check
  // to see if the test point is on the same side as a reference
  // point, which is the third point of the triangle.
  auto SameSide = [out](const Z3Vec2 &u, const Z3Vec2 &v,
                        const Z3Vec2 &p1, const Z3Vec2 &p2) -> Z3Bool {
      Z3Vec2 edge = v - u;
      Z3Real c1 = cross(edge, p1 - u);
      Z3Real c2 = cross(edge, p2 - u);

      // Excluding the edge itself.
      return SameNonzeroSign(out, c1, c2);
    };

  return SameSide(a, b, c, pt) &&
    SameSide(b, c, a, pt) &&
    SameSide(c, a, b, pt);
}

static void AssertRotationBounds(std::string *out,
                                 SymmetryGroup sg,
                                 const Z3Frame &frame) {
  // Without loss of generality, we can put a bound α/2 on the angular
  // distance of the rotation. α is the maximum angular distance between
  // two vertices on the corresponding polyhedron (e.g. the icosahedron
  // for the icosahedral group). α/2 is thus a bound on the maximum
  // angulard distance between any point on the unit sphere and a vertex
  // of the symmetry group's polyhedron.
  //
  // Thinking about the cube: α will be 90 degrees. If we are trying
  // to rotate some distinguished vertex from the position s to the
  // position e (where θ is the angle subtended), we can always
  // produce an equivalent rotation with θ <= α/2. We do this by
  // picking the closest (angular distance) vertex v (in the original
  // orientation) to e; this can be no more than α/2 away. Then we
  // just apply the symmetry group operations to move s to v, and
  // then rotate from v to e with an angulard distance of no more than
  // α/2.
  //
  // This argument assumes vertex transitivity, but applies regardless
  // of the transitive set (e.g. if face transitive, consider the dual
  // polyhedron, which has the same symmetry group).
  //

  // There's a nice way to place bounds on the angle, which is to
  // use the trace of the matrix (sum of main diagonal).
  // trace(M) ≥ 1 + 2cos(β) ensures that the rotation has an
  // angular distance of no more than β.
  //

  Z3Real bound = [out, sg]() -> Z3Real {
      switch (sg) {
      case SymmetryGroup::OCTAHEDRAL: {
        // For the octahedral group, α = 90 degrees, and cos(α/2) is
        // 1/sqrt(2). So we have trace(M) ≥ 1 + 2/sqrt(2)
        // (Simplify: 2/sqrt(2) = (sqrt(2) * sqrt(2))/sqrt(2) = sqrt(2).)
        //
        // trace(M) ≥ 1 + sqrt(2)

        Z3Real sqrt2 = NewReal(out, "sqrt2");
        AppendFormat(out, "(assert (= 2.0 (* sqrt2 sqrt2)))\n");
        return Z3Real(1) + sqrt2;
      }
      case SymmetryGroup::ICOSAHEDRAL:
        // For the icosahedral group, α = arccos(1/sqrt(5)), and
        // cos(α/2) = sqrt((5 + sqrt(5)) / 10)    (wolfram alpha).
        // trace(M) ≥ 1 + 2 * sqrt((5 + sqrt(5)) / 10)

        // XXX We can express the bound exactly, but I don't think
        // it's likely to be helpful since it would involve roots of
        // roots?
        return Z3Real("2.701301616704079864363");
      default:
        LOG(FATAL) << "unimplemented!";
        return Z3Real("error");
      }
    }();


  Z3Real trace = frame.x.x + frame.y.y + frame.z.z;
  AppendFormat(out, ";; constrain rotation due to symmetries\n");
  AppendFormat(out, "(assert (>= {} {}))\n", trace.s, bound.s);
}

static std::pair<std::vector<Z3Vec2>, std::vector<Z3Vec2>>
MakeParameterization(std::string *out,
                     const std::vector<Z3Vec3> &outer_verts,
                     const std::vector<Z3Vec3> &inner_verts) {

  if (parameterization == Parameterization::QUATS) {
    // Parameters.
    Z3Quat qo = NewQuat(out, "o");
    Z3Quat qi = NewQuat(out, "i");
    Z3Vec2 trans(NewReal(out, "tx"), NewReal(out, "ty"));

    // AppendFormat(out, "(assert (= {} 0.0))\n", qi.x.s);
    // AppendFormat(out, "(assert (= {} 0.0))\n", qi.y.s);
    // AppendFormat(out, "(assert (= {} 0.0))\n", qi.z.s);
    // AppendFormat(out, "(assert (= {} 1.0))\n", qi.w.s);

    Z3Vec2 id(Z3Real(0), Z3Real(0));

    // Polyhedra in their rotated orientations.
    std::vector<Z3Vec2> outer_shadow =
      EmitShadowFromQuat(out, outer_verts, qo, id, "o");
    std::vector<Z3Vec2> inner_shadow =
      EmitShadowFromQuat(out, inner_verts, qi, trans, "i");

    return std::make_pair(outer_shadow, inner_shadow);
  } else if (parameterization == Parameterization::MATRICES ||
             parameterization == Parameterization::BOUNDED_MATRICES) {
    Z3Frame outer_frame = NewFrame(out, "o");
    Z3Frame inner_frame = NewFrame(out, "i");

    AssertRotationFrame(out, outer_frame);
    AssertRotationFrame(out, inner_frame);

    if (parameterization == Parameterization::BOUNDED_MATRICES) {
      AssertRotationBounds(out, symmetry_group, outer_frame);
      AssertRotationBounds(out, symmetry_group, inner_frame);
    }

    Z3Vec2 trans(NewReal(out, "tx"), NewReal(out, "ty"));

    Z3Vec2 id(Z3Real(0), Z3Real(0));

    std::vector<Z3Vec2> outer_shadow =
      EmitShadowFromFrame(out, outer_verts, outer_frame, id, "o");
    std::vector<Z3Vec2> inner_shadow =
      EmitShadowFromFrame(out, inner_verts, inner_frame, trans, "i");

    return std::make_pair(outer_shadow, inner_shadow);
  } else {
    LOG(FATAL) << "Bad parameterization";
    return {};
  }
}

static void EmitProblem(const SymbolicPolyhedron &outer,
                        const SymbolicPolyhedron &inner,
                        std::string_view filename) {
  std::string out;

  // Polyhedra in their base orientations.
  std::vector<Z3Vec3> outer_verts = EmitPolyhedron(outer, "o", &out);
  std::vector<Z3Vec3> inner_verts = EmitPolyhedron(inner, "i", &out);

  const auto &[outer_shadow, inner_shadow] =
    MakeParameterization(&out, outer_verts, inner_verts);

  // Containment constraints...
  if (containment == Containment::TRIANGULATION) {
    for (int i = 0; i < inner_shadow.size(); i++) {
      const Z3Vec2 &pt = inner_shadow[i];
      // Must be in some triangle.
      std::vector<Z3Bool> tris;
      for (const auto &[a, b, c] : outer.faces->triangulation) {
        tris.push_back(EmitInTriangle(&out,
                                      outer_shadow[a],
                                      outer_shadow[b],
                                      outer_shadow[c],
                                      pt));
      }
      AppendFormat(&out,
                   ";; inner point {} in triangle?\n"
                   "(assert {})\n", i, Or(tris).s);
    }
  } else if (containment == Containment::COMBINATION) {
    AppendFormat(&out,
                 "\n\n"
                 ";; Inner shadow is in outer shadow. Each point\n"
                 ";; must be a convex combination of the points in\n"
                 ";; the outer shadow.\n");
    for (int i = 0; i < inner_shadow.size(); i++) {
      const Z3Vec2 &ipt = inner_shadow[i];
      std::vector<Z3Real> weights;
      std::vector<Z3Vec2> weighted_sum;
      for (int j = 0; j < outer_shadow.size(); j++) {
        const Z3Vec2 &opt = outer_shadow[j];
        Z3Real weight = NewReal(&out, std::format("w{}_{}", i, j));
        // Require *strictly* greater than zero.
        AppendFormat(&out, "(assert (> {} 0.0))\n", weight.s);
        // Redundant with the total sum below, but harmless?
        AppendFormat(&out, "(assert (< {} 1.0))\n", weight.s);
        weights.push_back(weight);
        weighted_sum.push_back(opt * weight);
      }
      AppendFormat(&out, ";; inner v{}'s weights sum to 1\n", i);
      AppendFormat(&out, "(assert (= 1.0 {}))\n", Sum(weights).s);

      Z3Vec2 sum = Sum(weighted_sum);
      AppendFormat(&out, ";; inner v{} is in outer hull\n", i);
      AppendFormat(&out, "(assert (= {} {}))\n", ipt.x.s, sum.x.s);
      AppendFormat(&out, "(assert (= {} {}))\n", ipt.y.s, sum.y.s);
    }
  }

  AppendFormat(&out,
               "\n"
               ";; try saving progress\n"
               "(set-option :solver.cancel-backup-file \"zuperts-partial.z3\")\n");

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
