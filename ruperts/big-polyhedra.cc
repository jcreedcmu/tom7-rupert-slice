
#include "big-polyhedra.h"

#include <algorithm>
#include <bit>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <format>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "ansi.h"
#include "bignum/big-numbers.h"
#include "bignum/big-overloads.h"
#include "bignum/big.h"
#include "point-map.h"
#include "polyhedra.h"
#include "util.h"
#include "yocto_matht.h"

[[maybe_unused]]
static double IsNear(double a, double b) {
  return std::abs(a - b) < 0.0000001;
}

#define CHECK_NEAR(f, g) do {                                           \
  const double fv = (f);                                                \
  const double gv = (g);                                                \
  const double e = std::abs(fv - gv);                                   \
  CHECK(e < 0.0000001) << "Expected " << #f << " and " << #g <<         \
    " to be close, but got: " <<                                        \
    std::format("{:.17g} and {:.17g}, with err {:.17g}", fv, gv, e);    \
  } while (0)

std::string ColorRat(const char *ansi_color,
                     const BigRat &r) {
  std::string s = r.ToString();
  s = Util::Replace(s, "/", std::format(AWHITE("/") "{}", ansi_color));
  return std::format("{}{}" ANSI_RESET, ansi_color, s);
}

std::string VecString(const BigVec2 &v) {
  return std::format(
      "(x: {} ≅ {:.17g}; "
      "y: {} ≅ {:.17g})",
      ColorRat(ANSI_RED, v.x), v.x.ToDouble(),
      ColorRat(ANSI_GREEN, v.y), v.y.ToDouble());
}

std::string VecString(const BigVec3 &v) {
  return std::format(
      "(x: {} ≅ {:.17g}; "
      "y: {} ≅ {:.17g}; "
      "z: {} ≅ {:.17g})",
      ColorRat(ANSI_RED, v.x), v.x.ToDouble(),
      ColorRat(ANSI_GREEN, v.y), v.y.ToDouble(),
      ColorRat(ANSI_BLUE, v.z), v.y.ToDouble());
}

std::string FrameString(const BigFrame &f) {
  return std::format(
      "== frame ==\n"
      "x: {}\n"
      "y: {}\n"
      "z: {}\n"
      "o: {}\n",
      VecString(f.x),
      VecString(f.y),
      VecString(f.z),
      VecString(f.o));
}

// XXX use ColorRat
std::string QuatString(const BigQuat &q) {
  return std::format(
      "x: " ARED("{}") " ≅ {:.17g}\n"
      "y: " AGREEN("{}") " ≅ {:.17g}\n"
      "z: " ABLUE("{}") " ≅ {:.17g}\n"
      "w: " AYELLOW("{}") " ≅ {:.17g}\n",
      q.x.ToString(), q.x.ToDouble(),
      q.y.ToString(), q.y.ToDouble(),
      q.z.ToString(), q.z.ToDouble(),
      q.w.ToString(), q.w.ToDouble());
}

std::string PlainVecString(const BigVec2 &v) {
  return std::format(
      "BigVec2{{\n"
      "  x = {} ≅ {:.17g}\n"
      "  y = {} ≅ {:.17g}\n"
      "}}\n",
      v.x.ToString(), v.x.ToDouble(),
      v.y.ToString(), v.y.ToDouble());
}

std::string PlainQuatString(const BigQuat &q) {
  return std::format(
      "BigQuat{{\n"
      "  x = {} ≅ {:.17g}\n"
      "  y = {} ≅ {:.17g}\n"
      "  z = {} ≅ {:.17g}\n"
      "  w = {} ≅ {:.17g}\n"
      "}}\n",
      q.x.ToString(), q.x.ToDouble(),
      q.y.ToString(), q.y.ToDouble(),
      q.z.ToString(), q.z.ToDouble(),
      q.w.ToString(), q.w.ToDouble());
}

BigQuat MakeBigQuat(const quat4 &smallquat) {
  return BigQuat(BigRat::FromDouble(smallquat.x),
                 BigRat::FromDouble(smallquat.y),
                 BigRat::FromDouble(smallquat.z),
                 BigRat::FromDouble(smallquat.w));
}

BigQuat ApproxBigQuat(const quat4 &smallquat, int64_t max_denom) {
  return BigQuat(BigRat::ApproxDouble(smallquat.x, max_denom),
                 BigRat::ApproxDouble(smallquat.y, max_denom),
                 BigRat::ApproxDouble(smallquat.z, max_denom),
                 BigRat::ApproxDouble(smallquat.w, max_denom));
}

BigQuat Normalize(const BigQuat &q, int digits) {
  printf("Quat: %s\n", QuatString(q).c_str());
  BigRat norm_sq = q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w;
  printf("Norm^2: %s\n", norm_sq.ToString().c_str());
  CHECK(norm_sq != BigRat(0));
  BigRat norm = BigRat::Sqrt(norm_sq, BigNumbers::Pow10(digits));
  printf("Norm: %s\n", norm.ToString().c_str());
  CHECK(norm != BigRat(0));
  return BigQuat(q.x / norm, q.y / norm, q.z / norm, q.w / norm);
}

BigFrame RotationFrame(const BigQuat &v) {
  BigRat two(2);
  return {
    {
      v.w * v.w + v.x * v.x - v.y * v.y - v.z * v.z,
      (v.x * v.y + v.z * v.w) * two,
      (v.z * v.x - v.y * v.w) * two
    },
    {
      (v.x * v.y - v.z * v.w) * two,
      v.w * v.w - v.x * v.x + v.y * v.y - v.z * v.z,
      (v.y * v.z + v.x * v.w) * two
    },
    {
      (v.z * v.x + v.y * v.w) * two,
      (v.y * v.z - v.x * v.w) * two,
      v.w * v.w - v.x * v.x - v.y * v.y + v.z * v.z
    },
    {BigRat(0), BigRat(0), BigRat(0)}
  };
}


BigFrame NonUnitRotationFrame(const BigQuat &v) {
  BigRat two(2);

  BigRat xx = v.x * v.x;
  BigRat yy = v.y * v.y;
  BigRat zz = v.z * v.z;
  BigRat ww = v.w * v.w;

  // This normalization term is |v|^-2, which is
  // 1/(sqrt(x^2 + y^2 + z^2 + w^2))^2, so we can avoid
  // the square root! It's always multiplied by 2 in the
  // terms below, so we also do that here.
  BigRat two_s = two / (xx + yy + zz + ww);

  BigRat xy = v.x * v.y;
  BigRat zx = v.z * v.x;
  BigRat zw = v.z * v.w;
  BigRat yw = v.y * v.w;
  BigRat yz = v.y * v.z;
  BigRat xw = v.x * v.w;

  BigRat one(1);

  /*
    1 - 2s(y^2 + z^2)    ,   2s (x y - z w)      ,   2s (x z + y w)
    2s (x y + z w)       ,   1 - 2s(x^2 + z^2)   ,   2s (y z - x w)
    2s (x z - y w)       ,   2s (y z + x w)      ,   1 - 2s(x^2 + y^2)
  */

  // Note that the yocto matrix looks a bit different from this. On the
  // diagonal, it's doing the trick that since v^2 = 1, we can write
  // 1 + s(y^2 + z^2) as (x^2 + y^2 + z^2 + w^2) - 2(y^2 + z^2) and
  // then cancel terms.
  //
  // Also note: Yocto stores these in column-major format. Since it's
  // a rotation matrix I think the transpose is just the inverse
  // (which should be fine for these problems if we are consistent),
  // but be careful.

  return {
    // Left
    {one - two_s * (yy + zz),
     two_s * (xy + zw),
     two_s * (zx - yw)},
    // Middle
    {two_s * (xy - zw),
     one - two_s * (xx + zz),
     two_s * (yz + xw)},
    // Right
    {two_s * (zx + yw),
     two_s * (yz - xw),
     one - two_s * (xx + yy)}};
}

[[maybe_unused]]
static BigVec3 ReferenceViewPosFromNonUnitQuat(const BigQuat &q) {
  BigFrame frame = NonUnitRotationFrame(q);
  // We just apply the inverse rotation to (0, 0, 1).
  BigFrame iframe = InverseRigid(frame);
  return TransformPoint(iframe, BigVec3(BigRat(0), BigRat(0), BigRat(1)));
}

// Same as Reference version, but skipping work we don't need.
BigVec3 ViewPosFromNonUnitQuat(const BigQuat &q) {
  BigRat one(1);
  BigRat two(2);

  BigRat xx = q.x * q.x;
  BigRat yy = q.y * q.y;
  BigRat zz = q.z * q.z;
  BigRat ww = q.w * q.w;

  BigRat two_s = two / (xx + yy + zz + ww);

  BigRat zx = q.z * q.x;
  BigRat yw = q.y * q.w;
  BigRat yz = q.y * q.z;
  BigRat xw = q.x * q.w;

  return BigVec3(two_s * (zx - yw),
                 two_s * (yz + xw),
                 one - two_s * (xx + yy));
}

BigQuat QuaternionFromViewPos(const BigVec3 &v) {
  CHECK(v.x != BigRat(0) || v.y != BigRat(0));
  LOG(FATAL) << "This doesn't work?";

  const BigRat &qx = v.y;
  const BigRat qy = -v.x;
  const BigRat qz = BigRat(0);

  BigRat v_dot_v = (v.x * v.x) + (v.y * v.y) + (v.z * v.z);
  return BigQuat(qx, qy, qz, v_dot_v + v.z);
}

BigMesh2D Shadow(const BigPoly &poly) {
  std::vector<BigVec2> vertices;
  vertices.reserve(poly.vertices.size());
  for (const BigVec3 &v : poly.vertices) {
    vertices.emplace_back(v.x, v.y);
  }

  return BigMesh2D{.vertices = std::move(vertices), .faces = poly.faces};
}

BigMesh2D RotateAndProject(const BigFrame &f, const BigPoly &poly) {
  std::vector<BigVec2> vertices;
  vertices.reserve(poly.vertices.size());
  for (const BigVec3 &v : poly.vertices) {
    vertices.push_back(TransformAndProjectPoint(f, v));
  }
  return BigMesh2D{.vertices = std::move(vertices), .faces = poly.faces};
}

BigPoly Rotate(const BigFrame &f, const BigPoly &poly) {
  std::vector<BigVec3> vertices;
  vertices.reserve(poly.vertices.size());
  for (const BigVec3 &v : poly.vertices) {
    vertices.push_back(TransformPoint(f, v));
  }
  return BigPoly{.vertices = std::move(vertices), .faces = poly.faces};
}

BigMesh2D Translate(const BigVec2 &t, const BigMesh2D &m) {
  std::vector<BigVec2> vertices;
  vertices.reserve(m.vertices.size());
  for (const BigVec2 &v : m.vertices) {
    vertices.push_back(v + t);
  }

  return BigMesh2D{.vertices = std::move(vertices), .faces = m.faces};
}

BigPoly Rotate(const BigQuat &q, const BigPoly &poly) {
  std::vector<BigVec3> vertices;
  vertices.reserve(poly.vertices.size());
  for (const BigVec3 &v : poly.vertices) {
    vertices.push_back(RotatePoint(q, v));
  }
  return BigPoly{.vertices = std::move(vertices), .faces = poly.faces};
}

template<class GetPt>
inline static bool PointInPolygonT(const BigVec2 &point,
                                   int size,
                                   const GetPt &get_pt) {
  int winding_number = 0;
  for (int i = 0; i < size; i++) {
    const BigVec2 p0 = get_pt(i);
    const BigVec2 p1 = get_pt((i + 1) % size);

    // Check if the ray from the point to infinity intersects the edge
    if (point.y > std::min(p0.y, p1.y)) {
      if (point.y <= std::max(p0.y, p1.y)) {
        if (point.x <= std::max(p0.x, p1.x)) {
          if (p0.y != p1.y) {
            BigRat vt = (point.y - p0.y) / (p1.y - p0.y);
            if (point.x < p0.x + vt * (p1.x - p0.x)) {
              winding_number++;
            }
          }
        }
      }
    }
  }

  // Point is inside if the winding number is odd
  return !!(winding_number & 1);
}

bool PointInPolygon(const BigVec2 &point,
                    const std::vector<BigVec2> &vertices,
                    const std::vector<int> &polygon) {
  return PointInPolygonT(point, polygon.size(),
                         [&](int idx) {
                           return vertices[polygon[idx]];
                         });
}

bool PointInPolygon(const BigVec2 &point,
                    const std::vector<BigVec2> &polygon) {
  return PointInPolygonT(point, polygon.size(),
                         [&](int idx) {
                           return polygon[idx];
                         });
}


// XXX Buggy? maybe it was just because quaternion multiplication
// was wrong! ugh! test again.
BigVec3 RotatePoint(const BigQuat &q, const BigVec3 &v) {
  vec3 dv = SmallVec(v);
  quat4 dp{.x = dv.x, .y = dv.y, .z = dv.z, .w = 0.0};

  quat4 dq = SmallQuat(q);
  quat4 dqi = quat_conjugate(dq);
  quat4 dpqi = dp * dqi;
  quat4 r = dq * dpqi;

  // PERF: This can be simplified.
  //  - w coefficient for pure quaternion is zero.
  //  - q^-1 * p * q will be a pure quaternion for
  //    unit q, so we don't need to compute the w coefficient.
  //  - We are rotating lots of points with the same quat, so
  //    we should precompute.
  //  - The inverse is just the conjugate for unit q,
  //    so we can inline those too.
  //  - We actually want RotateAndProjectTo2D, since we're
  //    not even using the z coordinate.
  BigQuat p(v.x, v.y, v.z, BigRat(0));
  BigQuat pqi = p * Conjugate(q);

  CHECK_NEAR(dpqi.x, pqi.x.ToDouble());
  CHECK_NEAR(dpqi.y, pqi.y.ToDouble());
  CHECK_NEAR(dpqi.z, pqi.z.ToDouble());
  CHECK_NEAR(dpqi.w, pqi.w.ToDouble());

  BigQuat pp = q * pqi;
  BigVec3 ret(std::move(pp.x), std::move(pp.y), std::move(pp.z));

  CHECK_NEAR(r.x, ret.x.ToDouble());
  CHECK_NEAR(r.y, ret.y.ToDouble());
  CHECK_NEAR(r.z, ret.z.ToDouble());

  return ret;
}

std::vector<int> BigHull(const std::vector<BigVec2> &bigvs) {
  PointMap2<char> duplicates;
  std::vector<vec2> dvs;
  dvs.reserve(bigvs.size());
  for (int i = 0; i < bigvs.size(); i++) {
    const BigVec2 &v = bigvs[i];
    vec2 dv(v.x.ToDouble(), v.y.ToDouble());
    const std::optional<int> old = duplicates.Get(dv);
    CHECK(!old.has_value()) <<
      std::format("Maybe too close to use "
                  "double QuickHull to compute BigHull (old: {} = {}); "
                  "new: {} = {}\n",
                  old.value(),
                  VecString(dvs[old.value()]),
                  i,
                  VecString(dv));
    duplicates.Add(dv, i);
    dvs.emplace_back(dv);
  }

  return QuickHull(dvs);
}

Polyhedron SmallPoly(const BigPoly &big) {
  std::vector<vec3> vertices;
  vertices.reserve(big.vertices.size());
  for (int i = 0; i < big.vertices.size(); i++) {
    const BigVec3 &v = big.vertices[i];
    vertices.push_back(vec3{
        .x = v.x.ToDouble(),
        .y = v.y.ToDouble(),
        .z = v.z.ToDouble(),
      });
  }

  return Polyhedron{
    .vertices = std::move(vertices),
    .faces = big.faces,
    .name = "converted",
  };
}

Mesh2D SmallMesh(const BigMesh2D &big) {
  std::vector<vec2> vertices;
  vertices.reserve(big.vertices.size());
  for (int i = 0; i < big.vertices.size(); i++) {
    const BigVec2 &v = big.vertices[i];
    vertices.push_back(vec2{
        .x = v.x.ToDouble(),
        .y = v.y.ToDouble(),
      });
  }

  return Mesh2D{
    .vertices = std::move(vertices),
    .faces = big.faces,
  };
}

BigRat SignedAreaOfConvexPoly(const std::vector<BigVec2> &pts) {
  if (pts.size() < 3) return BigRat{0};
  BigRat area{0};
  // Iterate through the polygon vertices, using the shoelace formula.
  for (size_t i = 0; i < pts.size(); i++) {
    const BigVec2 &v0 = pts[i];
    const BigVec2 &v1 = pts[(i + 1) % pts.size()];
    area += v0.x * v1.y - v1.x * v0.y;
  }

  return area / BigRat(2);
}

// via https://en.wikipedia.org/wiki/Shoelace_formula
BigRat SignedAreaOfHull(const BigMesh2D &mesh,
                        const std::vector<int> &hull) {
  if (hull.size() < 3) return BigRat(0);
  BigRat area{0};
  // Iterate through the polygon vertices, using the shoelace formula.
  for (size_t i = 0; i < hull.size(); i++) {
    const BigVec2 &v0 = mesh.vertices[hull[i]];
    const BigVec2 &v1 = mesh.vertices[hull[(i + 1) % hull.size()]];
    area += v0.x * v1.y - v1.x * v0.y;
  }

  return area / BigRat(2);
}


BigPoly MakeBigPolyFromVertices(std::vector<BigVec3> vertices,
                                const char *name) {
  // XXX check
  PointMap3<int> duplicates;

  std::vector<vec3> dvertices;
  dvertices.reserve(vertices.size());
  for (const BigVec3 &v : vertices) {
    #ifndef BIG_USE_GMP
    LOG(FATAL) << "ToDouble only really works with GMP mode";
    #endif
    double x = v.x.ToDouble();
    double y = v.y.ToDouble();
    double z = v.z.ToDouble();
    // printf("%.17g,%.17g,%.17g\n", x, y, z);
    dvertices.emplace_back(x, y, z);
  }

  std::optional<Polyhedron> opoly =
    PolyhedronFromConvexVertices(dvertices, "bigpoly");
  CHECK(opoly.has_value());

  // Now match up the vertices with the original poly.
  const Polyhedron &poly = opoly.value();
  CHECK(poly.vertices.size() == dvertices.size());

  for (int i = 0; i < poly.vertices.size(); i++) {
    const BigVec3 &b = vertices[i];
    const vec3 &v = poly.vertices[i];
    double x = b.x.ToDouble();
    double y = b.y.ToDouble();
    double z = b.z.ToDouble();

    CHECK(std::abs(x - v.x) < 0.000001 &&
          std::abs(y - v.y) < 0.000001 &&
          std::abs(z - v.z) < 0.000001) << "Expected "
      "ConvexPolyhedronFromVertices to preserve the order of "
      "the vertices.";
  }

  BigPoly bpoly;
  bpoly.vertices = std::move(vertices);
  bpoly.faces = poly.faces;
  bpoly.name = name;
  return bpoly;
}

static void AddEvenPermutations(
    const BigRat &a, const BigRat &b, const BigRat &c,
    std::vector<BigVec3> *vertices) {
  // (a, b, c) - even
  // (b, c, a) - even
  // (c, a, b) - even

  vertices->emplace_back(a, b, c);

  if (a == b && b == c) return;

  vertices->emplace_back(b, c, a);
  vertices->emplace_back(c, a, b);
}

BigPoly BigRidode(int digits) {
  const BigInt inv_epsilon = BigNumbers::Pow10(digits);
  const BigRat phi = (BigRat(1) + BigRat::Sqrt(BigRat(5), inv_epsilon)) / BigRat(2);
  const BigRat phi_squared = phi * phi;
  const BigRat phi_cubed = phi_squared * phi;
  const BigRat zero = BigRat(0);
  const BigRat one = BigRat(1);
  const BigRat neg_one = BigRat(-1);

  std::vector<BigVec3> vertices;
  for (int b = 0b000; b < 0b1000; b++) {
    BigRat s1 = (b & 0b100) ? neg_one : one;
    BigRat s2 = (b & 0b010) ? neg_one : one;
    BigRat s3 = (b & 0b001) ? neg_one : one;

    // (±1, ±1, ±φ^3),
    // (±φ^2, ±φ, ±2φ),
    AddEvenPermutations(s1, s2, s3 * phi_cubed, &vertices);
    AddEvenPermutations(s1 * phi_squared,
                        s2 * phi,
                        s3 * BigRat(2) * phi, &vertices);
  }

  for (int b = 0b00; b < 0b100; b++) {
    BigRat s1 = (b & 0b10) ? neg_one : one;
    BigRat s2 = (b & 0b01) ? neg_one : one;
    // (±(2+φ), 0, ±φ^2),
    AddEvenPermutations(s1 * (BigRat(2) + phi),
                        zero,
                        s2 * phi_squared, &vertices);
  }

  CHECK(vertices.size() == 60) << vertices.size();
  return MakeBigPolyFromVertices(std::move(vertices), "rhombicosidodecahedron");
}

BigPoly BigDhexe(int digits) {
  std::vector<BigVec3> vertices;

  // Following https://dmccooey.com/polyhedra/DeltoidalHexecontahedron.txt

  const BigInt inv_epsilon = BigNumbers::Pow10(digits);

  const BigRat sqrt_5 = BigRat::Sqrt(BigRat(5), inv_epsilon);

  const BigRat ZZ = BigRat(0);
  const BigRat C0 = (BigRat(5) - sqrt_5) / BigRat(4);
  const BigRat C1 = (BigRat(15) + sqrt_5) / BigRat(22);
  const BigRat C2 = sqrt_5 / BigRat(2);
  const BigRat C3 = (BigRat(5) + sqrt_5) / BigRat(6);
  const BigRat C4 = (BigRat(5) + BigRat(4) * sqrt_5) / BigRat(11);
  const BigRat C5 = (BigRat(5) + sqrt_5) / BigRat(4);
  const BigRat C6 = (BigRat(5) + BigRat(3) * sqrt_5) / BigRat(6);
  const BigRat C7 = (BigRat(25) + BigRat(9) * sqrt_5) / BigRat(22);
  const BigRat C8 = sqrt_5;

  CHECK(std::abs(C0.ToDouble() - 0.690983005625052575897706582817) < 1.0e-10);
  CHECK(std::abs(C1.ToDouble() - 0.783457635340899531654962439488) < 1.0e-10);
  CHECK(std::abs(C2.ToDouble() - 1.11803398874989484820458683437 ) < 1.0e-10);
  CHECK(std::abs(C3.ToDouble() - 1.20601132958329828273486227812 ) < 1.0e-10);
  CHECK(std::abs(C4.ToDouble() - 1.26766108272719625323969951590 ) < 1.0e-10);
  CHECK(std::abs(C5.ToDouble() - 1.80901699437494742410229341718 ) < 1.0e-10);
  CHECK(std::abs(C6.ToDouble() - 1.95136732208322818153792016770 ) < 1.0e-10);
  CHECK(std::abs(C7.ToDouble() - 2.05111871806809578489466195539 ) < 1.0e-10);
  CHECK(std::abs(C8.ToDouble() - 2.23606797749978969640917366873 ) < 1.0e-10);

  vertices.emplace_back( ZZ,  ZZ,  C8);
  vertices.emplace_back( ZZ,  ZZ, -C8);
  vertices.emplace_back( C8,  ZZ,  ZZ);
  vertices.emplace_back(-C8,  ZZ,  ZZ);
  vertices.emplace_back( ZZ,  C8,  ZZ);
  vertices.emplace_back( ZZ, -C8,  ZZ);
  vertices.emplace_back( ZZ,  C1,  C7);
  vertices.emplace_back( ZZ,  C1, -C7);
  vertices.emplace_back( ZZ, -C1,  C7);
  vertices.emplace_back( ZZ, -C1, -C7);
  vertices.emplace_back( C7,  ZZ,  C1);
  vertices.emplace_back( C7,  ZZ, -C1);
  vertices.emplace_back(-C7,  ZZ,  C1);
  vertices.emplace_back(-C7,  ZZ, -C1);
  vertices.emplace_back( C1,  C7,  ZZ);
  vertices.emplace_back( C1, -C7,  ZZ);
  vertices.emplace_back(-C1,  C7,  ZZ);
  vertices.emplace_back(-C1, -C7,  ZZ);
  vertices.emplace_back( C3,  ZZ,  C6);
  vertices.emplace_back( C3,  ZZ, -C6);
  vertices.emplace_back(-C3,  ZZ,  C6);
  vertices.emplace_back(-C3,  ZZ, -C6);
  vertices.emplace_back( C6,  C3,  ZZ);
  vertices.emplace_back( C6, -C3,  ZZ);
  vertices.emplace_back(-C6,  C3,  ZZ);
  vertices.emplace_back(-C6, -C3,  ZZ);
  vertices.emplace_back( ZZ,  C6,  C3);
  vertices.emplace_back( ZZ,  C6, -C3);
  vertices.emplace_back( ZZ, -C6,  C3);
  vertices.emplace_back( ZZ, -C6, -C3);
  vertices.emplace_back( C0,  C2,  C5);
  vertices.emplace_back( C0,  C2, -C5);
  vertices.emplace_back( C0, -C2,  C5);
  vertices.emplace_back( C0, -C2, -C5);
  vertices.emplace_back(-C0,  C2,  C5);
  vertices.emplace_back(-C0,  C2, -C5);
  vertices.emplace_back(-C0, -C2,  C5);
  vertices.emplace_back(-C0, -C2, -C5);
  vertices.emplace_back( C5,  C0,  C2);
  vertices.emplace_back( C5,  C0, -C2);
  vertices.emplace_back( C5, -C0,  C2);
  vertices.emplace_back( C5, -C0, -C2);
  vertices.emplace_back(-C5,  C0,  C2);
  vertices.emplace_back(-C5,  C0, -C2);
  vertices.emplace_back(-C5, -C0,  C2);
  vertices.emplace_back(-C5, -C0, -C2);
  vertices.emplace_back( C2,  C5,  C0);
  vertices.emplace_back( C2,  C5, -C0);
  vertices.emplace_back( C2, -C5,  C0);
  vertices.emplace_back( C2, -C5, -C0);
  vertices.emplace_back(-C2,  C5,  C0);
  vertices.emplace_back(-C2,  C5, -C0);
  vertices.emplace_back(-C2, -C5,  C0);
  vertices.emplace_back(-C2, -C5, -C0);
  vertices.emplace_back( C4,  C4,  C4);
  vertices.emplace_back( C4,  C4, -C4);
  vertices.emplace_back( C4, -C4,  C4);
  vertices.emplace_back( C4, -C4, -C4);
  vertices.emplace_back(-C4,  C4,  C4);
  vertices.emplace_back(-C4,  C4, -C4);
  vertices.emplace_back(-C4, -C4,  C4);
  vertices.emplace_back(-C4, -C4, -C4);

  CHECK(vertices.size() == 62);
  return MakeBigPolyFromVertices(
      std::move(vertices), "deltoidalhexecontahedron");
}

BigPoly BigScube(int digits) {
  const BigInt inv_epsilon = BigNumbers::Pow10(digits);

  BigRat term = BigRat{3} * BigRat::Sqrt(BigRat{33}, inv_epsilon);

  const BigRat tribonacci =
    (BigRat{1} + BigRat::Cbrt(BigRat{19} + term, inv_epsilon) +
     BigRat::Cbrt(BigRat{19} - term, inv_epsilon)) / BigRat{3};

  const BigRat a = BigRat{1};
  const BigRat b = BigRat{1} / tribonacci;
  const BigRat c = tribonacci;

  std::vector<BigVec3> vertices;

  // 1 = negative, 0 = positive
  for (const uint8_t s : {0b100, 0b010, 0b001, 0b111}) {
    BigRat s1 = (s & 0b100) ? -BigRat(1) : BigRat(1);
    BigRat s2 = (s & 0b010) ? -BigRat(1) : BigRat(1);
    BigRat s3 = (s & 0b001) ? -BigRat(1) : BigRat(1);

    vertices.emplace_back(a * s1, b * s2, c * s3);
    vertices.emplace_back(b * s1, c * s2, a * s3);
    vertices.emplace_back(c * s1, a * s2, b * s3);
  }

  // 1 = negative, 0 = positive
  for (const uint8_t s : {0b011, 0b110, 0b101, 0b000}) {
    BigRat s1 = (s & 0b100) ? -BigRat(1) : BigRat(1);
    BigRat s2 = (s & 0b010) ? -BigRat(1) : BigRat(1);
    BigRat s3 = (s & 0b001) ? -BigRat(1) : BigRat(1);

    vertices.emplace_back(a * s1, c * s2, b * s3);
    vertices.emplace_back(b * s1, a * s2, c * s3);
    vertices.emplace_back(c * s1, b * s2, a * s3);
  }

  return MakeBigPolyFromVertices(
      std::move(vertices), "snubcube");
}


BigPoly BigPhexe(int digits) {
  const BigInt inv_epsilon = BigNumbers::Pow10(digits);
  const BigRat phi =
    (BigRat(1) + BigRat::Sqrt(BigRat(5), inv_epsilon)) / BigRat(2);

  std::vector<BigVec3> vertices;

  static BigRat x_term = BigRat::Sqrt(phi - BigRat(5, 27),
                                      inv_epsilon);
  static BigRat x =
    BigRat::Cbrt((phi + x_term) / BigRat(2), inv_epsilon) +
    BigRat::Cbrt((phi - x_term) / BigRat(2), inv_epsilon);

  auto Sqrt = [&inv_epsilon](const BigRat &xx) {
      return BigRat::Sqrt(xx, inv_epsilon);
    };

  // From
  // https://dmccooey.com/polyhedra/LpentagonalHexecontahedron.txt

  const BigRat xx = x * x;

  const BigRat C0 = phi * Sqrt(BigRat(3) - xx) / BigRat(2);
  const BigRat C1 =
    phi * Sqrt((x - BigRat{1} - (BigRat{1} / x)) * phi) / (BigRat(2) * x);
  const BigRat C2 = phi * Sqrt((x - BigRat{1} - (BigRat{1} / x)) * phi) / BigRat(2);
  const BigRat C3 = xx * phi * Sqrt(BigRat(3) - xx) / BigRat(2);
  const BigRat C4 = phi * Sqrt(BigRat{1} - x + (BigRat{1} + phi) / x) / BigRat(2);
  const BigRat C5 = Sqrt(x * (x + phi) + BigRat{1}) / (BigRat(2) * x);
  const BigRat C6 = Sqrt((x + BigRat(2)) * phi + BigRat(2)) / (BigRat(2) * x);
  const BigRat C7 =
    Sqrt(-xx * (BigRat(2) + phi) +
         x * (BigRat{1} + BigRat(3) * phi) + BigRat{4}) / BigRat(2);
  const BigRat C8 = (BigRat{1} + phi) *
    Sqrt(BigRat{1} + (BigRat{1} / x)) / (BigRat(2) * x);
  const BigRat C9 = Sqrt(BigRat(2) + BigRat(3) * phi -
                         BigRat(2) * x + (BigRat(3) / x)) / BigRat(2);
  const BigRat C10 =
    Sqrt(xx * (BigRat(392) + BigRat{225} * phi) + x *
         (BigRat{249} + BigRat{670} * phi) +
         (BigRat{470} + BigRat{157} * phi)) / BigRat(62);
  const BigRat C11 = phi * Sqrt(x * (x + phi) + BigRat{1}) / (BigRat(2) * x);
  const BigRat C12 = phi * Sqrt(xx + x + BigRat{1} + phi) / (BigRat(2) * x);
  const BigRat C13 =
      phi * Sqrt(xx + BigRat(2) * x * phi + BigRat(2)) / (BigRat(2) * x);
  const BigRat C14 = Sqrt(xx * (BigRat{1} + BigRat(2) * phi) - phi) / BigRat(2);
  const BigRat C15 = phi * Sqrt(xx + x) / BigRat(2);
  const BigRat C16 =
    (phi * phi * phi) * Sqrt(x * (x + phi) + BigRat{1}) / (BigRat(2) * xx);
  const BigRat C17 =
    Sqrt(xx * (BigRat{617} + BigRat(842) * phi) + x *
         (BigRat{919} + BigRat{1589} * phi) +
         (BigRat{627} + BigRat{784} * phi)) / BigRat(62);
  const BigRat C18 =
    (phi * phi) * Sqrt(x * (x + phi) + BigRat{1}) / (BigRat(2) * x);
  const BigRat C19 = phi * Sqrt(x * (x + phi) + BigRat{1}) / BigRat(2);

  // Check that the computed values are very close to their quoted
  // value.
  CHECK(std::abs(C0.ToDouble() - 0.192893711352359022108262546061) < 1e-10)
      << C0.ToString();
  CHECK(std::abs(C1.ToDouble() - 0.218483370127321224365534157111) < 1e-10)
      << C1.ToString();
  CHECK(std::abs(C2.ToDouble() - 0.374821658114562295266609516608) < 1e-10)
      << C2.ToString();
  CHECK(std::abs(C3.ToDouble() - 0.567715369466921317374872062669) < 1e-10)
      << C3.ToString();
  CHECK(std::abs(C4.ToDouble() - 0.728335176957191477360671629838) < 1e-10)
      << C4.ToString();
  CHECK(std::abs(C5.ToDouble() - 0.755467260516595579705585253517) < 1e-10)
      << C5.ToString();
  CHECK(std::abs(C6.ToDouble() - 0.824957552676275846265811111988) < 1e-10)
      << C6.ToString();
  CHECK(std::abs(C7.ToDouble() - 0.921228888309550499468934175898) < 1e-10)
      << C7.ToString();
  CHECK(std::abs(C8.ToDouble() - 0.959987701391583803994339068107) < 1e-10)
      << C8.ToString();
  CHECK(std::abs(C9.ToDouble() - 1.13706613386050418840961998424) < 1e-10)
      << C9.ToString();
  CHECK(std::abs(C10.ToDouble() - 1.16712343647533397917215468549) < 1e-10)
      << C10.ToString();
  CHECK(std::abs(C11.ToDouble() - 1.22237170490362309266282747264) < 1e-10)
      << C11.ToString();
  CHECK(std::abs(C12.ToDouble() - 1.27209628257581214613814794036) < 1e-10)
      << C12.ToString();
  CHECK(std::abs(C13.ToDouble() - 1.52770307085850512136921113078) < 1e-10)
      << C13.ToString();
  CHECK(std::abs(C14.ToDouble() - 1.64691794069037444140475745697) < 1e-10)
      << C14.ToString();
  CHECK(std::abs(C15.ToDouble() - 1.74618644098582634573474528789) < 1e-10)
      << C15.ToString();
  CHECK(std::abs(C16.ToDouble() - 1.86540131081769566577029161408) < 1e-10)
      << C16.ToString();
  CHECK(std::abs(C17.ToDouble() - 1.88844538928366915418351670356) < 1e-10)
      << C17.ToString();
  CHECK(std::abs(C18.ToDouble() - 1.97783896542021867236841272616) < 1e-10)
      << C18.ToString();
  CHECK(std::abs(C19.ToDouble() - 2.097053835252087992403959052348) < 1e-10)
      << C19.ToString();

  BigRat zz{0};
  vertices.emplace_back( -C0,  -C1, -C19);
  vertices.emplace_back( -C0,   C1,  C19);
  vertices.emplace_back(  C0,   C1, -C19);
  vertices.emplace_back(  C0,  -C1,  C19);
  vertices.emplace_back(-C19,  -C0,  -C1);
  vertices.emplace_back(-C19,   C0,   C1);
  vertices.emplace_back( C19,   C0,  -C1);
  vertices.emplace_back( C19,  -C0,   C1);
  vertices.emplace_back( -C1, -C19,  -C0);
  vertices.emplace_back( -C1,  C19,   C0);
  vertices.emplace_back(  C1,  C19,  -C0);
  vertices.emplace_back(  C1, -C19,   C0);
  vertices.emplace_back(  zz,  -C5, -C18);
  vertices.emplace_back(  zz,  -C5,  C18);
  vertices.emplace_back(  zz,   C5, -C18);
  vertices.emplace_back(  zz,   C5,  C18);
  vertices.emplace_back(-C18,   zz,  -C5);
  vertices.emplace_back(-C18,   zz,   C5);
  vertices.emplace_back( C18,   zz,  -C5);
  vertices.emplace_back( C18,   zz,   C5);
  vertices.emplace_back( -C5, -C18,   zz);
  vertices.emplace_back( -C5,  C18,   zz);
  vertices.emplace_back(  C5, -C18,   zz);
  vertices.emplace_back(  C5,  C18,   zz);
  vertices.emplace_back(-C10,   zz, -C17);
  vertices.emplace_back(-C10,   zz,  C17);
  vertices.emplace_back( C10,   zz, -C17);
  vertices.emplace_back( C10,   zz,  C17);
  vertices.emplace_back(-C17, -C10,   zz);
  vertices.emplace_back(-C17,  C10,   zz);
  vertices.emplace_back( C17, -C10,   zz);
  vertices.emplace_back( C17,  C10,   zz);
  vertices.emplace_back(  zz, -C17, -C10);
  vertices.emplace_back(  zz, -C17,  C10);
  vertices.emplace_back(  zz,  C17, -C10);
  vertices.emplace_back(  zz,  C17,  C10);
  vertices.emplace_back( -C3,   C6, -C16);
  vertices.emplace_back( -C3,  -C6,  C16);
  vertices.emplace_back(  C3,  -C6, -C16);
  vertices.emplace_back(  C3,   C6,  C16);
  vertices.emplace_back(-C16,   C3,  -C6);
  vertices.emplace_back(-C16,  -C3,   C6);
  vertices.emplace_back( C16,  -C3,  -C6);
  vertices.emplace_back( C16,   C3,   C6);
  vertices.emplace_back( -C6,  C16,  -C3);
  vertices.emplace_back( -C6, -C16,   C3);
  vertices.emplace_back(  C6, -C16,  -C3);
  vertices.emplace_back(  C6,  C16,   C3);
  vertices.emplace_back( -C2,  -C9, -C15);
  vertices.emplace_back( -C2,   C9,  C15);
  vertices.emplace_back(  C2,   C9, -C15);
  vertices.emplace_back(  C2,  -C9,  C15);
  vertices.emplace_back(-C15,  -C2,  -C9);
  vertices.emplace_back(-C15,   C2,   C9);
  vertices.emplace_back( C15,   C2,  -C9);
  vertices.emplace_back( C15,  -C2,   C9);
  vertices.emplace_back( -C9, -C15,  -C2);
  vertices.emplace_back( -C9,  C15,   C2);
  vertices.emplace_back(  C9,  C15,  -C2);
  vertices.emplace_back(  C9, -C15,   C2);
  vertices.emplace_back( -C7,  -C8, -C14);
  vertices.emplace_back( -C7,   C8,  C14);
  vertices.emplace_back(  C7,   C8, -C14);
  vertices.emplace_back(  C7,  -C8,  C14);
  vertices.emplace_back(-C14,  -C7,  -C8);
  vertices.emplace_back(-C14,   C7,   C8);
  vertices.emplace_back( C14,   C7,  -C8);
  vertices.emplace_back( C14,  -C7,   C8);
  vertices.emplace_back( -C8, -C14,  -C7);
  vertices.emplace_back( -C8,  C14,   C7);
  vertices.emplace_back(  C8,  C14,  -C7);
  vertices.emplace_back(  C8, -C14,   C7);
  vertices.emplace_back( -C4,  C12, -C13);
  vertices.emplace_back( -C4, -C12,  C13);
  vertices.emplace_back(  C4, -C12, -C13);
  vertices.emplace_back(  C4,  C12,  C13);
  vertices.emplace_back(-C13,   C4, -C12);
  vertices.emplace_back(-C13,  -C4,  C12);
  vertices.emplace_back( C13,  -C4, -C12);
  vertices.emplace_back( C13,   C4,  C12);
  vertices.emplace_back(-C12,  C13,  -C4);
  vertices.emplace_back(-C12, -C13,   C4);
  vertices.emplace_back( C12, -C13,  -C4);
  vertices.emplace_back( C12,  C13,   C4);
  vertices.emplace_back(-C11, -C11, -C11);
  vertices.emplace_back(-C11, -C11,  C11);
  vertices.emplace_back(-C11,  C11, -C11);
  vertices.emplace_back(-C11,  C11,  C11);
  vertices.emplace_back( C11, -C11, -C11);
  vertices.emplace_back( C11, -C11,  C11);
  vertices.emplace_back( C11,  C11, -C11);
  vertices.emplace_back( C11,  C11,  C11);

  CHECK(vertices.size() == 92);
  return MakeBigPolyFromVertices(
      std::move(vertices), "pentagonalhexecontahedron");
}

BigPoly BigSdode(int digits) {
  const BigInt inv_epsilon = BigNumbers::Pow10(digits);

  auto Sqrt = [&inv_epsilon](const BigRat &xx) {
      return BigRat::Sqrt(xx, inv_epsilon);
    };

  const BigRat phi =
    (BigRat(1) + Sqrt(BigRat(5))) / BigRat(2);
  const BigRat phi_squared = phi * phi;

  const BigRat term = Sqrt(phi - BigRat(5, 27));
  const BigRat xi =
    BigRat::Cbrt((phi + term) / BigRat(2), inv_epsilon) +
    BigRat::Cbrt((phi - term) / BigRat(2), inv_epsilon);

  const BigRat xi_squared = xi * xi;
  const BigRat inv_xi = BigRat(1) / xi;

  std::vector<BigVec3> vertices;
  auto AddEvenPermutations = [&](const BigRat &a,
                                 const BigRat &b,
                                 const BigRat &c) {

      // (a, b, c) - even
      // (b, c, a) - even
      // (c, a, b) - even

      vertices.emplace_back(a, b, c);

      if (a == b && b == c) return;

      vertices.emplace_back(b, c, a);
      vertices.emplace_back(c, a, b);
    };

  for (uint8_t bits = 0b000; bits < 0b1000; bits++) {
    BigRat s1 = (bits & 0b100) ? BigRat(-1) : BigRat(1);
    BigRat s2 = (bits & 0b010) ? BigRat(-1) : BigRat(1);
    BigRat s3 = (bits & 0b001) ? BigRat(-1) : BigRat(1);

    if ((std::popcount<uint8_t>(bits) & 1) == 1) {
      // Odd number of negative signs.

      AddEvenPermutations(
          s1 * phi * Sqrt(phi * (xi - BigRat{1} - inv_xi)),
          s2 * xi * phi * Sqrt(BigRat{3} - xi_squared),
          s3 * phi * Sqrt(xi * (xi + phi) + BigRat{1}));
      AddEvenPermutations(
          s1 * phi * Sqrt(BigRat{3} - xi_squared),
          s2 * xi * phi * Sqrt(BigRat{1} - xi + (BigRat{1} + phi) / xi),
          s3 * phi * Sqrt(xi * (xi + BigRat{1})));
      AddEvenPermutations(
          s1 * xi_squared * phi * Sqrt(phi * (xi - BigRat{1} - inv_xi)),
          s2 * phi * Sqrt(xi + BigRat{1} - phi),
          s3 * Sqrt(xi_squared * (BigRat{1} + BigRat{2} * phi) - phi));

    } else {
      // Even number of negative signs.

      AddEvenPermutations(
          s1 * xi_squared * phi * Sqrt(BigRat{3} - xi_squared),
          s2 * xi * phi * Sqrt(phi * (xi - BigRat{1} - inv_xi)),
          s3 * phi_squared * inv_xi * Sqrt(xi * (xi + phi) + BigRat{1}));

      AddEvenPermutations(
          s1 * Sqrt(phi * (xi + BigRat{2}) + BigRat{2}),
          s2 * phi * Sqrt(BigRat{1} - xi + (BigRat{1} + phi) / xi),
          s3 * xi * Sqrt(xi * (BigRat{1} + phi) - phi));
    }
  }

  return MakeBigPolyFromVertices(
      std::move(vertices), "snubdodecahedron");
}


BigPoly BigTriac(int digits) {
  std::vector<BigVec3> vertices;

  const BigInt inv_epsilon = BigNumbers::Pow10(digits);

  auto Sqrt = [&inv_epsilon](const BigRat &xx) {
      return BigRat::Sqrt(xx, inv_epsilon);
    };

  const BigRat phi = (BigRat(1) + Sqrt(BigRat(5))) / BigRat(2);
  const BigRat sqrtp2 = Sqrt(phi + BigRat(2));
  const BigRat r = BigRat(5) / (BigRat(3) * phi * sqrtp2);
  const BigRat s = ((BigRat(7) * phi - BigRat(6)) * sqrtp2) / BigRat(11);

  // They should be close to the quoted values.
  CHECK(std::abs(r.ToDouble() - 0.5415328270548438) < 1e-12);
  CHECK(std::abs(s.ToDouble() - 0.9210096876986302) < 1e-12);

  // cube
  for (uint8_t bits = 0b000; bits < 0b1000; bits++) {
    BigRat s1 = BigRat((bits & 0b100) ? -1 : +1);
    BigRat s2 = BigRat((bits & 0b010) ? -1 : +1);
    BigRat s3 = BigRat((bits & 0b001) ? -1 : +1);
    vertices.emplace_back(s1 * r, s2 * r, s3 * r);
  }

  for (BigRat sign : {BigRat(-1), BigRat(1)}) {
    vertices.emplace_back(sign * s, BigRat(0), BigRat(0));
    vertices.emplace_back(BigRat(0), sign * s, BigRat(0));
    vertices.emplace_back(BigRat(0), BigRat(0), sign * s);
  }

  for (uint8_t bits = 0b00; bits < 0b100; bits++) {
    BigRat s1 = BigRat((bits & 0b10) ? -1 : +1);
    BigRat s2 = BigRat((bits & 0b01) ? -1 : +1);
    AddEvenPermutations(BigRat(0),
                        s1 / sqrtp2,
                        s2 * phi / sqrtp2,
                        &vertices);

    AddEvenPermutations(BigRat(0), s1 * phi * r, s2 * r / phi,
                        &vertices);
  }

  for (uint8_t bits = 0b000; bits < 0b1000; bits++) {
    BigRat s1 = BigRat((bits & 0b100) ? -1 : +1);
    BigRat s2 = BigRat((bits & 0b010) ? -1 : +1);
    BigRat s3 = BigRat((bits & 0b001) ? -1 : +1);
    AddEvenPermutations(
        s1 * s * phi / BigRat(2),
        s2 * s / BigRat(2),
        s3 * s / (BigRat(2) * phi),
        &vertices);
  }

  CHECK(vertices.size() == 62);
  return MakeBigPolyFromVertices(
      std::move(vertices), "disdyakistriacontahedron");
}

BigPoly BigCube(int digits) {
  //                  +y
  //      a------b     | +z
  //     /|     /|     |/
  //    / |    / |     0--- +x
  //   d------c  |
  //   |  |   |  |
  //   |  e---|--f
  //   | /    | /
  //   |/     |/
  //   h------g

  std::vector<BigVec3> vertices;
  auto AddVertex = [&vertices](int x, int y, int z) {
      int idx = (int)vertices.size();
      vertices.emplace_back(BigVec3{BigRat(x), BigRat(y), BigRat(z)});
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
  return BigPoly{
    .vertices = std::move(vertices),
    .faces = faces,
    .name = "cube",
  };
}

BigPoly BigTetra(int digits_unused) {
  std::vector<BigVec3> vertices{
    BigVec3{BigRat(1), BigRat(1), BigRat(1)},
    BigVec3{BigRat(1), BigRat(-1), BigRat(-1)},
    BigVec3{BigRat(-1), BigRat(1), BigRat(-1)},
    BigVec3{BigRat(-1), BigRat(-1), BigRat(1)},
  };

  return MakeBigPolyFromVertices(std::move(vertices), "tetrahedron");
}


// Is pt strictly within the triangle a-b-c? Works with both winding orders.
bool InTriangle(const BigVec2 &a, const BigVec2 &b, const BigVec2 &c,
                const BigVec2 &pt) {
  // The idea behind this test is that for each edge, we check
  // to see if the test point is on the same side as a reference
  // point, which is the third point of the triangle.
  auto SameSide = [](const BigVec2 &u, const BigVec2 &v,
                     const BigVec2 &p1, const BigVec2 &p2) {
      BigVec2 edge = v - u;
      BigRat c1 = cross(edge, p1 - u);
      BigRat c2 = cross(edge, p2 - u);

      int s1 = BigRat::Sign(c1);
      int s2 = BigRat::Sign(c2);

      // Note that this excludes the edge itself.
      return s1 != 0 && s2 != 0 && s1 == s2;
    };

  return SameSide(a, b, c, pt) &&
    SameSide(b, c, a, pt) &&
    SameSide(c, a, b, pt);
}

bool InMesh(const BigMesh2D &mesh, const BigVec2 &pt) {
  // using the triangulation
  for (const auto &[ai, bi, ci] : mesh.faces->triangulation) {
    const BigVec2 &a = mesh.vertices[ai];
    const BigVec2 &b = mesh.vertices[bi];
    const BigVec2 &c = mesh.vertices[ci];

    if (InTriangle(a, b, c, pt)) return true;
  }

  return false;
}

std::optional<std::tuple<int, int, int>>
InMeshExhaustive(const BigMesh2D &mesh, const BigVec2 &pt) {

  for (int ai = 0; ai < mesh.vertices.size(); ai++) {
    const BigVec2 &a = mesh.vertices[ai];
    for (int bi = 0; bi < ai; bi++) {
      const BigVec2 &b = mesh.vertices[bi];
      if (b == a) continue;
      for (int ci = 0; ci < bi; ci++) {
        const BigVec2 &c = mesh.vertices[ci];
        if (c == b || c == a) continue;

        // XXX check for degenerate triangles.
        if (InTriangle(a, b, c, pt)) return std::make_tuple(ai, bi, ci);
      }
    }
  }

  return std::nullopt;
}

int GetClosestPoint(const BigMesh2D &mesh, const BigVec2 &pt) {
  CHECK(!mesh.vertices.empty());
  BigRat best_sqdist = distance_squared(mesh.vertices[0], pt);
  int best_idx = 0;

  for (int i = 1; i < mesh.vertices.size(); i++) {
    const BigVec2 &a = mesh.vertices[i];
    BigRat sqdist = distance_squared(a, pt);
    if (sqdist < best_sqdist) {
      best_sqdist = std::move(sqdist);
      best_idx = i;
    }
  }

  return best_idx;
}

std::pair<int, BigRat> GetClosestPoint(const std::vector<BigVec2> &vertices,
                                       const std::vector<int> &hull,
                                       const BigVec2 &pt) {
  CHECK(!hull.empty());
  BigRat best_sqdist = distance_squared(vertices[hull[0]], pt);
  int best_idx = hull[0];

  for (int i = 1; i < hull.size(); i++) {
    int idx = hull[i];
    const BigVec2 &a = vertices[idx];
    BigRat sqdist = distance_squared(a, pt);
    if (sqdist < best_sqdist) {
      best_sqdist = std::move(sqdist);
      best_idx = idx;
    }
  }

  return std::make_pair(best_idx, std::move(best_sqdist));
}


// The QuickHull implementation below and its helper routines are based on code
// by Miguel Vieira (see LICENSES) although I have heavily modified it; see
// the floating point version in polyhedra.cc for more info.

// Returns the index of the farthest point from segment (a, b).
// Requires that all points are to the left of the segment (a,b) (or colinear).
static int GetFarthest(const BigVec2 &a, const BigVec2 &b,
                       const std::vector<BigVec2> &v,
                       const std::vector<int> &pts) {
  CHECK(!pts.empty());
  const BigRat dx = b.x - a.x;
  const BigRat dy = b.y - a.y;
  auto SqDist = [&](const BigVec2 &p) -> BigRat {
      return dx * (a.y - p.y) - dy * (a.x - p.x);
    };

  int best_idx = pts[0];
  BigRat best_dist = SqDist(v[best_idx]);

  for (int i = 1; i < pts.size(); i++) {
    int p = pts[i];
    BigRat d = SqDist(v[p]);
    if (d < best_dist) {
      best_idx = p;
      best_dist = std::move(d);
    }
  }

  return best_idx;
}


// The z-value of the cross product of segments
// (a, b) and (a, c). Positive means c is ccw (to the left)
// from (a, b), negative cw. Zero means it's colinear.
enum Orientation {
  CCW, COLINEAR, CW,
};
static Orientation GetOrientation(
    const BigVec2 &a, const BigVec2 &b, const BigVec2 &c) {
  int s = BigRat::Sign(cross(b - a, c - a));
  if (s < 0) return CW;
  if (s > 0) return CCW;
  return COLINEAR;
}

// Recursive call of the quickhull algorithm.
static void QuickHullRec(const std::vector<BigVec2> &vertices,
                         const std::vector<int> &pts,
                         int a, int b,
                         std::vector<int> *hull) {
  static constexpr bool SELF_CHECK = false;
  static constexpr int VERBOSE = 0;

  if (pts.empty()) {
    return;
  }

  if (SELF_CHECK) {
    for (int x : pts) {
      CHECK(x != a && x != b) << x << " candidate points should "
        "not include the endpoints of the recursed upon segment.";
      for (int y : *hull) {
        CHECK(x != y) << x << " is already in the hull!";
      }
    }
  }

  const BigVec2 &aa = vertices[a];
  const BigVec2 &bb = vertices[b];

  if (SELF_CHECK) {
    for (int x : pts) {
      const BigVec2 &xx = vertices[x];
      CHECK(aa == xx || bb == xx ||
            GetOrientation(aa, bb, xx) == CCW);
    }
  }

  int f = GetFarthest(aa, bb, vertices, pts);
  const BigVec2 &ff = vertices[f];
  if (VERBOSE) printf("Farthest is %d (%s)\n", f, VecString(ff).c_str());

  // Collect points to the left of segment (a, f) and to the left
  // of segment f, b (which we call "right"). A point cannot be in both
  // sets, because that would require it to be farther away than f, but f
  // is maximal.
  //
  //             f
  //       left / \   right
  //      set  /   \   set
  //          a     b
  //
  std::vector<int> left, right;
  for (int i : pts) {
    const BigVec2 &ii = vertices[i];
    // In the presence of exact duplicates for one of the endpoints,
    // we need to filter them out here or else we can end up in
    // infinite loops. Removing duplicates does not affect the hull.
    if (ii == aa || ii == bb || ii == ff) continue;

    if (GetOrientation(aa, ff, ii) == CCW) {
      left.push_back(i);
    } else if (GetOrientation(ff, bb, ii) == CCW) {
      right.push_back(i);
    }
  }
  if (VERBOSE) printf("%d left, %d right vertices.\n",
                      (int)left.size(), (int)right.size());
  QuickHullRec(vertices, left, a, f, hull);

  // Add f to the hull
  hull->push_back(f);

  if (VERBOSE) printf("%d right vertices.\n", (int)right.size());
  QuickHullRec(vertices, right, f, b, hull);
}

// QuickHull algorithm.
// https://en.wikipedia.org/wiki/QuickHull
std::vector<int> BigQuickHull(const std::vector<BigVec2> &vertices) {
  std::vector<int> hull;
  if (vertices.empty()) return {};
  if (vertices.size() == 1) return {0};
  if (vertices.size() == 2) return {0, 1};

  // Returns true if a is lexicographically before b.
  auto LeftOf = [](const BigVec2 &a, const BigVec2 &b) -> bool {
      return (a.x < b.x || (a.x == b.x && a.y < b.y));
    };

  // Get the leftmost (a) and rightmost (b) points.
  int a = 0, b = 0;
  for (int i = 1; i < (int)vertices.size(); i++) {
    if (LeftOf(vertices[i], vertices[a])) a = i;
    if (LeftOf(vertices[b], vertices[i])) b = i;
  }

  CHECK(a != b);

  // Split the points on either side of segment (a, b).
  std::vector<int> left, right;
  for (int i = 0; i < (int)vertices.size(); i++) {
    if (i != a && i != b) {
      Orientation side = GetOrientation(vertices[a], vertices[b], vertices[i]);
      if (side == CCW) left.push_back(i);
      else if (side == CW) right.push_back(i);
      // Ignore if colinear.
    }
  }

  // Be careful to add points to the hull
  // in the correct order. Add our leftmost point.
  hull.push_back(a);

  // Add hull points from the left (top)
  QuickHullRec(vertices, left, a, b, &hull);

  // Add our rightmost point
  hull.push_back(b);

  // Add hull points from the right (bottom)
  QuickHullRec(vertices, right, b, a, &hull);

  return hull;
}

bool InHull(const std::vector<BigVec2> &vertices,
            const std::vector<int> &hull,
            const BigVec2 &pt) {
  // No area.
  if (hull.size() < 3) return false;

  std::optional<Orientation> same_side;
  for (int i = 0; i < hull.size(); i++) {
    const BigVec2 &a = vertices[hull[i]];
    const BigVec2 &b = vertices[hull[(i + 1) % hull.size()]];

    Orientation side = GetOrientation(a, b, pt);
    if (side == COLINEAR) return false;
    if (!same_side.has_value()) {
      same_side = side;
    } else {
      if (same_side.value() != side) return false;
    }
  }
  // Always on the same side.
  return true;
}

// Return the closest point (to x,y) on the given line segment.
// It may be one of the endpoints.
BigRat SquaredDistanceToClosestPointOnSegment(
    // Line segment
    const BigVec2 &v0,
    const BigVec2 &v1,
    // Point to test
    const BigVec2 &pt) {
  const BigRat zero(0);
  const BigRat one(1);

  BigVec2 v = v1 - v0;

  BigRat sqlen = length_squared(v);
  if (sqlen == zero) {
    // Degenerate case where line segment is just a point,
    // so there is only one choice.
    return distance_squared(pt, v0);
  }

  BigVec2 w = pt - v0;
  BigRat t = BigRat::Max(zero, BigRat::Min(one, dot(v, w) / sqlen));

  BigVec2 closest = v0 + (v * t);
  return distance_squared(pt, closest);
}


BigRat SquaredDistanceToHull(const std::vector<BigVec2> &vertices,
                             const std::vector<int> &hull,
                             const BigVec2 &pt) {
  std::optional<BigRat> min_sqlen;
  for (int i = 0; i < hull.size(); i++) {
    const BigVec2 &a = vertices[hull[i]];
    const BigVec2 &b = vertices[hull[(i + 1) % hull.size()]];

    BigRat sqlen = SquaredDistanceToClosestPointOnSegment(a, b, pt);
    if (!min_sqlen.has_value() || sqlen < min_sqlen.value()) {
      min_sqlen = {std::move(sqlen)};
    }
  }

  CHECK(min_sqlen.has_value()) << "empty hull?";
  return min_sqlen.value();
}

#if 0
std::pair<BigQuat, BigVec3> UnpackFrame(const BigFrame &f) {

  auto Rotation = [](const BigFrame &a) -> BigMat3 {
      return {a.x, a.y, a.z};
    };


  const BigMat3 m = Rotation(f);

  double w = sqrt(std::max(0.0, 1.0 + m[0][0] + m[1][1] + m[2][2])) * 0.5;
  double x = sqrt(std::max(0.0, 1.0 + m[0][0] - m[1][1] - m[2][2])) * 0.5;
  double y = sqrt(std::max(0.0, 1.0 - m[0][0] + m[1][1] - m[2][2])) * 0.5;
  double z = sqrt(std::max(0.0, 1.0 - m[0][0] - m[1][1] + m[2][2])) * 0.5;

  if (m[1][2] - m[2][1] < 0.0) x = -x;
  if (m[2][0] - m[0][2] < 0.0) y = -y;
  if (m[0][1] - m[1][0] < 0.0) z = -z;

  return std::make_pair(normalize(quat4(x, y, z, w)),
                        yocto::translation(f));
}
#endif

bool ValidateSolution(const BigPoly &poly,
                      const frame3 &outer_frame,
                      const frame3 &inner_frame,
                      int digits) {

  const auto &[douter_rot, dotrans] =
    UnpackFrame(outer_frame);
  const auto &[dinner_rot, ditrans] =
    UnpackFrame(inner_frame);

  CHECK(dotrans.x == 0.0 &&
        dotrans.y == 0.0 &&
        dotrans.z == 0.0) << "This can be handled, but we expect "
    "exact zero translation for the outer frame.";

  // z component does not matter, because we project along z.
  BigVec2 itrans(BigRat::FromDouble(ditrans.x),
                 BigRat::FromDouble(ditrans.y));

  BigQuat oq(BigRat::FromDouble(douter_rot.x),
             BigRat::FromDouble(douter_rot.y),
             BigRat::FromDouble(douter_rot.z),
             BigRat::FromDouble(douter_rot.w));

  BigQuat iq(BigRat::FromDouble(dinner_rot.x),
             BigRat::FromDouble(dinner_rot.y),
             BigRat::FromDouble(dinner_rot.z),
             BigRat::FromDouble(dinner_rot.w));

  BigFrame big_outer_frame = NonUnitRotationFrame(oq);
  BigFrame big_inner_frame = NonUnitRotationFrame(iq);

  BigPoly outer = Rotate(big_outer_frame, poly);
  BigPoly inner = Rotate(big_inner_frame, poly);

  BigMesh2D souter = Shadow(outer);
  BigMesh2D sinner = Translate(itrans, Shadow(inner));

  for (int i = 0; i < sinner.vertices.size(); i++) {
    const BigVec2 &v = sinner.vertices[i];
    const std::optional<std::tuple<int, int, int>> triangle =
      InMeshExhaustive(souter, v);
    bool in = triangle.has_value();
    if (!in) return false;
  }

  return true;
}

BigVec3 ScaleToMakeIntegral(const BigVec3 &v) {
  const auto &[xn, xd] = v.x.Parts();
  const auto &[yn, yd] = v.y.Parts();
  const auto &[zn, zd] = v.z.Parts();

  // Multiply each by the denominators.
  // In the first, for example, xd and 1/xd cancel out.
  BigInt x = xn * yd * zd;
  BigInt y = yn * xd * zd;
  BigInt z = zn * xd * yd;

  // Now we can simplify by the GCD.
  BigInt d = BigInt::GCD(x, BigInt::GCD(y, z));
  return BigVec3(BigRat(x, d), BigRat(y, d), BigRat(z, d));
}
