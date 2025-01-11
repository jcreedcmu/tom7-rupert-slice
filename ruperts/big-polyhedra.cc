
#include "big-polyhedra.h"

#include <cstdio>
#include <cstdlib>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "ansi.h"
#include "base/stringprintf.h"
#include "bignum/big-numbers.h"
#include "bignum/big-overloads.h"
#include "bignum/big.h"
#include "point-map.h"
#include "polyhedra.h"
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
    StringPrintf("%.17g and %.17g, with err %.17g", fv, gv, e);         \
  } while (0)


std::string VecString(const BigVec2 &v) {
  return StringPrintf(
      "(x: " ARED("%s") " ≅ %.17g; "
      "y: " AGREEN("%s") " ≅ %.17g)",
      v.x.ToString().c_str(), v.x.ToDouble(),
      v.y.ToString().c_str(), v.y.ToDouble());
}

std::string QuatString(const BigQuat &q) {
  return StringPrintf(
      "x: " ARED("%s") " ≅ %.17g\n"
      "y: " AGREEN("%s") " ≅ %.17g\n"
      "z: " ABLUE("%s") " ≅ %.17g\n"
      "w: " AYELLOW("%s") " ≅ %.17g\n",
      q.x.ToString().c_str(), q.x.ToDouble(),
      q.y.ToString().c_str(), q.y.ToDouble(),
      q.z.ToString().c_str(), q.z.ToDouble(),
      q.w.ToString().c_str(), q.w.ToDouble());
}


BigQuat Normalize(const BigQuat &q, int digits) {
  printf("Quat: %s\n", QuatString(q).c_str());
  BigRat norm_sq = q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w;
  printf("Norm^2: %s\n", norm_sq.ToString().c_str());
  CHECK(norm_sq != BigRat(0));
  BigRat norm = BigRat::Sqrt(norm_sq, BigNumbers::Digits(digits));
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


BigMesh2D Shadow(const BigPoly &poly) {
  std::vector<BigVec2> vertices;
  vertices.reserve(poly.vertices.size());
  for (const BigVec3 &v : poly.vertices) {
    vertices.emplace_back(v.x, v.y);
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

// XXX Buggy?
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
  BigQuat pqi = p * UnitInverse(q);

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
      StringPrintf("Maybe too close to use "
                   "double QuickHull to compute BigHull (old: %d = %s); "
                   "new: %d = %s\n",
                   old.value(),
                   VecString(dvs[old.value()]).c_str(),
                   i,
                   VecString(dv).c_str());
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

BigPoly MakeBigPolyFromVertices(std::vector<BigVec3> vertices) {
  // XXX check
  PointMap3<int> duplicates;

  std::vector<vec3> dvertices;
  dvertices.reserve(vertices.size());
  for (const BigVec3 &v : vertices) {
    #ifndef BIG_USE_GMP
    #error ToDouble only really works with GMP mode
    #endif
    double x = v.x.ToDouble();
    double y = v.y.ToDouble();
    double z = v.z.ToDouble();
    printf("%.17g,%.17g,%.17g\n", x, y, z);
    dvertices.emplace_back(x, y, z);
  }

  std::optional<Polyhedron> opoly =
    ConvexPolyhedronFromVertices(dvertices, "bigpoly");
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
  const BigRat epsilon = BigNumbers::Digits(digits);
  const BigRat phi = (BigRat(1) + BigRat::Sqrt(BigRat(5), epsilon)) / BigRat(2);
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
  return MakeBigPolyFromVertices(std::move(vertices));
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
  BigRat best_sqdist = DistanceSquared(mesh.vertices[0], pt);
  int best_idx = 0;

  for (int i = 1; i < mesh.vertices.size(); i++) {
    const BigVec2 &a = mesh.vertices[i];
    BigRat sqdist = DistanceSquared(a, pt);
    if (sqdist < best_sqdist) {
      best_sqdist = std::move(sqdist);
      best_idx = i;
    }
  }

  return best_idx;
}

