
#include "ansi.h"

#include <cstdint>
#include <cstdlib>
#include <deque>
#include <cstdio>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "base/logging.h"
#include "bignum/big-overloads.h"
#include "bignum/big.h"
#include "base/stringprintf.h"
#include "point-map.h"
#include "polyhedra.h"
#include "solutions.h"
#include "rendering.h"

// TODO: To cc-lib bignum

// We use rational numbers with at least this many digits of
// precision.
static constexpr int DIGITS = 100;

struct BigVec3 {
  BigVec3(BigRat x, BigRat y, BigRat z) :
    x(std::move(x)), y(std::move(y)), z(std::move(z)) {}
  BigRat x = BigRat(0), y = BigRat(0), z = BigRat(0);
};

inline BigVec3 operator +(const BigVec3 &a, const BigVec3 &b) {
  return BigVec3(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline BigVec3 operator -(const BigVec3 &a, const BigVec3 &b) {
  return BigVec3(a.x - b.x, a.y - b.y, a.z - b.z);
}

struct BigVec2 {
  BigVec2(BigRat x, BigRat y) :
    x(std::move(x)), y(std::move(y)) {}
  BigRat x = BigRat(0), y = BigRat(0);
};

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


// Represents the tail of the Taylor series expansion of
// arctan(1/x).
struct ArctanSeries {
  // for 1/x.
  ArctanSeries(int x) : xx(x * x), xpow(x) {}

  // Return the 0-based index of the current term ( i.e. the term that
  // Peek or Pop returns).
  int N() const { return n - (int)terms.size(); }

  // This pops the first element of the series tail, computing
  // more if necessary.
  BigRat Pop() {
    if (terms.empty()) Push();
    CHECK(!terms.empty());
    BigRat r = std::move(terms.front());
    terms.pop_front();
    return r;
  }

  const BigRat &Peek() {
    if (terms.empty()) Push();
    CHECK(!terms.empty());
    return terms.front();
  }

  // Bound on the sum of the tail.
  // Leibniz's rule says that this is bounded by
  // the second term in the tail.
  BigRat Bound() {
    while (terms.size() < 2) Push();
    return BigRat::Abs(terms[1]);
  }

 private:
  void Push() {
    //         (1/x)^(2n + 1)
    // (-1)^n --------------
    //         2n + 1

    // computed as
    //
    //           (-1)^n
    //          --------
    //          x^(2n+1)
    //        ------------
    //          2n + 1

    const int v = 2 * n + 1;
    // Signs alternate positive and negative.
    BigRat numer = BigRat(BigInt((n & 1) ? -1 : 1), xpow);
    terms.push_back(BigRat::Div(numer, BigInt(v)));

    // Increase exponent by 2.
    xpow = xpow * xx;
    n++;
  }

  // x^2
  const int64_t xx = 0;
  // Next term to be computed.
  int n = 0;
  // the power of x for term n.
  BigInt xpow;
  std::deque<BigRat> terms;
};

[[maybe_unused]]
static BigRat MakePi(int digits) {
  static constexpr bool VERBOSE = false;

  BigRat epsilon(BigInt(1), BigInt::Pow(BigInt(10), digits + 1));
  if (VERBOSE) {
    printf("Compute pi with epsilon = %s\n",
           epsilon.ToString().c_str());
  }

  // https://en.wikipedia.org/wiki/Machin-like_formula
  // π / 4 = 4 * arctan(1/5) - arctan(1/239)
  // arctan(x) = Σ (-1)^n * (x^(2n + 1)) / (2n + 1)
  //    = x^1 / 1 - x^3 / 3 + x^5 / 5 - x^7 / 7 + ...

  ArctanSeries a(5), b(239);

  BigRat sum = BigRat(0);
  for (;;) {
    if (VERBOSE) printf("Enter with sum = %s\n", sum.ToString().c_str());
    BigRat err_bound = BigRat::Max(a.Bound(), b.Bound());
    if (err_bound < epsilon) {
      // We computed π / 4.
      return sum * BigRat(4);
    }

    // Add the larger of the two terms.
    const BigRat &terma = a.Peek();
    const BigRat &termb = b.Peek();

    BigRat terma4 = terma * BigRat(4);

    // Add the term that has the larger magnitude.
    if (BigRat::Abs(terma4) > BigRat::Abs(termb)) {
      if (VERBOSE)
        printf("Add term #%d of a: %s * 4\n",
               a.N(),
               terma.ToString().c_str());
      sum = sum + terma4;
      a.Pop();
    } else {
      if (VERBOSE)
        printf("Subtract term #%d of b: %s\n",
               b.N(),
               termb.ToString().c_str());

      sum = sum - termb;
      b.Pop();
    }
  }
}

BigRat Sqrt(BigRat xx, int digits) {
  BigRat epsilon(BigInt(1), BigInt::Pow(BigInt(10), digits + 1));
  BigRat two(2);

  // "Heron's Method".
  BigRat x = BigInt(1);
  for (;;) {
    // So we have xx = x * y.
    BigRat y = xx / x;
    if (BigRat::Abs(x - y) < epsilon) {
      return y;
    }
    x = (x + y) / two;
  }
}

// Quaternion as xi + yj + zk + w
struct BigQuat {
  BigQuat(BigRat x, BigRat y, BigRat z, BigRat w) :
    x(std::move(x)), y(std::move(y)), z(std::move(z)), w(std::move(w)) {}
  BigRat x = BigRat(0), y = BigRat(0), z = BigRat(0), w = BigRat(1);
};


std::string QuatString(const BigQuat &q) {
  return StringPrintf(
      "x: " ARED("%s") "\n"
      "y: " AGREEN("%s") "\n"
      "z: " ABLUE("%s") "\n"
      "w: " AYELLOW("%s") "\n",
      q.x.ToString().c_str(),
      q.y.ToString().c_str(),
      q.z.ToString().c_str(),
      q.w.ToString().c_str());
}

BigQuat Normalize(const BigQuat &q, int digits) {
  printf("Quat: %s\n", QuatString(q).c_str());
  BigRat norm_sq = q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w;
  printf("Norm^2: %s\n", norm_sq.ToString().c_str());
  CHECK(norm_sq != BigRat(0));
  BigRat norm = Sqrt(norm_sq, digits);
  printf("Norm: %s\n", norm.ToString().c_str());
  CHECK(norm != BigRat(0));
  return BigQuat(q.x / norm, q.y / norm, q.z / norm, q.w / norm);
}

BigQuat UnitInverse(const BigQuat &q) {
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

inline BigVec3 RotatePoint(const BigQuat &q, const BigVec3 &v) {
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
  BigQuat pp = q * p * UnitInverse(q);
  return BigVec3(std::move(pp.x), std::move(pp.y), std::move(pp.z));
}

struct BigPoly {
  std::vector<BigVec3> vertices;
  const Faces *faces = nullptr;
};

struct BigMesh2D {
  std::vector<BigVec2> vertices;
  const Faces *faces = nullptr;
};

static BigPoly Rotate(const BigQuat &q, const BigPoly &poly) {
  std::vector<BigVec3> vertices;
  vertices.reserve(poly.vertices.size());
  for (const BigVec3 &v : poly.vertices) {
    vertices.push_back(RotatePoint(q, v));
  }
  return BigPoly{.vertices = std::move(vertices), .faces = poly.faces};
}

static BigMesh2D Shadow(const BigPoly &poly) {
  std::vector<BigVec2> vertices;
  vertices.reserve(poly.vertices.size());
  for (const BigVec3 &v : poly.vertices) {
    vertices.emplace_back(v.x, v.y);
  }

  return BigMesh2D{.vertices = std::move(vertices), .faces = poly.faces};
}

static BigMesh2D Translate(const BigVec2 &t, const BigMesh2D &m) {
  std::vector<BigVec2> vertices;
  vertices.reserve(m.vertices.size());
  for (const BigVec2 &v : m.vertices) {
    vertices.push_back(v + t);
  }

  return BigMesh2D{.vertices = std::move(vertices), .faces = m.faces};
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


static BigPoly BigRidode() {
  const BigRat phi = (BigRat(1) + Sqrt(BigRat(5), DIGITS)) / BigRat(2);
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

[[maybe_unused]]
static bool InMesh(const BigMesh2D &mesh, const BigVec2 &pt) {
  // using the triangulation
  for (const auto &[ai, bi, ci] : mesh.faces->triangulation) {
    const BigVec2 &a = mesh.vertices[ai];
    const BigVec2 &b = mesh.vertices[bi];
    const BigVec2 &c = mesh.vertices[ci];

    if (InTriangle(a, b, c, pt)) return true;
  }

  return false;
}

// Check if the point is strictly within *any* triangle induced
// by the point set.
static std::optional<std::tuple<int, int, int>>
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


static void Validate() {
  // BigRat pi = MakePi(DIGITS);
  // printf("pi: %s\n", pi.ToString().c_str());

  // BigRat sqrt2 = Sqrt(BigRat(2), DIGITS);
  // printf("sqrt2: %s\n", sqrt2.ToString().c_str());

  #if 0
  for (const vec3 &v : Rhombicosidodecahedron().vertices) {
    printf("%.17g,%.17g,%.17g\n", v.x, v.y, v.z);
  }
  printf("-----big----\n");
  #endif

  BigPoly ridode = BigRidode();

  SolutionDB db;
  SolutionDB::Solution sol = db.GetSolution(455);
  printf("Solution:\nOuter: %s\nInner: %s\n",
         FrameString(sol.outer_frame).c_str(),
         FrameString(sol.inner_frame).c_str());

  const auto &[douter_rot, dotrans] =
    UnpackFrame(sol.outer_frame);
  const auto &[dinner_rot, ditrans] =
    UnpackFrame(sol.inner_frame);

  CHECK(dotrans.x == 0.0 &&
        dotrans.y == 0.0 &&
        dotrans.z == 0.0) << "This can be handled, but we expect "
    "exact zero translation for the outer frame.";

  BigVec2 itrans(BigRat::FromDouble(ditrans.x),
                 BigRat::FromDouble(ditrans.y));

  printf("Made itrans\n");

  printf("Outer: %s\n", QuatString(douter_rot).c_str());
  printf("Inner: %s\n", QuatString(dinner_rot).c_str());

  BigQuat oq(BigRat::FromDouble(douter_rot.x),
             BigRat::FromDouble(douter_rot.y),
             BigRat::FromDouble(douter_rot.z),
             BigRat::FromDouble(douter_rot.w));

  BigQuat iq(BigRat::FromDouble(dinner_rot.x),
             BigRat::FromDouble(dinner_rot.y),
             BigRat::FromDouble(dinner_rot.z),
             BigRat::FromDouble(dinner_rot.w));

  printf("Outer: %s\n", QuatString(oq).c_str());

  // Represent the double quaternions as rational with a lot of
  // digits of precision.
  printf("Outer:\n");
  BigQuat outer_rot = Normalize(oq, DIGITS);

  printf("Inner:\n");
  BigQuat inner_rot = Normalize(iq, DIGITS);

  printf("Made BigQuats\n");

  BigPoly outer = Rotate(outer_rot, ridode);
  BigPoly inner = Rotate(inner_rot, ridode);

  printf("Rotated\n");

  BigMesh2D souter = Shadow(outer);
  BigMesh2D sinner = Translate(itrans, Shadow(inner));

  printf("Check:\n");
  // Now check
  bool valid = true;
  std::vector<int> ins, outs;

  Polyhedron renderpoly = SmallPoly(ridode);
  Mesh2D small_souter = SmallMesh(souter);
  Mesh2D small_sinner = SmallMesh(sinner);
  auto RenderInside = [&](
      std::optional<std::tuple<int, int, int>> triangle,
      int ptidx) {

      Rendering rendering(renderpoly, 1920, 1080);

      // Polyhedron outer = Rotate(renderpoly, sol.outer_frame);
      // Polyhedron inner = Rotate(renderpoly, sol.inner_frame);
      // Mesh2D souter = Shadow(outer);
      // Mesh2D sinner = Shadow(inner);

      rendering.RenderMesh(small_souter);
      rendering.DarkenBG();

      if (triangle.has_value()) {
        const auto &[a, b, c] = triangle.value();
        rendering.RenderTriangle(small_sinner, a, b, c, 0x3333AAAA);
        rendering.MarkPoints(small_sinner, {ptidx}, 20.0f, 0x00FF00AA);
      } else {
        rendering.MarkPoints(small_sinner, {ptidx}, 20.0f, 0xFF0000AA);
      }

      rendering.Save(StringPrintf("validate-%d.png", ptidx));
    };

  // std::vector<int> inner_hull = BigHull(sinner.vertices);

  // for (int i = 0; i < sinner.vertices.size(); i++) {
  // for (int i : inner_hull) {
  for (int i = 0; i < sinner.vertices.size(); i++) {
    const BigVec2 &v = sinner.vertices[i];
    const std::optional<std::tuple<int, int, int>> triangle =
      InMeshExhaustive(souter, v);
    bool in = triangle.has_value();
    printf("Point %d is %s\n", i, in ? AGREEN("in") : ARED("out"));
    valid = valid && in;
    if (in) {
      RenderInside(triangle, i);
      ins.push_back(i);
    } else {
      outs.push_back(i);
    }
  }

  {
    Rendering rendering(renderpoly, 1920, 1080);

    rendering.RenderTriangulation(small_souter, 0x00FF0044);
    rendering.RenderTriangulation(small_sinner, 0xFF000044);

    rendering.MarkPoints(small_sinner, ins, 10.0f, 0x22FF22AA);
    rendering.MarkPoints(small_sinner, outs, 20.0f, 0xFF2211AA);
    rendering.Save("validate.png");
  }

  printf("Done\n");
}

int main(int argc, char **argv) {
  ANSI::Init();

  Validate();

  printf("Done.\n");
  return 0;
}
