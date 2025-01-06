
#include "ansi.h"

#include <cstdint>
#include <cstdlib>
#include <deque>
#include <cstdio>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "base/logging.h"
#include "bignum/big-overloads.h"
#include "bignum/big.h"
#include "point-set.h"
#include "polyhedra.h"

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


BigQuat Normalize(const BigQuat &q, int digits) {
  BigRat norm_sq = q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w;
  BigRat norm = Sqrt(norm_sq, digits);
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

inline BigVec3 Rotate(const BigQuat &q, const BigVec3 &v) {
  // PERF: This can be simplified.
  //  - w coefficient for pure quaternion is zero.
  //  - q^-1 * p * q will be a pure quaternion for
  //    unit q, so we don't need to compute the w coefficient.
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
  std::vector<BigVec3> vertices;
  const Faces *faces = nullptr;
};

BigPoly MakeBigPolyFromVertices(std::vector<BigVec3> vertices) {
  PointSet duplicates;

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
    AddEvenPermutations(s1 * phi_squared, s2 * phi, s3 * 2.0 * phi, &vertices);
  }

  for (int b = 0b00; b < 0b100; b++) {
    double s1 = (b & 0b10) ? -1 : +1;
    double s2 = (b & 0b01) ? -1 : +1;
    // (±(2+φ), 0, ±φ^2),
    AddEvenPermutations(s1 * (2.0 + phi), 0.0, s2 * phi_squared, &vertices);
  }

  CHECK(vertices.size() == 60) << vertices.size();
  return MakeBigPolyFromVertices(std::move(vertices));
}

static void Validate() {
  BigRat pi = MakePi(DIGITS);

  printf("pi: %s\n", pi.ToString().c_str());

  BigRat sqrt2 = Sqrt(BigRat(2), DIGITS);
  printf("sqrt2: %s\n", sqrt2.ToString().c_str());

  #if 0
  for (const vec3 &v : Rhombicosidodecahedron().vertices) {
    printf("%.17g,%.17g,%.17g\n", v.x, v.y, v.z);
  }
  printf("-----big----\n");
  #endif

  BigPoly ridode = BigRidode();
}

int main(int argc, char **argv) {
  ANSI::Init();

  Validate();

  return 0;
}
