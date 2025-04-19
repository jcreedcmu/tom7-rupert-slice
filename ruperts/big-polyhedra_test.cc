#include "big-polyhedra.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "ansi.h"
#include "arcfour.h"
#include "bignum/big-overloads.h"
#include "bignum/big.h"
#include "randutil.h"
#include "timer.h"

// To big-overloads?
// I remember from working with "half" that this has to parse them
// each time, so maybe we should discourage it.
BigRat operator""_r(const char* str, std::size_t len) {
  std::string_view s(str, len);
  std::size_t pos = s.find('/');
  if (pos == std::string_view::npos) {
    return BigRat(BigInt(s));
  } else {
    std::string_view numer = s.substr(0, pos);
    std::string_view denom = s.substr(pos + 1);
    return BigRat(BigInt(numer), BigInt(denom));
  }
}

static void TestLiterals() {
  CHECK("1/3"_r == BigRat(1, 3));
  CHECK("-1/3"_r == BigRat(-1, 3));
  CHECK("1/-3"_r == BigRat(-1, 3));
  CHECK("0"_r == BigRat(0));
  CHECK("123456789123456789/986543219865432198654321"_r ==
        BigRat(BigInt("123456789123456789"),
               BigInt("986543219865432198654321")));
}

static void TestBigQuickHull() {

  // TODO: More thorough testing. This one should be mathematically
  // exact. This idea of starting with the hull and then generating
  // points inside it would extend naturally to more test cases!

  BigVec2 o("0"_r, "0"_r);
  BigVec2 a("-1/3"_r, "-1/4"_r);
  BigVec2 b("8"_r, "-1/4"_r);
  BigVec2 c("-1/3"_r, "5"_r);

  //     -1/3, -1/4                8, -1/4
  //         a-----------------------b
  //         |                   .
  //         |   +o          .
  //         |           .
  //         |       .
  //         |   .
  //         c
  //     -1/3, 5

  ArcFour rc("qh");
  std::vector<BigVec2> input = {o, a, b, c};
  for (int i = 0; i < 5000; i++) {
    // Now add some more points in the hull. This includes
    // duplicates and colinear points.
    const BigVec2 &p = input[RandTo(&rc, input.size())];
    const BigVec2 &q = input[RandTo(&rc, input.size())];

    BigVec2 avg((p.x + q.x) / "2"_r, (p.y + q.y) / "2"_r);
    // Don't duplicate hull points, as this complicates the test.
    // (It should work correctly, though!)
    if (avg == a || avg == b || avg == c) continue;
    input.push_back(std::move(avg));
  }

  Timer timer;
  std::vector<int> hull = BigQuickHull(input);
  printf("Got hull in %s\n", ANSI::Time(timer.Seconds()).c_str());
  std::sort(hull.begin(), hull.end());

  // Should be exactly the triangle.
  CHECK(hull.size() == 3);
  CHECK(hull[0] == 1);
  CHECK(hull[1] == 2);
  CHECK(hull[2] == 3);
}

static void TestQuatMult() {
  const BigQuat i(BigRat(1), BigRat(0), BigRat(0), BigRat(0));
  const BigQuat j(BigRat(0), BigRat(1), BigRat(0), BigRat(0));
  const BigQuat k(BigRat(0), BigRat(0), BigRat(1), BigRat(0));
  const BigQuat one(BigRat(0), BigRat(0), BigRat(0), BigRat(1));
  const BigQuat neg_one(BigRat(0), BigRat(0), BigRat(0), BigRat(-1));
  const BigQuat zero(BigRat(0), BigRat(0), BigRat(0), BigRat(0));

  CHECK(k == i * j);
  CHECK(i == j * k);
  CHECK(j == k * i);

  CHECK(neg_one * k == j * i);
  CHECK(neg_one * i == k * j);
  CHECK(neg_one * j == i * k);

  CHECK(neg_one == i * i);
  CHECK(neg_one == j * j);
  CHECK(neg_one == k * k);

  CHECK(zero == i * zero);
  CHECK(zero == zero * i);
  CHECK(zero == zero * zero);

  CHECK(BigQuat(BigRat(1), BigRat(0), BigRat(0), BigRat(1)) *
        BigQuat(BigRat(0), BigRat(1), BigRat(0), BigRat(1)) ==
        BigQuat(BigRat(1), BigRat(1), BigRat(1), BigRat(1)));
}

int main(int argc, char **argv) {
  ANSI::Init();

  TestLiterals();
  TestBigQuickHull();
  TestQuatMult();


  printf("OK");
  return 0;
}
