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

static void TestViewPosFromQuat() {
  printf("Testing ViewPosFromNonUnitQuat...\n");

  // Helper to check the core property: Applying the inverse rotation
  // (R^T) defined by q to the view position v should yield (0, 0, 1).
  // Also checks that v is a unit vector.
  auto CheckView = [](const BigQuat &q, const BigVec3 &v) {
      CHECK(length_squared(v) == "1"_r)
        << "Must be unit. But got: v = " << VecString(v)
        << " len^2 = " << length_squared(v).ToString();

      // Rotating the view should give us (0, 0, 1).
      BigFrame frame = NonUnitRotationFrame(q);
      BigVec3 transformed_v = TransformPoint(frame, v);

      const BigVec3 expected_z("0"_r, "0"_r, "1"_r);
      CHECK(transformed_v == expected_z)
        << "q = " << QuatString(q)
        << "v = " << VecString(v)
        << "R^T * v = " << VecString(transformed_v)
        << "Expected " << VecString(expected_z);
    };

  // Case 1: Identity quaternion
  {
    BigQuat q_id("0"_r, "0"_r, "0"_r, "1"_r);
    BigVec3 v_id = ViewPosFromNonUnitQuat(q_id);
    // For identity rotation R=I, R^T=I. We need I*v = (0,0,1), so v = (0,0,1).
    CHECK(v_id == BigVec3("0"_r, "0"_r, "1"_r));
    CheckView(q_id, v_id);
  }

  {
    // This is a unit quaternion.
    BigQuat q_180y("0"_r, "1"_r, "0"_r, "0"_r);
    BigVec3 v_180y = ViewPosFromNonUnitQuat(q_180y);
    CHECK(v_180y == BigVec3("0"_r, "0"_r, "-1"_r));
    CheckView(q_180y, v_180y);
    // Check non-unit version
    BigQuat q_180y_nu = q_180y * "2"_r;
    BigVec3 v_180y_nu = ViewPosFromNonUnitQuat(q_180y_nu);
    CHECK(v_180y_nu == v_180y);
    CheckView(q_180y_nu, v_180y_nu);
  }

  // Case 3: 90 deg rotation around Z axis
  {
    BigQuat q_90z("0"_r, "0"_r, "1"_r, "1"_r);
    BigVec3 v_90z = ViewPosFromNonUnitQuat(q_90z);
    CHECK(v_90z == BigVec3("0"_r, "0"_r, "1"_r)) << VecString(v_90z);
    CheckView(q_90z, v_90z);
  }

  {
    BigQuat q_90y("0"_r, "1"_r, "0"_r, "1"_r);
    BigVec3 v_90y = ViewPosFromNonUnitQuat(q_90y);
    CHECK(v_90y == BigVec3("-1"_r, "0"_r, "0"_r)) << VecString(v_90y);
    CheckView(q_90y, v_90y);
  }

  {
    BigQuat q_xyz("1"_r, "1"_r, "1"_r, "1"_r);
    BigVec3 v_xyz = ViewPosFromNonUnitQuat(q_xyz);
    CHECK(v_xyz == BigVec3("0"_r, "1"_r, "0"_r)) << VecString(v_xyz);
    CheckView(q_xyz, v_xyz);
  }

  // Case 6: General rational quaternion
  {
    BigQuat q_gen("1/2"_r, "-1/3"_r, "3/4"_r, "1/5"_r);
    BigVec3 v_gen = ViewPosFromNonUnitQuat(q_gen);
    // No simple check for the value of v, just check the property
    CheckView(q_gen, v_gen);
  }

  for (int i = 0; i < 1000; i++) {
    ArcFour rc("qh");
    BigRat x = BigRat::FromDouble(RandDouble(&rc) * 5 - 2.5);
    BigRat y = BigRat::FromDouble(RandDouble(&rc) * 5 - 2.5);
    BigRat z = BigRat::FromDouble(RandDouble(&rc) * 5 - 2.5);
    BigRat w = BigRat::FromDouble(RandDouble(&rc) * 5 - 2.5);

    BigQuat q(x, y, z, w);
    if (x == BigRat(0) && y == BigRat(0) && z == BigRat(0) && w == BigRat(0))
      continue;

    CheckView(q, ViewPosFromNonUnitQuat(q));
  }

  printf("TestViewPosFromQuat OK.\n");
}

int main(int argc, char **argv) {
  ANSI::Init();

  TestLiterals();
  TestBigQuickHull();
  TestQuatMult();

  TestViewPosFromQuat();

  printf("OK");
  return 0;
}
