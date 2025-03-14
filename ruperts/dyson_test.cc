#include "dyson.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <format>
#include <optional>
#include <string>
#include <utility>

#include "ansi.h"
#include "arcfour.h"
#include "base/logging.h"
#include "polyhedra.h"
#include "randutil.h"
#include "yocto_matht.h"

using vec3 = Dyson::vec3;
using quat4 = Dyson::quat4;
using frame3 = Dyson::frame3;

#define CHECK_NEAR(f1, f2) \
  CHECK(std::abs((f1) - (f2)) < 1e-10)

static std::string IntersectionString(
    const std::optional<std::pair<double, double>> &isect) {
  if (isect.has_value()) {
    return std::format("{} to {}",
                       isect.value().first,
                       isect.value().second);
  } else {
    return "(none)";
  }
}

static void TestUnitCubeIntersection() {
  {
    vec3 p1(-1.0, 0.5, 0.5);
    vec3 p2(0.5, 0.5, 0.5);
    auto isect = Dyson::SegmentIntersectsUnitCube(p1, p2);
    CHECK(isect.has_value());
    CHECK(isect->first >= 0.0 &&
          isect->first <= 1.0 &&
          isect->second >= 0.0 &&
          isect->second <= 1.0) << IntersectionString(isect);
  }

  {
    // As an infinite line, this would intersect the cube, but it
    // stops outside.
    vec3 p1(-2.0, 0.5, 0.5);
    vec3 p2(-1.25, 0.5, 0.5);
    auto isect = Dyson::SegmentIntersectsUnitCube(p1, p2);
    CHECK(!isect.has_value()) << IntersectionString(isect);
  }
}

static void TestRandomCubeIntersection() {
  ArcFour rc("test");
  for (int i = 0; i < 1000; i++) {
    // Random point in cube.
    vec3 p{
      0.05 + RandDouble(&rc) * 0.9,
      0.05 + RandDouble(&rc) * 0.9,
      0.05 + RandDouble(&rc) * 0.9};

    // Random point NOT in cube.
    vec3 q{
      0.05 + RandDouble(&rc) * 0.9,
      0.05 + RandDouble(&rc) * 0.9,
      0.05 + RandDouble(&rc) * 0.9};
    // Add or subtract 1 along at least one axis.

    uint8_t b = 0;
    while ((b & 0b111111) == 0) b = rc.Byte();
    if (b & 0b100000) {
      q.x += 1.0;
    } else if (b & 0b100) {
      q.x -= 1.0;
    }

    if (b & 0b010000) {
      q.y += 1.0;
    } else if (b & 0b010) {
      q.y -= 1.0;
    }

    if (b & 0b001000) {
      q.z += 1.0;
    } else if (b & 0b001) {
      q.z -= 1.0;
    }

    if (rc.Byte() & 1) {
      std::swap(p, q);
    }

    auto isect = Dyson::SegmentIntersectsUnitCube(p, q);
    CHECK(isect.has_value());
    CHECK(isect->first >= 0.0 &&
          isect->first <= 1.0 &&
          isect->second >= 0.0 &&
          isect->second <= 1.0) << IntersectionString(isect) <<
      std::format("p: {}, {}, {}\n"
                  "q: {}, {}, {}\n",
                  p.x, p.y, p.z,
                  q.x, q.y, q.z);
  }
}

static void TestExtractRotation() {
  ArcFour rc("test");

  for (int i = 0; i < 100'000; i++) {
    quat4 orig = RandomQuaternion(&rc);

    frame3 frame = rotation_frame(orig);

    quat4 extracted = Dyson::rotation_quat(frame);

    frame3 reframe = rotation_frame(extracted);

    for (int c = 0; c < 3; c++) {
      for (int r = 0; r < 3; r++) {
        CHECK_NEAR(frame[c][r], reframe[c][r]) <<
          "Correct so far: " << i << "\n\n"
          AYELLOW("Original") ":\n" << FrameString(frame) << "\n\n"
          AORANGE("Quat") ":\n" << QuatString(extracted) << "\n\n"
          ARED("Resulting frame") ":\n" << FrameString(reframe);
      }
    }
  }
}

int main(int argc, char **argv) {
  ANSI::Init();
  TestExtractRotation();
  TestUnitCubeIntersection();
  TestRandomCubeIntersection();

  printf("OK\n");
  return 0;
}
