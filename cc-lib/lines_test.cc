
#include "lines.h"

#include <optional>
#include <unordered_set>

#include "arcfour.h"
#include "base/logging.h"
#include "base/stringprintf.h"
#include "hashing.h"
#include "randutil.h"
#include "set-util.h"

using namespace std;

#define CHECK_ALMOST_EQ(a, b) do { \
  auto aa = (a); \
  auto bb = (b); \
  CHECK(fabs(aa - bb) < EPSILON) << aa << " vs " << bb << " in " \
                                 << #a << " =~= " << #b ; \
 } while(0)

static void TestBresenham() {
  printf("Bresenham:\n");
  for (const std::pair<int, int> point : Line<int>{0, 0, 5, 5}) {
    printf("%d,%d  ", point.first, point.second);
  }

  printf("\n--\n");
  for (const std::pair<int, int> point : Line<int>{5, 5, 0, 0}) {
    printf("%d,%d  ", point.first, point.second);
  }

  printf("\n--\n");
  for (const std::pair<int, int> point : Line<int>{5, 4, 0, 0}) {
    printf("%d,%d  ", point.first, point.second);
  }

  printf("\n--\n");
  for (auto [x, y] : Line<int>{4, 5, 0, 0}) {
    printf("%d,%d  ", x, y);
  }

  printf("\n--\n");
}

static void TestEmpty() {
  for (auto [x, y] : Line<int>::Empty()) {
    LOG(FATAL) << "Should not emit any points for empty line. Got "
               << x << ", " << y;
  }
}

static void TestWu() {
  printf("Wu:\n");
  auto Plot = [](int x, int y, float f) {
      printf("%d,%d %.2f  ", x, y, f);
    };
  LineAA::Draw<int>(0.5f, 4.0f, 2.5f, 3.0f, Plot);
  printf("\n---\n");
}

// TODO: This could certainly be more comprehensive!
static void TestIntersection() {
  // Parallel.
  CHECK(!LineIntersection(3, 1,   10, 1,
                          2, 5,   10, 5).has_value());

  // Parallel.
  CHECK(!LineIntersection(2, 1,   5, 2,
                          1, 3,   4, 4).has_value());

  // Segments not long enough to intersect
  CHECK(!LineIntersection(2, 1,   5, 2,
                          2, 3,   3, 2).has_value());
  CHECK(!LineIntersection(2.0f, 1.0f,   5.0f, 2.0f,
                          2.0f, 3.0f,   3.0f, 2.0f).has_value());

  // Trivial cross at 0.
  auto z = LineIntersection(0, -1,  0, 1,
                            -1, 0,  1, 0);
  CHECK(z.has_value());
  static constexpr float EPSILON = 1e-10;
  CHECK(fabs(z.value().first) < EPSILON &&
        fabs(z.value().second) < EPSILON);

  {
    auto li = LineIntersection(1.0f, 1.0f,  4.0f, 4.0f,
                               3.0f, 1.0f,  2.0f, 4.0f);
    CHECK(li.has_value());
    auto [x, y] = li.value();
    CHECK(fabs(x - 2.5f) < EPSILON) << x;
    CHECK(fabs(y - 2.5f) < EPSILON) << y;
  }

  // Same but with integer coordinates.
  {
    auto li = LineIntersection(1, 1,  4, 4,
                               3, 1,  2, 4);
    CHECK(li.has_value());
    auto [x, y] = li.value();
    CHECK(fabs(x - 2.5f) < EPSILON) << x;
    CHECK(fabs(y - 2.5f) < EPSILON) << y;
  }
}

// Using PointLineDistance as a reference.
static void TestVertHoriz() {
  static constexpr float EPSILON = 0.0001f;
  ArcFour rc{"lines_test"};
  for (int i = 0; i < 10000; i++) {
    float a = RandDouble(&rc) * 100.0f - 50.0f;
    float b = RandDouble(&rc) * 100.0f - 50.0f;
    float c = RandDouble(&rc) * 100.0f - 50.0f;
    float x = RandDouble(&rc) * 100.0f - 50.0f;
    float y = RandDouble(&rc) * 100.0f - 50.0f;

    float dh = PointLineDistance(a, b, c, b, x, y);
    float dht = PointHorizLineDistance(a, b, c, x, y);
    CHECK(fabsf(dh - dht) < EPSILON) << dh << " vs " << dht;
    float dv = PointLineDistance(a, b, a, c, x, y);
    float dvt = PointVertLineDistance(a, b, c, x, y);
    CHECK(fabsf(dv - dvt) < EPSILON) << dv << " vs " << dvt;
  }
}

static void TestReflect() {
  static constexpr float EPSILON = 0.00001f;
  {
    auto [x, y] = ReflectPointAboutLine<float>(
        // horizontal line y=1
        -3.2f, 1,
        777.1f, 1,
        4.0, 3.0);
    CHECK_ALMOST_EQ(x, 4.0f);
    CHECK_ALMOST_EQ(y, -1.0f);
  }

  {
    auto [x, y] = ReflectPointAboutLine<float>(
        // vertical line x = 3
        3.0f, 1.0f,
        3.0f, 9.0f,
        7.0f, 33.3);
    CHECK_ALMOST_EQ(x, -1.0f);
    CHECK_ALMOST_EQ(y, 33.3f);
  }

  {
    auto [x, y] = ReflectPointAboutLine<float>(
        // nontrivial diagonal
        -1.0f, -2.0f,
        2.0f, 4.0f,
        1.0f, -2.0f);
    CHECK_ALMOST_EQ(x, -11.0f / 5.0f);
    CHECK_ALMOST_EQ(y, -2.0f / 5.0f);
  }

  {
    auto [x, y] = ReflectPointAboutLine<float>(
        // nontrivial diagonal
        -1.0f, -2.0f,
        2.0f, 4.0f,
        // origin lands on it
        0.0f, 0.0f);
    CHECK_ALMOST_EQ(x, 0.0f);
    CHECK_ALMOST_EQ(y, 0.0f);
  }

}

#if 0
// TODO: FIXME
static void TestClipBresenham() {
  ArcFour rc("clip");

  for (int i = 0; i < 100; i++) {
    std::unordered_set<std::pair<int, int>,
                       Hashing<std::pair<int, int>>> expected;
    std::unordered_set<std::pair<int, int>,
                       Hashing<std::pair<int, int>>> actual;

    int x0 = RandTo(&rc, 10);
    int y0 = RandTo(&rc, 10);
    int x1 = RandTo(&rc, 10);
    int y1 = RandTo(&rc, 10);

    int xcmin = 2;
    int ycmin = 3;
    int xcmax = 8;
    int ycmax = 7;

    printf("Test clip: (%d,%d) to (%d,%d), clipped in box (%d,%d) to (%d,%d)\n",
           x0, y0, x1, y1, xcmin, ycmin, xcmax, ycmax);

    for (const auto &[x, y] : Line<int>(x0, y0, x1, y1)) {
      if (x >= xcmin && y >= ycmin &&
          x <= xcmax && y <= ycmax) {
        expected.insert(std::make_pair(x, y));
      }
    }

    printf("Done expected\n");

    for (const auto &[x, y] : Line<int>::ClippedLine(x0, y0, x1, y1,
                                                     xcmin, ycmin,
                                                     xcmax, ycmax)) {
      actual.insert(std::make_pair(x, y));
      CHECK(x >= 0 && y >= 0 && x <= 10 && y <= 10);
    }

    printf("Done actual\n");

    if (expected != actual) {
      printf("Expected:");
      for (const auto &[a, b] : SetToSortedVec(expected)) {
        printf(" (%d,%d)", a, b);
      }

      printf("\nActual:");
      for (const auto &[a, b] : SetToSortedVec(actual)) {
        printf(" (%d,%d)", a, b);
      }
      printf("\n");

      LOG(FATAL) << "Did not get expected points.\n";
    }
    printf("OK\n");
  }
  printf("ClippedLine OK\n");
}
#endif

// Check entire clipped line.
#define CHECK_CLIPPED(clipped_arg, ex0, ey0, ex1, ey1) do {        \
    auto clipped_ = (clipped_arg);                                 \
    CHECK(clipped_.has_value());                                   \
    const auto &[cx0, cy0, cx1, cy1] = clipped_.value();           \
    static constexpr float EPSILON = 0.00001f;                     \
    CHECK_ALMOST_EQ(cx0, ex0);                                     \
    CHECK_ALMOST_EQ(cy0, ey0);                                     \
    CHECK_ALMOST_EQ(cx1, ex1);                                     \
    CHECK_ALMOST_EQ(cy1, ey1);                                     \
  } while (0)

static void TestClipRectangle() {
  float xmin = 1.0f;
  float ymin = 2.0f;
  float xmax = 4.0f;
  float ymax = 3.0f;

  // 1. Completely Inside.
  CHECK_CLIPPED(
      ClipLineToRectangle(2.0f, 2.5f, 3.0f, 2.8f, xmin, ymin, xmax, ymax),
      2.0f, 2.5f, 3.0f, 2.8f);

  // 2. Completely Outside (left).
  CHECK(!ClipLineToRectangle(-1.0f, 2.5f, 0.0f, 2.8f, xmin, ymin, xmax, ymax).
        has_value());

  // 3. Completely Outside (right).
  CHECK(!ClipLineToRectangle(5.0f, 2.5f, 6.0f, 2.8f, xmin, ymin, xmax, ymax).
        has_value());

  // 4. Completely Outside (above).
  CHECK(!ClipLineToRectangle(2.0f, 4.0f, 3.0f, 5.0f, xmin, ymin, xmax, ymax).
        has_value());

  // 5. Completely Outside (below).
  CHECK(!ClipLineToRectangle(2.0f, 0.0f, 3.0f, 1.0f, xmin, ymin, xmax, ymax).
        has_value());

  // 6. Partially Inside (intersecting left).
  CHECK_CLIPPED(
      ClipLineToRectangle(0.0f, 2.5f, 2.0f, 2.5f, xmin, ymin, xmax, ymax),
      1.0f, 2.5f, 2.0f, 2.5f);


  // 7. Partially Inside (intersecting right).
  CHECK_CLIPPED(
      ClipLineToRectangle(3.0f, 2.5f, 5.0f, 2.5f, xmin, ymin, xmax, ymax),
      3.0f, 2.5f, 4.0f, 2.5f);

  // 8. Partially Inside (intersecting top).
  CHECK_CLIPPED(
      ClipLineToRectangle(2.5f, 2.5f, 2.5f, 4.0f, xmin, ymin, xmax, ymax),
      2.5f, 2.5f, 2.5f, 3.0f);

  // 9. Partially Inside (intersecting bottom).
  CHECK_CLIPPED(
      ClipLineToRectangle(2.5f, 1.0f, 2.5f, 2.5f, xmin, ymin, xmax, ymax),
      2.5f, 2.0f, 2.5f, 2.5f);

  // 10. Horizontal line, completely inside.
  CHECK_CLIPPED(
      ClipLineToRectangle(2.0f, 2.5f, 3.0f, 2.5f, xmin, ymin, xmax, ymax),
      2.0f, 2.5f, 3.0f, 2.5f);

  // 11. Vertical line, completely inside.
  CHECK_CLIPPED(
      ClipLineToRectangle(2.5f, 2.2f, 2.5f, 2.8f, xmin, ymin, xmax, ymax),
      2.5f, 2.2f, 2.5f, 2.8f);

  // 12. Horizontal line, intersecting left and right.
  CHECK_CLIPPED(
    ClipLineToRectangle(0.0f, 2.5f, 5.0f, 2.5f, xmin, ymin, xmax, ymax),
    1.0f, 2.5f, 4.0f, 2.5f);

  // 13. Vertical line, intersecting top and bottom.
  CHECK_CLIPPED(
      ClipLineToRectangle(2.5f, 1.0f, 2.5f, 4.0f, xmin, ymin, xmax, ymax),
      2.5f, 2.0f, 2.5f, 3.0f);

  // 14. Diagonal line, intersecting top and bottom.
  {
    auto clipped = ClipLineToRectangle(0.0f, 1.0f, 5.0f, 4.0f,
                                       xmin, ymin, xmax, ymax);
    auto ix1 = LineIntersection(0.0f, 1.0f, 5.0f, 4.0f, xmin, ymin, xmax, ymin);
    auto ix2 = LineIntersection(0.0f, 1.0f, 5.0f, 4.0f, xmin, ymax, xmax, ymax);
    CHECK(ix1.has_value());
    CHECK(ix2.has_value());
    CHECK_CLIPPED(clipped,
                  ix1->first, ix1->second, ix2->first, ix2->second);
  }

  // 15. Diagonal line intersecting left and right.
  {
    auto clipped = ClipLineToRectangle(0.0f, 2.5f, 5.0f, 2.5f,
                                       xmin, ymin, xmax, ymax);
    auto ix1 = LineIntersection(0.0f, 2.5f, 5.0f, 2.5f, xmin, ymin, xmin, ymax);
    auto ix2 = LineIntersection(0.0f, 2.5f, 5.0f, 2.5f, xmax, ymin, xmax, ymax);
    CHECK(ix1.has_value());
    CHECK(ix2.has_value());
    CHECK_CLIPPED(clipped,
                  ix1->first, ix1->second, ix2->first, ix2->second);
  }

  // 16. Diagonal line intersecting top and left
  {
    auto clipped = ClipLineToRectangle<double>(0.0, 2.0, 3.0, 4.0,
                                               xmin, ymin, xmax, ymax);
    // left
    auto ix1 = LineIntersection<double>(
        0.0, 2.0, 3.0, 4.0, xmin, ymin, xmin, ymax);
    // top
    auto ix2 = LineIntersection<double>(
        0.0, 2.0, 3.0, 4.0, xmin, ymax, xmax, ymax);
    CHECK(ix1.has_value());
    CHECK(ix2.has_value());
    CHECK_CLIPPED(clipped, ix1->first, ix1->second, ix2->first, ix2->second);
  }
}

int main() {

  TestBresenham();
  // TestClipBresenham();
  TestClipRectangle();
  TestEmpty();
  TestWu();
  TestIntersection();
  TestVertHoriz();
  TestReflect();

  // TODO: Test point-line distance stuff.

  printf("OK, but need to manually check the Bresenham and Wu results\n");

  return 0;
}
