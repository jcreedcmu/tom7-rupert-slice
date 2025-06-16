
#include "byte-set.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <format>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "ansi.h"
#include "arcfour.h"
#include "base/logging.h"
#include "base/stringprintf.h"
#include "interval-cover.h"
#include "randutil.h"

static std::string ByteSetString(const ByteSet &s) {
  IntervalCover<int> cover(0);
  std::string ret;
  for (uint8_t b : s) {
    cover.SetPoint(b, 1);
    AppendFormat(&ret, " {:02x}", b);
  }
  AppendFormat(&ret, " (which is:");
  for (uint64_t pt = cover.First(); !cover.IsAfterLast(pt);
       pt = cover.Next(pt)) {
    IntervalCover<int>::Span s = cover.GetPoint(pt);
    if (s.data > 0) {
      AppendFormat(&ret, " {:02x}" AGREY("-") "{:02x}",
                   (int)s.start, (int)s.end - 1);
    }
  }
  AppendFormat(&ret, ")");

  return ret;
}

static std::string ByteSet64String(const ByteSet64 &bs) {
  switch (bs.type) {
  case ByteSet64::EMPTY: {
    std::string ret = "EMPTY ";
    for (int i = 0; i < 7; i++) {
      AppendFormat(&ret, "{:02x}", bs.payload[i]);
    }
    return ret;
  }
  case ByteSet64::VALUES: {
    std::string ret = "VALUES ";
    for (int i = 0; i < 7; i++) {
      AppendFormat(&ret, " {:02x}", bs.payload[i]);
    }
    return ret;
  }
  case ByteSet64::RANGES: {
    std::string ret = "RANGES ";
    for (int i = 0; i < 6; i += 2) {
      AppendFormat(&ret, " {:02x}" AGREY("×") "{:02x}",
                   bs.payload[i],
                   bs.payload[i + 1]);
    }
    AppendFormat(&ret, " ({:02x})", bs.payload[6]);
    return ret;
  }
  default:
    return "INVALID";
  }
}

#define CHECK_EQUAL_SETS(s, s64) do {               \
    for (int x = 0; x < 256; x++) {                 \
      CHECK(s.Contains(x) == s64.Contains(x)) <<    \
        "Wrong on " << x << ":\nSet:\n" <<          \
        ByteSetString(s) << "\nS64:\n" <<           \
        ByteSet64String(s64);                       \
    }                                               \
    CHECK(s.Size() == s64.Size());                  \
  } while (0)

// s <= s64
#define CHECK_SUPERSET(s, s64) do {                 \
    for (int x = 0; x < 256; x++) {                 \
      if (s.Contains(x)) {                          \
      CHECK(s64.Contains(x)) << "Wrong on " <<      \
        x << ":\nSet:\n" << ByteSetString(s) <<     \
        "\nS64:\n" << ByteSet64String(s64);         \
      }                                             \
    }                                               \
  } while (0)


static void TestByteSetBasic() {
  {
    ByteSet s;
    CHECK(s.Empty());
    // Mostly check that these compile.
    CHECK(s == s);
    CHECK(!(s < s));
    CHECK(!(s > s));
    CHECK(s >= s);
    CHECK(s <= s);
    CHECK(!(s != s));
    for (int i = 0; i < 256; i++) {
      CHECK(!s.Contains(i));
    }
    CHECK(s.begin() == s.end());

    s.Add(42);
    CHECK(!s.Empty());
    for (int i = 0; i < 256; i++) {
      if (s.Contains(i)) {
        CHECK(i == 42) << i;
      }
    }
    CHECK(s.begin() != s.end());
    CHECK(*s.begin() == 42);
    auto it = s.begin();
    ++it;
    CHECK(it == s.end());
  }
}

static void TestByteSetMultiple() {
  ByteSet s;
  CHECK_EQ(s.Size(), 0);
  s.Add(10);
  CHECK_EQ(s.Size(), 1) << s.Size();
  s.Add(20);
  CHECK_EQ(s.Size(), 2) << s.Size();
  s.Add(255);
  CHECK_EQ(s.Size(), 3);
  CHECK(s.Contains(10));
  CHECK(s.Contains(20));
  CHECK(s.Contains(255));
  CHECK(!s.Contains(15));

  auto it = s.begin();
  CHECK(it != s.end());
  CHECK(*it == 10);

  ++it;
  CHECK(it != s.end());
  CHECK(*it == 20);

  ++it;
  CHECK(it != s.end());
  CHECK(*it == 255);

  ++it;
  CHECK(it == s.end());
}

static void TestByteSet64Empty() {
  ByteSet empty_set;
  ByteSet64 bs64(empty_set);
  CHECK_EQ(bs64.type, ByteSet64::EMPTY);
}

static void TestByteSet64Singleton() {
  for (int b = 0; b < 256; b++) {
    ByteSet singleton_set;
    singleton_set.Add(b);
    ByteSet64 bs64(singleton_set);
    // Should be able to represent all singletons exactly.
    CHECK_EQ(bs64.type, ByteSet64::VALUES);
    CHECK(bs64.Contains(b));
    for (int i = 0; i < 256; i++) {
      if (bs64.Contains(i)) {
        CHECK(i == b);
      }
    }
  }
}

static void TestByteSetComparison() {
  ByteSet bs1;
  bs1.Add(0x42);
  bs1.Add(0xFF);

  ByteSet bs2;
  bs2.Add(0xFF);

  CHECK(bs1 != bs2);
  CHECK((bs1 < bs2) ^ (bs1 > bs2));
  CHECK((bs1 <= bs2) ^ (bs1 >= bs2));
  CHECK((bs1 < bs2) ^ (bs1 >= bs2));
  CHECK((bs1 <= bs2) ^ (bs1 > bs2));
  CHECK(!(bs1 == bs2));

  bs2.Add(0x42);
  CHECK(bs1 == bs2);
  CHECK(!(bs1 < bs2));
  CHECK(!(bs1 > bs2));
  CHECK(!(bs1 != bs2));
  CHECK(bs1 <= bs2);
  CHECK(bs1 >= bs2);
}

static void TestByteSet64Comparison() {
  {
    ByteSet64 bs1;
    bs1.Add(0x42);
    bs1.Add(0xFF);

    ByteSet64 bs2;
    bs2.Add(0xFF);

    CHECK(bs1 != bs2);
    CHECK((bs1 < bs2) ^ (bs1 > bs2));
    CHECK((bs1 <= bs2) ^ (bs1 >= bs2));
    CHECK((bs1 < bs2) ^ (bs1 >= bs2));
    CHECK((bs1 <= bs2) ^ (bs1 > bs2));
    CHECK(!(bs1 == bs2));

    bs2.Add(0x42);
    CHECK(bs1 == bs2);
    CHECK(!(bs1 < bs2));
    CHECK(!(bs1 > bs2));
    CHECK(!(bs1 != bs2));
    CHECK(bs1 <= bs2);
    CHECK(bs1 >= bs2);
  }

  // Now compare some different representations of the
  // same set.
  {
    ByteSet64 a, b, c;
    a.Set(ByteSet64::VALUES, 1, 80, 90, 90, 90, 90, 90);
    b.Set(ByteSet64::VALUES, 80, 90, 1, 1, 1, 1, 1);
    c.Set(ByteSet64::VALUES, 90, 1, 1, 1, 1, 1);
    CHECK(a == b);
    CHECK(a != c);
    CHECK(c != b);
    CHECK((c < b) ^ (c > b));
    CHECK((b < c) ^ (b > c));

    c.Set(ByteSet64::RANGES, 1, 1, 80, 1, 90, 1);
    for (int i = 0; i < 256; i++) {
      CHECK(b.Contains(i) == c.Contains(i)) << i;
    }
    CHECK(b == c) << ByteSet64String(b) << " " << ByteSet64String(c);
    CHECK(a == c);

    b.Set(ByteSet64::VALUES, 80, 1, 90, 80, 1, 1, 1);
    CHECK(b == c);
    CHECK(c == b);
    CHECK(a == b);
  }


  {
    ByteSet64 a, b, c;
    a.Set(ByteSet64::VALUES, 0xFE, 0xFF, 0x00, 0x01, 0x42, 0x42, 0x42);
    b.Set(ByteSet64::RANGES, 0xFE, 4, 0x42, 1, 0x00, 0);
    c.Set(ByteSet64::RANGES, 0xFE, 3, 0x42, 1, 0x00, 0);
    for (int i = 0; i < 256; i++) {
      CHECK(a.Contains(i) == b.Contains(i)) << i << "\n" <<
        ByteSet64String(a) << " " << ByteSet64String(b);
    }
    CHECK(a == b);
    CHECK(a != c);
    CHECK(c != b);
    CHECK((c < b) ^ (c > b));
    CHECK((b < c) ^ (b > c));
  }


}

static void TestByteSet64OneMissing() {
  for (int b = 0; b < 256; b++) {
    ByteSet almost_set;
    for (int i = 0; i < 256; i++) {
      if (i != b) almost_set.Add(i);
    }
    ByteSet64 bs64(almost_set);

    auto Error = [&]() {
        return std::format("\nb {:02x}.\n"
                            "set:\n{}\n"
                            "set64:\n{}\n",
                            b,
                            ByteSetString(almost_set),
                            ByteSet64String(bs64));
      };

    // Should be able to represent these exactly using ranges.
    CHECK_EQ(bs64.type, ByteSet64::RANGES) << Error();
    CHECK_EQUAL_SETS(almost_set, bs64);

    // Now make sure we can close any final gap, since ranges
    // of length 255 are a somewhat special case.
    bs64.Add(b);
    CHECK(bs64.Contains(b));
    CHECK(bs64.Size() == 256);
  }
}

static void TestByteSet64Pair() {
  for (uint8_t b1 : { 0, 1, 2, 42, 99, 254, 255 }) {
    for (int b2 = 0; b2 < 256; b2++) {
      ByteSet s;
      s.Add(b1);
      // note b2 may equal b1.
      s.Add(b2);
      ByteSet64 bs64(s);
      // Should be able to represent all pairs exactly.
      CHECK_EQ(bs64.type, ByteSet64::VALUES);
      auto Error = [&]() {
          return std::format("\nb1 {:02x} b2 {:02x}.\n"
                              "set:\n{}\n"
                              "set64:\n{}\n",
                              b1, b2,
                              ByteSetString(s),
                              ByteSet64String(bs64));
        };
      CHECK(bs64.Contains(b1)) << b1 << Error();
      CHECK(bs64.Contains(b2)) << b2 << Error();
      for (int i = 0; i < 256; i++) {
        if (bs64.Contains(i)) {
          CHECK(i == b1 || i == b2);
        }
      }
    }
  }
}

static void TestByteSet64Universal() {
  ByteSet top;
  for (int i = 0; i < 256; i++) top.Add(i);
  ByteSet64 top64(top);
  CHECK(top64.Size() == 256);
}

static void TestByteSet64Intervals() {
  ArcFour rc("ivals");

  for (int i = 0; i < 100000; i++) {
    int n = 1 + RandTo(&rc, 5);

    ByteSet s;
    while (n--) {
      uint8_t start = rc.Byte();
      uint8_t len = rc.Byte();
      for (int b = 0; b < len; b++) {
        s.Add((start + b) & 0xFF);
      }
    }

    ByteSet64 s64(s);
    CHECK_SUPERSET(s, s64);
  }
}

static void TestByteSet64Add() {
  {
    ByteSet64 bs;
    for (int i = 0; i < 10; i++) {
      bs.Add(i);
      CHECK(bs.Contains(i)) << i << "\n" << ByteSet64String(bs);
    }
    // It cannot fit as individuals, and is not empty.
    CHECK(bs.type == ByteSet64::RANGES);
  }

  {
    ByteSet64 bs;
    for (int i = 0; i < 256; i += 2) {
      bs.Add(i);
      CHECK(bs.Contains(i));
    }

    for (int i = 0; i < 256; i += 2) {
      CHECK(bs.Contains(i));
    }
  }

  ArcFour rc("add");
  for (int iter = 0; iter < 20000; iter++) {
    ByteSet64 s;
    int n = RandTo(&rc, 7);
    std::vector<std::pair<int, int>> ranges;
    for (int i = 0; i < n; i++) {
      ranges.emplace_back(rc.Byte(), rc.Byte());
    }

    for (const auto &[start, len] : ranges) {
      for (int v = 0; v < len; v++) {
        s.Add(start + v);
        CHECK(s.Contains(start + v));
      }
    }

    for (const auto &[start, len] : ranges) {
      for (int v = 0; v < len; v++) {
        CHECK(s.Contains(start + v));
      }
    }
  }


}

#define CHECK_IS(bs, typ, p0, p1, p2, p3, p4, p5, p6) do {    \
    CHECK((typ) == bs.type);                                  \
    std::array<uint8_t, 7> a = {p0, p1, p2, p3, p4, p5, p6};  \
    for (int i = 0; i < 7; i++) {                             \
      CHECK(bs.payload[i] == a[i]) << ByteSet64String(bs);    \
    }                                                         \
  } while (0)

// These test heuristics for expanding ranges. The abstract type
// does not guarantee that you get a specific result, so these
// might break with future improvements.
static void TestByteSet64AddHeuristics() {
  // Test extending ranges on one side or the other.
  {
    ByteSet64 bs;
    bs.Set(ByteSet64::RANGES, 0x01, 9, 0xA0, 1, 0xB0, 1);
    CHECK(!bs.Contains(0xA1));
    // Extend second range upward.
    bs.Add(0xA1);
    CHECK_IS(bs, ByteSet64::RANGES, 0x01, 9, 0xA0, 2, 0xB0, 1, 0);
    // Extend third range downward.
    bs.Add(0xAF);
    CHECK_IS(bs, ByteSet64::RANGES, 0x01, 9, 0xA0, 2, 0xAF, 2, 0);
    // Already there; no-op.
    bs.Add(0xA0);
    CHECK_IS(bs, ByteSet64::RANGES, 0x01, 9, 0xA0, 2, 0xAF, 2, 0);
    // Not close to the first range, but it's still the best choice.
    bs.Add(0x30);
    CHECK_IS(bs, ByteSet64::RANGES, 0x01, 48, 0xA0, 2, 0xAF, 2, 0);
    // Extend first range downward, touching zero.
    bs.Add(0x00);
    CHECK_IS(bs, ByteSet64::RANGES, 0x00, 49, 0xA0, 2, 0xAF, 2, 0);

    // TODO: Test wraparound by adding FF or something close here.
    // But this behavior is not yet implemented!

    constexpr int A = 137;
    constexpr int B = 11;
    static_assert(std::gcd(A, 256) == 1);
    uint8_t x = 42;
    for (int i = 0; i < 256; i++) {
      bs.Add(x);
      CHECK(bs.Contains(x));
      x = A * x + B;
    }

    // There are multiple possible representations here, because
    // we don't yet fuse ranges. So just check that it is the
    // universal set.
    CHECK(bs.Size() == 256);
    for (int i = 0; i < 256; i++) {
      CHECK(bs.Contains(i));
    }
  }

}

static void TestMap() {
  {
    ByteSet s;
    s.Add(10);
    s.Add(10);
    s.Add(20);
    s.Add(255);

    auto add_one = [](uint8_t x) { return x + 1; };
    ByteSet mapped = s.Map(add_one);

    ByteSet expected;
    expected.Add(11);
    expected.Add(21);
    expected.Add(0);
    CHECK(expected == mapped);
  }

  {
    ByteSet s = ByteSet::Top();
    ByteSet mapped = s.Map([](uint8_t x){ return x; });
    CHECK(s == mapped);
  }
}

static void TestByteSetUnionIntersection() {
  {
    ByteSet s1;
    s1.Add(10);
    s1.Add(20);

    ByteSet s2;
    s2.Add(20);
    s2.Add(30);

    ByteSet union_set = ByteSet::Union(s1, s2);
    CHECK(union_set.Contains(10));
    CHECK(union_set.Contains(20));
    CHECK(union_set.Contains(30));
    CHECK_EQ(union_set.Size(), 3);

    ByteSet intersection_set = ByteSet::Intersection(s1, s2);
    CHECK(!intersection_set.Contains(10));
    CHECK(intersection_set.Contains(20));
    CHECK(!intersection_set.Contains(30));
    CHECK_EQ(intersection_set.Size(), 1);
  }

  {
    ByteSet s1;
    ByteSet s2;
    CHECK(ByteSet::Union(s1, s2).Empty());
    CHECK(ByteSet::Intersection(s1, s2).Empty());
  }

  {
    ByteSet s1 = ByteSet::Singleton(10);
    ByteSet s2;

    CHECK(ByteSet::Union(s1, s2) == s1);
    CHECK(ByteSet::Intersection(s1, s2) == s2);
  }

  {
    ByteSet s1 = ByteSet::Top();
    CHECK_EQ(s1.Size(), 256);
    ByteSet s2;
    s2.Add(10);
    CHECK(ByteSet::Union(s1, s2) == s1);
    CHECK(ByteSet::Intersection(s1, s2) == s2);
  }
}

static void TestByteSetClear() {
  ByteSet s;
  s.Add(10);
  s.Add(20);
  s.Add(100);
  s.Add(129);
  s.Add(195);
  CHECK(!s.Empty());
  s.Clear();
  CHECK(s.Empty());
  CHECK_EQ(s.Size(), 0);
  for (int i = 0; i < 256; i++) {
    CHECK(!s.Contains(i));
  }
}

static void TestByteSetSingleton() {
  for (int i = 0; i < 256; i++) {
    ByteSet s = ByteSet::Singleton(i);
    CHECK_EQ(s.Size(), 1);
    CHECK(s.Contains(i));
    CHECK_EQ(s.GetSingleton(), i);
    for (int j = 0; j < 256; j++) {
      if (i != j) {
        CHECK(!s.Contains(j));
      }
    }
  }
}

static void TestByteSetIterator() {
  ArcFour rc("byteset");
  for (int stride = 1; stride < 250; stride <<= 1) {
    ByteSet s;
    std::vector<uint8_t> expected_values;
    for (int i = 0; i < 256; i += stride) {
      expected_values.push_back(i);
    }

    // Add them in a random order.
    std::vector<uint8_t> shuffled = expected_values;
    Shuffle(&rc, &shuffled);
    for (int i : shuffled) s.Add(i);

    CHECK_EQ(s.Size(), expected_values.size());

    // But iteration should always happen in sorted order.
    std::vector<uint8_t> actual_values;
    for (uint8_t v : s) {
      actual_values.push_back(v);
    }

    CHECK(actual_values == expected_values);
  }
}

static void TestByteSetConstructor() {
  ByteSet b({0x00, 0x01, 0x02, 0x03});

  CHECK(b.Size() == 4);
  CHECK(!b.Contains(0xa2));
}

int main(int argc, char **argv) {
  ANSI::Init();

  TestByteSetBasic();
  TestByteSetMultiple();
  TestByteSetComparison();
  TestByteSetUnionIntersection();
  TestByteSetClear();
  TestByteSetSingleton();
  TestByteSetIterator();
  TestByteSetConstructor();

  TestByteSet64Empty();
  TestByteSet64Singleton();
  TestByteSet64Pair();
  TestByteSet64Universal();
  TestByteSet64OneMissing();
  TestByteSet64Intervals();
  TestByteSet64Comparison();

  TestByteSet64Add();
  TestByteSet64AddHeuristics();

  TestMap();

  printf("OK\n");
  return 0;
}
