
#include "interval-cover-util.h"

#include "ansi.h"
#include "util.h"

namespace {
struct A {
  int x = 0;
  int y = 0;
};

[[maybe_unused]]
static inline bool operator==(const A &a, const A &b) {
  return a.x == b.x && a.y == b.y;
}
}

static std::string AToString(const A &a) {
  return StringPrintf("%d|%d", a.x, a.y);
}

static A AFromString(std::string_view s) {
  std::vector<std::string> v = Util::Tokenize(s, '|');
  CHECK(v.size() == 2);
  return A{.x = Util::stoi(v[0]), .y = Util::stoi(v[1])};
}

static void RoundTripGeneric() {
  IntervalCover<A> ic(A{.x = 777, .y = 999});

  ic.SetSpan(12345, 5000000000, A{.x = 111, .y = 958});

  std::string s = IntervalCoverUtil::Serialize<A>(ic, AToString);

  IntervalCover<A> oc = IntervalCoverUtil::Parse<A>(
      s, AFromString, A{.x = 666, .y = 666});

  std::string t = IntervalCoverUtil::Serialize<A>(oc, AToString);
  CHECK(s == t) << s << "\n---- vs ----\n" << t;

  CHECK(ic == oc);
}

static void RoundTripBool() {
  IntervalCover<bool> ivals(false);

  ivals.SetPoint(7, true);
  ivals.SetSpan(1000, 2000, true);
  ivals.SetPoint(1009, false);

  std::string s = IntervalCoverUtil::ToString(ivals);

  IntervalCover<bool> ovals = IntervalCoverUtil::ParseBool(s);

  std::string t = IntervalCoverUtil::ToString(ivals);
  CHECK(s == t) << s << "\n---- vs ----\n" << t;

  CHECK(ivals == ovals);
}

static void RoundTripEmpty() {
  IntervalCover<bool> ivals(false);
  CHECK(IntervalCover<bool>::IsAfterLast(ivals.GetPoint(0).end));
  std::string s = IntervalCoverUtil::ToString(ivals);
  IntervalCover<bool> ovals = IntervalCoverUtil::ParseBool(s);
  CHECK(IntervalCover<bool>::IsAfterLast(ovals.GetPoint(0).end));
  std::string t = IntervalCoverUtil::ToString(ivals);
  CHECK(s == t) << s << "\n---- vs ----\n" << t;
  CHECK(ivals == ovals);
}

int main(int argc, char **argv) {
  ANSI::Init();

  RoundTripBool();
  RoundTripEmpty();
  RoundTripGeneric();

  printf("OK\n");
  return 0;
}
