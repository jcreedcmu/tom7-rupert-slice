
#include "interval-cover-util.h"

#include "ansi.h"
#include "util.h"

namespace {
struct A {
  int x = 0;
  int y = 0;
  /*
  bool operator==(const A &other) {
    return x == other.x && y == other.y;
  }
  */
};

static inline bool operator==(const A &a, const A &b) {
  return a.x == b.x && a.y == b.y;
}
}

static std::string ToString(const A &a) {
  return StringPrintf("%d|%d", a.x, a.y);
}

static A FromString(std::string_view s) {
  std::vector<std::string> v = Util::Tokenize(s, '|');
  CHECK(v.size() == 2);
  return A{.x = Util::stoi(v[0]), .y = Util::stoi(v[1])};
}

static void RoundTrip() {
  IntervalCover<A> ic(A{.x = 777, .y = 999});

  ic.SetSpan(12345, 5000000000, A{.x = 111, .y = 958});

  std::string s = IntervalCoverUtil::ToString<A>(ic, ToString);

  IntervalCover<A> oc = IntervalCoverUtil::FromString<A>(
      s, FromString, A{.x = 666, .y = 666});

  std::string t = IntervalCoverUtil::ToString<A>(oc, ToString);
  CHECK(s == t) << s << "\n---- vs ----\n" << t;

  CHECK(ic == oc);
}


int main(int argc, char **argv) {
  ANSI::Init();

  RoundTrip();

  printf("OK\n");
  return 0;
}
