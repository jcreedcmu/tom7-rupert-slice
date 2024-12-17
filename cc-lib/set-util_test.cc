
#include "set-util.h"

#include <unordered_set>
#include <vector>

#include "ansi.h"
#include "base/logging.h"
#include "hashing.h"

static void TestToSortedVec() {
  std::unordered_set<int64_t> s;
  s.insert(333);
  s.insert(444);
  s.insert(555);
  s.insert(666);

  const auto v = SetToSortedVec(s);
  CHECK(v.size() == 4);
  CHECK(v[0] == 333);
  CHECK(v[1] == 444);
  CHECK(v[2] == 555);
  CHECK(v[3] == 666);
}

static void TestToSortedVecCustom() {
  std::unordered_set<std::vector<int>, Hashing<std::vector<int>>> s;

  s.insert({1, 2, 3});
  s.insert({5, 4, 3, 2});
  s.insert({9});

  const auto v = SetToSortedVec(s, [](const auto &a, const auto &b) {
      return a.size() < b.size();
    });
  CHECK(v.size() == 3);
  CHECK(v[0].size() == 1);
  CHECK(v[1].size() == 3);
  CHECK(v[2].size() == 4);
}

int main(int argc, char **argv) {
  ANSI::Init();

  TestToSortedVec();
  TestToSortedVecCustom();

  printf("OK\n");
  return 0;
}
