
#include "map-util.h"

#include <map>
#include <unordered_map>
#include <string>
#include <cstdint>

#include "base/logging.h"

static void TestCountMap() {
  {
    std::unordered_map<std::string, int> m;
    m["goodbye"] = 50;
    m["hello"] = 100;
    m["a"] = 3;

    std::vector<std::pair<std::string, int>> v =
      CountMapToDescendingVector(m);
    CHECK(v.size() == 3);
    CHECK(v[0].first == "hello");
    CHECK(v[0].second == 100);
    CHECK(v[2].second == 3);
  }

  {
    std::map<char, int64_t> m;
    m['g'] = 50;
    m['h'] = 100;
    m['a'] = 3;

    std::vector<std::pair<char, int64_t>> v =
      CountMapToDescendingVector(m);
    CHECK(v.size() == 3);
    CHECK(v[0].first == 'h');
    CHECK(v[0].second == 100);
    CHECK(v[2].second == 3);
  }
}

static void TestSortMap() {
  std::unordered_map<uint16_t, std::string> m;
  m[0x1234] = "good";
  m[0x0002] = "ok";
  m[0x8888] = "best";

  std::vector<std::pair<uint16_t, std::string>> v =
    MapToSortedVec(m);
  CHECK(v.size() == 3);
  CHECK(v[0].first == 0x0002);
  CHECK(v[0].second == "ok");
  CHECK(v[1].first == 0x1234);
  CHECK(v[1].second == "good");
  CHECK(v[2].first == 0x8888);
  CHECK(v[2].second == "best");
}


int main(int argc, char **argv) {
  TestSortMap();
  TestCountMap();

  printf("OK\n");
  return 0;
}
