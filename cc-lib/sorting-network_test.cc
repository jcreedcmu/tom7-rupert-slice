
#include "sorting-network.h"

#include <array>
#include <tuple>

#include "ansi.h"
#include "base/logging.h"

static void TestArray() {
  {
    std::array<int, 3> a{5, 3, 1};
    for (int i = 0; i < 2; i++) {
      FixedSort<3>(&a);
      CHECK(a[0] == 1);
      CHECK(a[1] == 3);
      CHECK(a[2] == 5);
    }
  }

  {
    std::array<int, 3> a{2, 3, 2};
    for (int i = 0; i < 2; i++) {
      FixedSort<3>(&a);
      CHECK(a[0] == 2);
      CHECK(a[1] == 2);
      CHECK(a[2] == 3);
    }
  }
}

static void TestTuple() {
  {
    std::tuple<int, int, int> a = std::make_tuple(3, 5, 1);
    for (int i = 0; i < 2; i++) {
      FixedSort<3>(&a);
      CHECK(a == std::make_tuple(1, 3, 5));
    }
  }

  {
    std::tuple<int, int, int> a = std::make_tuple(3, 2, 2);
    for (int i = 0; i < 2; i++) {
      FixedSort<3>(&a);
      CHECK(a == std::make_tuple(2, 2, 3));
    }
  }
}


int main(int argc, char **argv) {
  ANSI::Init();

  TestArray();
  TestTuple();

  printf("OK\n");
  return 0;
}
