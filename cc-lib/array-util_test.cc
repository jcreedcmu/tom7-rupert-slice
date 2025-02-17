
#include "array-util.h"

#include <cstdio>
#include <array>

#include "ansi.h"
#include "base/logging.h"

static void TestSubArray() {
  std::array<int, 5> arr{3, 5, 7, 9, 1234};

  const auto &[b, c] = SubArray<1, 2>(arr);
  CHECK(b == 5);
  CHECK(c == 7);
}

int main(int argc, char **argv) {
  ANSI::Init();

  TestSubArray();

  printf("OK\n");
  return 0;
}
