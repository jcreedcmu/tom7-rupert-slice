#include "gtl/top_n.h"

#include <memory>
#include <vector>

#include "base/logging.h"

static void Test() {
  // Top 10 integers under 100! You won't believe number #4!
  gtl::TopN<int> top(10);

  int a = 1;
  for (int i = 0; i < 100; i++) {
    a += 71;
    a %= 100;
    top.push(a);
  }

  std::unique_ptr<std::vector<int>> v(top.Extract());
  top.Reset();

  // Now should have 99, 98, ...
  CHECK(v->size() == 10);
  for (int i = 0; i < 10; i++) {
    CHECK((*v)[i] == 99 - i) << (*v)[i];
  }
}

int main(int argc, char **argv) {
  Test();

  printf("OK\n");
  return 0;
}

