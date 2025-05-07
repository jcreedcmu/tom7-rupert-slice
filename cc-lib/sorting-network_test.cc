
#include "sorting-network.h"

#include <cstddef>
#include <array>
#include <tuple>
#include <format>

#include "ansi.h"
#include "base/logging.h"
#include "arcfour.h"
#include "randutil.h"

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

template<size_t N>
struct TestN {
  void operator()() {
    if constexpr (N <= 1) return;

    ArcFour rc(std::format("test{}", N));
    for (int i = 0; i < 100; i++) {
      std::array<int, N> a;
      for (int j = 0; j < (int)N; j++) {
        a[j] = RandTo(&rc, N * 2);
      }
      std::array<int, N> a_in = a;
      FixedSort<N>(&a);

      for (int j = 1; j < (int)N; j++) {
        if (!(a[j - 1] <= a[j])) {
          printf("Input:\n");
          for (int m = 0; m < (int)N; m++) {
            printf("  %d\n", a_in[m]);
          }
          printf("Output:\n");
          for (int m = 0; m < (int)N; m++) {
            printf("  %d\n", a[m]);
          }
          LOG(FATAL) << "Not Sorted.\nSize " << N << " iter " << i;
        }
      }
    }
  }
};

template<std::size_t N, template<size_t> class F>
inline void CountDown() {
  if constexpr (N == 0) {
    return;
  } else {
    TestN<N> test;
    test();
    CountDown<N - 1, F>();
  }
}

int main(int argc, char **argv) {
  ANSI::Init();

  TestArray();
  TestTuple();

  CountDown<MAX_FIXED_SORT_SUPPORTED, TestN>();

  printf("OK\n");
  return 0;
}
