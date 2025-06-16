
#include <cstdint>
#include <cstdio>

#include "ansi.h"
#include "timer.h"

#include "base/do-not-optimize.h"
#include "byte-set.h"

static constexpr int OUTER = 10000;
static constexpr int INNER = 1'000'000;

static void BenchSizeSIMD() {
  double total_sec = 0.0;
  for (int i = 0; i < OUTER; i++) {
    ByteSet s;
    s.Add(i);
    s.Add(i * 3);
    s.Add(i * 129 + 11);

    Timer timer;
    for (int j = 0; j < INNER; j++) {
      DoNotOptimize(s);
      int size = s.Size();
      DoNotOptimize(size);
    }
    total_sec += timer.Seconds();
  }

  printf("Total time (SIMD): %s\n",
         ANSI::Time(total_sec).c_str());
}

static void BenchSizeASM() {
  double total_sec = 0.0;
  for (int i = 0; i < OUTER; i++) {
    ByteSet s;
    s.Add(i);
    s.Add(i * 3);
    s.Add(i * 129 + 11);

    Timer timer;
    for (int j = 0; j < INNER; j++) {
      DoNotOptimize(s);
      int size = s.SizeASM();
      DoNotOptimize(size);
    }
    total_sec += timer.Seconds();
  }

  printf("Total time (ASM): %s\n",
         ANSI::Time(total_sec).c_str());
}

template<class BS>
static void BenchByteSet(const char *which) {
  Timer timer;
  constexpr int TIMES = 10'000'000;

  for (int i = 0; i < TIMES; i++) {
    BS s;
    s.Add(i);
    s.Add(i * 151);
    s.Add(i * 299 + 137);
    for (int j = 0; j < 4; j++) {
      s.Add(i * j * 17 + 11);
    }
    DoNotOptimize(s);
    int size = s.Size();
    DoNotOptimize(size);

    for (uint8_t b : s) {
      DoNotOptimize(b);
    }

    bool a = s.Contains(0x2A) || s.Contains(0xCE);
    DoNotOptimize(a);

    BS t = BS::Singleton(0x4a);
    t.Add(0xff);
    t.Add(0x00);
    s.AddSet(t);
    DoNotOptimize(s);

    for (uint8_t b : s) {
      DoNotOptimize(b);
    }
  }

  const double total_sec = timer.Seconds();
  printf("Total time (%s): %s\n",
         which,
         ANSI::Time(total_sec).c_str());
}

int main(int argc, char **argv) {
  ANSI::Init();

  #if 0
  printf("\nBenchmarking %lld Size calls:\n",
         int64_t(OUTER) * int64_t(INNER));
  BenchSizeSIMD();
  BenchSizeASM();
  #endif

  BenchByteSet<ByteSet64>("64");
  BenchByteSet<ByteSet>("new");

  printf("OK\n");
  return 0;
}
