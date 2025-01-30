
#include "atomic-util.h"

#include <vector>
#include <thread>

#include "base/logging.h"
#include "ansi.h"

// Always group 8 counters together, as this is much more efficient.
// But here I want to test different arities of the macros.
DECLARE_COUNTERS(bytes, lines, errors);
DECLARE_COUNTERS(last);

DECLARE_COUNTERS(one, two, three, four, five, six, seven, eight);

// Must go first!
static void TestCountersSimple() {
  CHECK(bytes.Read() == 0);
  CHECK(lines.Read() == 0);
  CHECK(errors.Read() == 0);
  lines++;

  CHECK(bytes.Read() == 0);
  CHECK(lines.Read() == 1);
  CHECK(errors.Read() == 0);

  bytes += 3;

  CHECK(bytes.Read() == 3);
  CHECK(lines.Read() == 1);
  CHECK(errors.Read() == 0);

  lines++;
  lines += 0;

  CHECK(bytes.Read() == 3);
  CHECK(lines.Read() == 2);
  CHECK(errors.Read() == 0);

  errors++;

  CHECK(bytes.Read() == 3);
  CHECK(lines.Read() == 2);
  CHECK(errors.Read() == 1);

  lines.Reset();

  CHECK(bytes.Read() == 3);
  CHECK(lines.Read() == 0);
  CHECK(errors.Read() == 1);

  lines++;

  CHECK(bytes.Read() == 3);
  CHECK(lines.Read() == 1);
  CHECK(errors.Read() == 1);

  CHECK(one.Read() == eight.Read());
}

static void TestThreaded() {
  bytes.Reset();
  lines.Reset();
  errors.Reset();

  static constexpr int NUM_THREADS = 48;

  std::vector<std::thread> ths;
  for (int i = 0; i < NUM_THREADS; i++) {
    ths.emplace_back([]() {
        last++;
        for (int j = 0; j < 10000; j++) {
          bytes++;
        }
        errors++;
        for (int j = 0; j < 1000; j++) {
          lines++;
        }
        for (int j = 0; j < 1000; j++) {
          bytes++;
        }
        last++;
      });
  }

  for (auto &th : ths)
    th.join();

  CHECK(bytes.Read() == 11000 * NUM_THREADS);
  CHECK(lines.Read() == 1000 * NUM_THREADS);
  CHECK(errors.Read() == NUM_THREADS);
  CHECK(last.Read() == 2 * NUM_THREADS);
}

int main(int argc, char **argv) {
  ANSI::Init();

  TestCountersSimple();

  TestThreaded();

  printf("OK\n");
  return 0;
}
