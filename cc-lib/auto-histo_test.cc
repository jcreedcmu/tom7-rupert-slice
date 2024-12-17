#include "auto-histo.h"

#include <cstdio>

#include "base/logging.h"

static void TestHisto() {
  AutoHisto histo(5);
  for (int i = 0; i < 7; i++) histo.Observe(i * 100);
  CHECK(!histo.Empty());
  CHECK(histo.NumSamples() == 7);
  CHECK(histo.IsIntegral());

  // We could test more about the values, but since the bucketing
  // strategy is unspecified, it's hard to get particularly detailed.
  // We could do an inspection-based test?
  AutoHisto::Histo h = histo.GetHisto(5);
  CHECK(h.min < h.max);
  CHECK(h.min_value <= h.max_value);
}


int main(int argc, char **argv) {
  ANSI::Init();

  TestHisto();

  printf("OK\n");
  return 0;
}
