
#include <bit>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <format>
#include <vector>

#include "ansi.h"
#include "arcfour.h"
#include "base/do-not-optimize.h"
#include "periodically.h"
#include "randutil.h"
#include "stats.h"
#include "status-bar.h"
#include "timer.h"
#include "auto-histo.h"

// Simple wrapper around std::chrono::steady_clock.
// Starts timing when constructed. TODO: Pause() etc.
struct BenchTimer {
  BenchTimer() : starttime(std::chrono::high_resolution_clock::now()) {}

  static_assert(std::chrono::high_resolution_clock::is_steady);

  double Seconds() const {
    const std::chrono::time_point<std::chrono::high_resolution_clock> stoptime =
      std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> elapsed = stoptime - starttime;
    return elapsed.count();
  }

  double MS() const {
    return Seconds() * 1000.0;
  }

  void Reset() {
    starttime = std::chrono::high_resolution_clock::now();
  }

 private:
  // morally const, but not const to allow assignment from other Timer
  // objects.
  std::chrono::time_point<std::chrono::high_resolution_clock> starttime;
};


inline bool StandardThreshold(double d) {
  return d < 1.0e-6;
}

inline bool FancyThreshold(double d) {
  static constexpr uint32_t target_exp =
      (std::bit_cast<uint64_t>(0x1.0p-20) >> 52) & 0x7FF;
  uint32_t exp = (std::bit_cast<uint64_t>(d) >> 52) & 0x7FF;
  return exp < target_exp;
}

inline constexpr int64_t NUM_ITERS = 10000;
static void TestStandard(ArcFour *rc,
                         std::vector<double> *samples) {
  double total = 0.0;
  for (int64_t i = 0; i < NUM_ITERS; i++) {
    double d = RandDouble(rc);
    if (rc->Byte() & 1) d = -d;

    BenchTimer timer;
    bool x = StandardThreshold(d);
    DoNotOptimize(x);
    total += timer.Seconds();
  }

  samples->push_back(total / NUM_ITERS);
}


static void TestFancy(ArcFour *rc,
                      std::vector<double> *samples) {
  double total = 0.0;
  for (int64_t i = 0; i < NUM_ITERS; i++) {
    double d = RandDouble(rc);
    if (rc->Byte() & 1) d = -d;

    BenchTimer timer;
    bool x = FancyThreshold(d);
    DoNotOptimize(x);
    total += timer.Seconds();
  }

  samples->push_back(total / NUM_ITERS);
}


static constexpr int NUM_LOOPS = 100000;
static void RandomInterleave() {
  ArcFour rc("bench");
  Periodically status_per(1.0);
  StatusBar status(1);

  std::vector<double> discard;
  TestStandard(&rc, &discard);
  TestFancy(&rc, &discard);

  std::vector<double> standard, fancy;
  for (int i = 0; i < NUM_LOOPS; i++) {
    if (rc.Byte() & 1) {
      TestStandard(&rc, &standard);
    } else {
      TestFancy(&rc, &fancy);
    }

    status_per.RunIf([&]() {
        status.Progressf(i, NUM_LOOPS, "benchmarking");
      });
  }

  auto GS = [](const char *what, const std::vector<double> &samples) {
      const Stats::Gaussian g = Stats::EstimateGaussian(samples);
      double lo = g.mean - g.PlusMinus95();
      double hi = g.mean + g.PlusMinus95();

      printf("Method: %s\n"
             "Mean time: [%s, %s]\n",
             what,
             ANSI::Time(lo).c_str(),
             ANSI::Time(hi).c_str());

      AutoHisto histo;
      for (double d : samples) {
        histo.Observe(d);
      }

      printf("%s\n", histo.UnlabeledHoriz(70).c_str());

      /*
      return std::format("[" AYELLOW("{:.12g}") ", "
                         AYELLOW("{:.12g}") "]",
                         lo, hi);
      */
    };

  GS("standard", standard);
  GS("fancy", fancy);
}


int main(int argc, char **argv) {
  ANSI::Init();

  RandomInterleave();

  return 0;
};
