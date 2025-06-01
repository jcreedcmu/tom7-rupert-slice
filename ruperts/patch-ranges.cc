#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "ansi.h"
#include "auto-histo.h"
#include "base/stringprintf.h"
#include "factorization.h"
#include "nd-solutions.h"
#include "patches.h"
#include "periodically.h"
#include "status-bar.h"
#include "threadutil.h"
#include "timer.h"
#include "util.h"

static void Histo() {
  PatchInfo patch_info = LoadPatchInfo("scube-patchinfo.txt");

  std::vector<std::pair<uint64_t, uint64_t>> pairs;
  for (const auto &[ocode, opatch] : patch_info.canonical) {
    for (const auto &[icode, ipatch] : patch_info.canonical) {
      pairs.emplace_back(ocode, icode);
    }
  }

  Periodically status_per(1.0);
  StatusBar status_bar(1);

  std::mutex mu;

  std::vector<double> all_scores;

  std::atomic<int> done = 0;

  ParallelApp(
      pairs,
      [&](const auto &outer_inner) {
        const auto &[ocode, icode] = outer_inner;

        std::vector<double> scores;

        {
          std::string filename = TwoPatchFilename(ocode, icode);
          NDSolutions<6> nds(filename);
          scores.reserve(nds.Size());

          for (int64_t i = 0; i < nds.Size(); i++) {
            const auto &[key, score, outer_, inner_] = nds[i];
            scores.push_back(score);
          }
        }

        size_t score_size = 0;
        {
          MutexLock ml(&mu);
          all_scores.reserve(all_scores.size() + scores.size());
          for (double d : scores) all_scores.push_back(d);
          score_size = all_scores.size();
        }

        done++;

        status_per.RunIf([&]() {
            /*
            status_bar.Printf("%s",
                              all_histo.SimpleANSI(30).c_str());
            */
            status_bar.Progress(done.load(), pairs.size(),
                                "[{}] Loading...", score_size);
          });
      }, 4);


  Timer sort_timer;
  status_bar.Print("Sorting {} scores.", all_scores.size());
  std::sort(all_scores.begin(), all_scores.end());
  status_bar.Print("Sorted in {}", ANSI::Time(sort_timer.Seconds()));

  AutoHisto all_histo(100000);
  // Do stripes so that we get well-allocated buckets.
  int64_t stride = all_scores.size() / 1000;
  // Make sure it is relatively prime.
  do {
    stride = Factorization::NextPrime(stride);
  } while (all_scores.size() % stride == 0);

  status_bar.Print("Stride size {} for {} scores",
                   stride, all_scores.size());
  int64_t a = 0;
  for (int64_t i = all_scores.size(); i--;) {
    all_histo.Observe(all_scores[a]);
    a += stride;
    if (a >= all_scores.size()) a -= all_scores.size();
  }

  // Generate quantiles file.
  std::string out;
  for (int i = 0; i < 16384; i++) {
    int64_t index =
      std::clamp(
          (int64_t)((i / 16383.0) * double(all_scores.size() - 1)),
          int64_t{0},
          (int64_t)all_scores.size() - 1);

    double p = all_scores[index];
    AppendFormat(&out, "{:.17g}\n", p);
  }

  Util::WriteFile("scube-score-quantiles.txt", out);
  status_bar.Printf("Wrote quantiles.\n");

  if (all_scores.empty()) {
    printf(ARED("Empty!") "\n");
    return;
  }

  printf("\n\n\n");
  printf("All min: %.17g\n"
         "All max: %.17g\n"
         "Final histo:\n%s\n",
         all_scores.front(),
         all_scores.back(),
         all_histo.SimpleANSI(54).c_str());
}

int main(int argc, char **argv) {
  ANSI::Init();

  Histo();

  return 0;
}
