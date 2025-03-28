
#include <cstdio>
#include <optional>

#include "ansi.h"

#include "shrinklutions.h"

static void BestSolutions() {
  ShrinklutionDB db;
  for (int i = 2; i < 24; i++) {
    if (std::optional<ShrinklutionDB::Solution> sol =
        db.GetBestSolutionFor(i)) {
      printf(AWHITE("% 2d") ". " AGREY("r=") "%.11g "
             AGREY("d=") "%.11g\n",
             i,
             sol.value().radius,
             sol.value().radius * 2.0);
    }
  }
}

int main(int argc, char **argv) {
  ANSI::Init();

  printf("\nBest known solutions:\n");
  BestSolutions();

  return 0;
}
