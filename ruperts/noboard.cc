
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "ansi.h"
#include "base/stringprintf.h"
#include "solutions.h"
#include "util.h"

using Solution = SolutionDB::Solution;
using Nopert = SolutionDB::Nopert;

static void PrintAll() {
  SolutionDB db;

  std::vector<Solution> solutions = db.GetAllNopertSolutions();
  std::unordered_map<int, int> solved;
  for (const Solution &sol : solutions) {
    std::string_view poly = sol.polyhedron;
    CHECK(Util::TryStripPrefix("nopert_", &poly)) << sol.polyhedron;
    printf("%s\n", std::string(poly).c_str());
    solved[atoi(std::string(poly).c_str())]++;
  }

  std::vector<Nopert> noperts = db.GetAllNoperts();

  std::sort(noperts.begin(), noperts.end(),
            [](const Nopert &a, const Nopert &b) {
              if (a.vertices.size() == b.vertices.size())
                return a.id < b.id;
              return a.vertices.size() < b.vertices.size();
            });

  for (const Nopert &nopert : noperts) {
    printf(AGREY("%d.") " " AWHITE("%4d") AGREY("v")
           " via " ACYAN("%s") " (%s)",
           nopert.id,
           (int)nopert.vertices.size(),
           SolutionDB::NopertMethodName(nopert.method),
           Util::FormatTime("%Y-%m-%d %H:%M", nopert.createdate).c_str());

    int nsol = solved[nopert.id];
    if (nsol > 0) {
      printf("  " ARED("✘") " solved");
    }
    printf("\n");
  }

  printf("\n" ABLUE("%d") " noperts.\n", (int)noperts.size());
}

int main(int argc, char **argv) {
  ANSI::Init();

  printf(AWHITE("Noperts:") "\n");
  PrintAll();

  return 0;
}
