
#include <cstdio>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>

#include "ansi.h"
#include "base/logging.h"
#include "solutions.h"
#include "util.h"

using Solution = SolutionDB::Solution;

static void Import(const std::string &dst_file, const std::string &src_file) {
  SolutionDB src(src_file.c_str());
  SolutionDB dst(dst_file.c_str());

  // Best scores in dst.
  std::unordered_map<std::string, Solution> dst_lowest_ratio;
  std::unordered_map<std::string, Solution> dst_highest_clearance;
  std::tie(dst_lowest_ratio, dst_highest_clearance) = dst.BestSolutions();

  std::unordered_set<std::string> dst_has_solution;
  for (const auto &[k, v] : dst_lowest_ratio) dst_has_solution.insert(k);
  for (const auto &[k, v] : dst_highest_clearance) dst_has_solution.insert(k);

  std::unordered_map<std::string, Solution> src_lowest_ratio;
  std::unordered_map<std::string, Solution> src_highest_clearance;
  std::tie(src_lowest_ratio, src_highest_clearance) = src.BestSolutions();

  int improved_rat = 0, improved_clear = 0;
  auto MaybeInsert = [&](const std::string &s, const Solution &sol) {
      if (Util::StartsWith(s, "nopert")) return;

      bool add = false;
      if (!dst_has_solution.contains(s)) {
        printf("Newly solved! " AWHITE("%s") "\n",
               s.c_str());
        add = true;
      } else {
        double orat = dst_lowest_ratio[sol.polyhedron].ratio;
        bool better_rat = sol.ratio < orat;
        double oclear = dst_highest_clearance[sol.polyhedron].clearance;
        bool better_clear = sol.clearance > oclear;
        if (better_rat || better_clear) {
          printf("Would add solution to " AWHITE("%s") "\n"
                 "  with ratio %.17g → %.17g" ANSI_RESET "%s\n"
                 "   and clearance %.17g → %.17g" ANSI_RESET "%s\n",
                 sol.polyhedron.c_str(),
                 orat, sol.ratio, better_rat ? "  " AGREEN("lower!") : "",
                 oclear, sol.clearance, better_clear ? "  " APURPLE("higher!") : "");
          add = true;
          if (better_rat) improved_rat++;
          if (better_clear) improved_clear++;
        }
      }

      if (add) {
        dst.AddSolution(
            sol.polyhedron,
            sol.outer_frame,
            sol.inner_frame,
            sol.method,
            // Don't copy source id, since it refers to a different db
            0,
            sol.ratio, sol.clearance);
      }
    };

  for (const auto &[name, sol] : src_lowest_ratio)
    MaybeInsert(name, sol);
  for (const auto &[name, sol] : src_highest_clearance)
    MaybeInsert(name, sol);

  printf("%d improved ratios. %d improved clearance.\n",
         improved_rat, improved_clear);

}

int main(int argc, char **argv) {
  CHECK(argc == 3) << "./mergedatabase dst.sqlite src.sqlite";

  Import(argv[1], argv[2]);

  return 0;
}
