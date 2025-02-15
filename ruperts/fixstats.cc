
// Recomputes stats for each solution.

#include <cstdio>
#include <optional>
#include <string>
#include <vector>

#include "ansi.h"
#include "base/stringprintf.h"
#include "polyhedra.h"
#include "solutions.h"
#include "timer.h"
#include "util.h"

using Attempt = SolutionDB::Attempt;
using Solution = SolutionDB::Solution;

static void FixAll() {
  Timer timer;
  SolutionDB db;
  std::vector<Solution> sols = db.GetAllSolutions();

  for (const Solution &sol : sols) {
    const auto &outer_frame = sol.outer_frame;
    const auto &inner_frame = sol.inner_frame;

    if (Util::StartsWith(sol.polyhedron, "nopert_")) {
      printf("Skip #%d " AWHITE("%s") "\n",
             sol.id, sol.polyhedron.c_str());
      continue;
    }

    // PERF: Maybe should avoid recomputing these?
    Polyhedron poly = PolyhedronByName(sol.polyhedron);

    std::optional<double> ratio = GetRatio(poly, outer_frame, inner_frame);
    std::optional<double> clearance =
      GetClearance(poly, outer_frame, inner_frame);

    if (!ratio.has_value() || !clearance.has_value()) {
      printf(ARED("Not valid!!") " Solution #%d\n", sol.id);
    } else {
      printf("Solution %d:\n"
             "  ratio %.17g -> %.17g\n"
             "  clear %.17g -> %.17g\n",
             sol.id,
             sol.ratio, ratio.value(),
             sol.clearance, clearance.value());

      db.ExecuteAndPrint(
          StringPrintf(
              "update solutions "
              "set ratio = %.17g, clearance = %.17g "
              "where id = %d",
              ratio.value(), clearance.value(), sol.id));
    }

    delete poly.faces;
  }

  printf("Finished in %s\n", ANSI::Time(timer.Seconds()).c_str());
}


int main(int argc, char **argv) {
  ANSI::Init();

  FixAll();

  return 0;
}
