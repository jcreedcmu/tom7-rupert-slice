
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "ansi.h"
#include "base/stringprintf.h"
#include "polyhedra.h"
#include "solutions.h"
#include "util.h"

using Attempt = SolutionDB::Attempt;
using Solution = SolutionDB::Solution;

std::string FullMethodName(const char *color,
                           int method,
                           int source) {
  std::string name = SolutionDB::MethodName(method);
  (void)Util::TryStripPrefix("METHOD_", &name);
  if (method == SolutionDB::METHOD_IMPROVE_RATIO) {
    return StringPrintf("%s%s" ANSI_RESET "[" AWHITE("%d") "]",
                        color, name.c_str(), source);
  } else {
    return StringPrintf("%s%s" ANSI_RESET,
                        color, name.c_str());
  }
}

static const std::unordered_set<std::string> &Wishlist() {
  static auto *s = new std::unordered_set<std::string>{
    "snubcube",
    "rhombicosidodecahedron",
    "snubdodecahedron",
    "pentagonalhexecontahedron",
    "deltoidalhexecontahedron",
  };

  return *s;
}

// "0" would be an integer in BoVeX.
static std::string Ftos(double d) {
  if (d == 0) return "0.0";
  return StringPrintf("%.17g", d);
}

static void PrintTable(const std::unordered_set<std::string> &filter) {
  std::map<std::string, std::vector<Solution>> solmap;
  std::set<std::string> names;

  printf("val solution-table =\n");

  {
    SolutionDB db;
    std::vector<Solution> sols = db.GetAllSolutions();

    for (const Solution &sol : sols) {
      names.insert(sol.polyhedron);
      solmap[sol.polyhedron].push_back(sol);
    }
  }

  for (const std::string &name : names) {
    if (Util::StartsWith(name, "nopert_"))
      continue;

    if (filter.empty() || filter.contains(name)) {

      std::string nickname = PolyhedronShortName(name);
      std::string human_name = PolyhedronHumanName(name);

      const std::vector<Solution> &sols = solmap[name];

      if (sols.empty() ||
          // Assuming the solutions are not valid if they are
          // for wishlist polyhedra
          Wishlist().contains(name)) {
        printf("  (\"%s\", NONE) ::\n",
               nickname.c_str());
      } else {

        // Get the best ratio and best clearance.
        Solution best_ratio = sols[0];
        Solution best_clearance = sols[0];

        for (const Solution &s : sols) {
          if (s.ratio < best_ratio.ratio) best_ratio = s;
          if (s.clearance > best_ratio.clearance) best_clearance = s;
        }

        printf("  (\"%s\", SOME(%s, %s)) ::\n",
               nickname.c_str(),
               Ftos(best_ratio.ratio).c_str(),
               Ftos(best_clearance.clearance).c_str());
      }
    }
  }

  printf("  nil\n");
}

int main(int argc, char **argv) {
  ANSI::Init();

  std::string name;
  std::unordered_set<std::string> filter;
  if (name == "wishlist") {
    filter = Wishlist();
  } else if (!name.empty()) {
    filter = {name};
  }

  PrintTable(filter);

  return 0;
}
