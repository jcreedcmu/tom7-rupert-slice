
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


static void PrintAll(bool noperts,
                     bool full,
                     const std::unordered_set<std::string> &filter) {
  std::unordered_map<std::string, std::vector<Attempt>> attmap;
  std::unordered_map<std::string, std::vector<Solution>> solmap;
  std::set<std::string> names;

  int64_t all_iters = 0;

  {
    SolutionDB db;
    std::vector<Attempt> atts = db.GetAllAttempts();
    std::vector<Solution> sols = db.GetAllSolutions();

    for (const Attempt &att : atts) {
      names.insert(att.polyhedron);
      attmap[att.polyhedron].push_back(att);
    }

    for (const Solution &sol : sols) {
      names.insert(sol.polyhedron);
      solmap[sol.polyhedron].push_back(sol);
    }
  }

  for (const std::string &name : names) {
    if (!noperts && Util::StartsWith(name, "nopert_"))
      continue;

    if (filter.empty() || filter.contains(name)) {
      if (Wishlist().contains(name)) {
        printf(AYELLOW("⋆") " ");
      }
      printf(AWHITE("%s") ":\n", name.c_str());

      const std::vector<Attempt> &atts = attmap[name];

      for (const Attempt &att : atts) {
        all_iters += att.iters;
      }

      if (full) {
        for (const Attempt &att : atts) {
          printf("  " AGREY("%d.")
                 " %lld iters of %s, best " AORANGE("%.11g") "\n",
                 att.id,
                 att.iters,
                 FullMethodName(ANSI_RED, att.method, att.source).c_str(),
                 att.best_error);
        }
      } else {
        // Collate attempts by method.
        struct Att {
          int64_t total_iters = 0;
          double best_error = std::numeric_limits<double>::infinity();
          int count = 0;
        };
        std::map<int, Att> attms;
        for (const Attempt &att : atts) {
          Att &a = attms[att.method];
          a.total_iters += att.iters;
          a.best_error = std::min(a.best_error, att.best_error);
          a.count++;
        }

        for (const auto &[m, att] : attms) {
          printf("  " AGREY("%d×")
                 " %s iters of " ARED("%s")
                 ", best " AORANGE("%.11g") "\n",
                 att.count,
                 FormatNum(att.total_iters).c_str(),
                 SolutionDB::MethodName(m),
                 att.best_error);
        }
      }

      const std::vector<Solution> &sols = solmap[name];
      if (full) {
        for (const Solution &sol : sols) {
          printf("  " AGREY("%d.")
                 " %s solved @" APURPLE("%.11g")
                 " " ABLUE("%.11g") "\n",
                 sol.id,
                 FullMethodName(
                     ANSI_GREEN, sol.method, sol.source).c_str(),
                 sol.ratio,
                 sol.clearance);
        }
      } else {
        if (sols.empty()) {
          printf("  No solutions.\n");
        } else {
          // Get the best ratio and best clearance.
          Solution best_ratio = sols[0];
          Solution best_clearance = sols[0];

          for (const Solution &s : sols) {
            if (s.ratio < best_ratio.ratio) best_ratio = s;
            if (s.clearance > best_ratio.clearance) best_clearance = s;
          }

          auto Print = [](const Solution &sol, double d,
                          const char *color,
                          const char *what) {
              printf("  " AGREY("%d.")
                     " %s best %s: %s%.17g" ANSI_RESET "\n",
                     sol.id,
                     FullMethodName(
                         ANSI_GREEN, sol.method, sol.source).c_str(),
                     what, color, d);
            };

          Print(best_ratio, best_ratio.ratio, ANSI_PURPLE, "ratio");
          Print(best_clearance, best_clearance.clearance,
                ANSI_BLUE, "clearance");
        }
      }
    }
  }

  printf("Total iters: %lld\n", all_iters);
}

int main(int argc, char **argv) {
  ANSI::Init();

  bool full = false;
  bool noperts = false;
  std::string name;

  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if (arg == "-full") {
      full = true;
    } else if (arg == "-noperts") {
      noperts = true;
    } else {
      CHECK(name.empty()) << "Just one name on the command line.";
      name = arg;
    }
  }

  std::unordered_set<std::string> filter;
  if (name == "wishlist") {
    filter = Wishlist();
  } else if (!name.empty()) {
    filter = {name};
  }

  PrintAll(noperts, full, filter);

  return 0;
}
