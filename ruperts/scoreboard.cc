
#include <cstdio>
#include <set>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "ansi.h"
#include "base/stringprintf.h"
#include "solutions.h"

using Attempt = SolutionDB::Attempt;
using Solution = SolutionDB::Solution;

std::string FullMethodName(const char *color,
                           int method,
                           int source) {
  if (method == SolutionDB::METHOD_IMPROVE) {
    return StringPrintf("%s%s" ANSI_RESET "[" AWHITE("%d") "]",
                        color, SolutionDB::MethodName(method), source);
  } else {
    return StringPrintf("%s%s" ANSI_RESET,
                        color, SolutionDB::MethodName(method));
  }
}

static const std::unordered_set<std::string> &Wishlist() {
  static auto *s = new std::unordered_set<std::string>{
    "snubcube",
    "rhombicosidodecahedron",
    "snubdodecahedron",
    "pentagonalhexecontahedron,"
    "deltoidalhexecontahedron,"
  };

  return *s;
}


static void PrintAll(const std::unordered_set<std::string> &filter) {

  std::unordered_map<std::string, std::vector<Attempt>> attmap;
  std::unordered_map<std::string, std::vector<Solution>> solmap;
  std::set<std::string> names;

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
    if (filter.empty() || filter.contains(name)) {
      if (Wishlist().contains(name)) {
        printf(AYELLOW("⋆") " ");
      }
      printf(AWHITE("%s") ":\n", name.c_str());
      // const std::vector<Attempt> &atts = attmap[name];
      for (const Attempt &att : attmap[name]) {
        printf("  " AGREY("%d.")
               " %lld iters of %s, best " AORANGE("%.11g") "\n",
               att.id,
               att.iters,
               FullMethodName(ANSI_RED, att.method, att.source).c_str(),
               att.best_error);
      }

      for (const Solution &sol : solmap[name]) {
        printf("  " AGREY("%d.")
               " %s solved @" APURPLE("%.11g") "\n",
               sol.id,
               FullMethodName(ANSI_GREEN, sol.method, sol.source).c_str(),
               sol.ratio);
      }
    }
  }
}

int main(int argc, char **argv) {
  ANSI::Init();

  if (argc > 1) {
    std::string name = argv[1];
    if (name == "wishlist") {
      PrintAll(Wishlist());
    } else {
      PrintAll({name});
    }
  } else {
    PrintAll({});
  }

  return 0;
}
