
#include <cstdio>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "ansi.h"
#include "solutions.h"

using Attempt = SolutionDB::Attempt;
using Solution = SolutionDB::Solution;

static void PrintAll() {

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
    printf(AWHITE("%s") ":\n", name.c_str());
    // const std::vector<Attempt> &atts = attmap[name];
    for (const Attempt &att : attmap[name]) {
      printf("  %lld iters of " ARED("%s") ", best " AORANGE("%.11g") "\n",
             att.iters, SolutionDB::MethodName(att.method),
             att.best_error);
    }

    for (const Solution &sol : solmap[name]) {
      printf("  " AGREEN("%s") " solved @" APURPLE("%.11g") "\n",
             SolutionDB::MethodName(sol.method),
             sol.ratio);
    }
  }
}

int main(int argc, char **argv) {
  ANSI::Init();

  PrintAll();

  return 0;
}
