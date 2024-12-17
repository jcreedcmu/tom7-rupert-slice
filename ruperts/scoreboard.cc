
#include "solutions.h"

#include <cstdio>
#include <vector>

#include "ansi.h"

static void PrintAll() {
  SolutionDB db;
  std::vector<SolutionDB::Solution> sols = db.GetAllSolutions();
  for (const SolutionDB::Solution &sol : sols) {
    printf(AYELLOW("%s") " via " AWHITE("%s") ":\n"
           "  time: %lld\n"
           "  ratio: %.17g\n",
           sol.polyhedron.c_str(),
           SolutionDB::MethodName(sol.method),
           sol.createdate,
           sol.ratio);
  }
}

int main(int argc, char **argv) {
  ANSI::Init();

  PrintAll();

  return 0;
}
