
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "ansi.h"
#include "solutions.h"
#include "util.h"

using Solution = SolutionDB::Solution;
using Nopert = SolutionDB::Nopert;
using NopertAttempt = SolutionDB::NopertAttempt;

static void PrintAll() {
  SolutionDB db;

  int ok = 0;
  int smallest = 99999;
  std::vector<Solution> solutions = db.GetAllNopertSolutions();
  std::unordered_map<int, int> solved;
  for (const Solution &sol : solutions) {
    std::string_view poly = sol.polyhedron;
    CHECK(Util::TryStripPrefix("nopert_", &poly)) << sol.polyhedron;
    // printf("%s\n", std::string(poly).c_str());
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
    printf(AGREY("% 3d.") " " AWHITE("%4d") AGREY("v")
           " via " ACYAN("%s") " (%s)",
           nopert.id,
           (int)nopert.vertices.size(),
           SolutionDB::NopertMethodName(nopert.method),
           Util::FormatTime("%Y-%m-%d %H:%M", nopert.createdate).c_str());

    int nsol = solved[nopert.id];
    if (nsol > 0) {
      printf("  " ARED("✘") " solved");
    } else {
      ok++;
      smallest = std::min(smallest, (int)nopert.vertices.size());
    }
    printf("\n");
  }

  printf("\n" ABLUE("%d") " noperts; " AGREEN("%d") " not rejected.\n"
         "Smallest: " AWHITE("%d") " verts.\n",
         (int)noperts.size(), ok,
         smallest);
}

static void PrintAttempts() {
  SolutionDB db;

  std::map<int, int64_t> attempts_by_method;
  std::map<int, int64_t> attempts_by_points;

  int64_t attempts_all = 0;
  std::vector<NopertAttempt> atts = db.GetAllNopertAttempts();
  for (const NopertAttempt &att : atts) {
    printf(AGREY("% 3d.") " " AWHITE("%d") AGREY("v") " via "
           ACYAN("%s") " " AYELLOW("%lld") " attempts\n",
           att.id,
           att.points,
           SolutionDB::NopertMethodName(att.method),
           att.attempts);
    attempts_by_method[att.method] += att.attempts;
    attempts_by_points[att.points] += att.attempts;
    attempts_all += att.attempts;
  }

  printf("\n" AWHITE("By method") ":\n");
  for (const auto &[method, num] : attempts_by_method) {
    printf("  " ACYAN("%s") ": " AYELLOW("%lld") "\n",
           SolutionDB::NopertMethodName(method), num);
  }

  printf("\n" AWHITE("By points") ":\n");
  for (const auto &[pts, num] : attempts_by_points) {
    printf("  " AWHITE("%d") ": " AYELLOW("%lld") "\n",
           pts, num);
  }

  printf("\n" AWHITE("Total") ": %lld\n", attempts_all);
}

int main(int argc, char **argv) {
  ANSI::Init();

  if (argc >= 2 && (std::string)argv[1] == "--attempts") {
    printf(AWHITE("Nopert attempts:") "\n");
    PrintAttempts();
  } else {
    printf(AWHITE("Noperts:") "\n");
    PrintAll();
  }

  return 0;
}
