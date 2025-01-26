
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
#include "util.h"

using Nopert = SolutionDB::Nopert;

static void PrintAll() {
  SolutionDB db;

  std::vector<Nopert> noperts = db.GetAllNoperts();

  std::sort(noperts.begin(), noperts.end(),
            [](const Nopert &a, const Nopert &b) {
              if (a.vertices.size() == b.vertices.size())
                return a.id < b.id;
              return a.vertices.size() < b.vertices.size();
            });

  for (const Nopert &nopert : noperts) {
    printf(AGREY("%d.") " " AWHITE("%4d") AGREY("v")
           " via " ACYAN("%s") " (%s)\n",
           nopert.id,
           (int)nopert.vertices.size(),
           SolutionDB::NopertMethodName(nopert.method),
           Util::FormatTime("%Y-%m-%d %H:%M", nopert.createdate).c_str());
  }

  printf("\n" ABLUE("%d") " noperts.\n", (int)noperts.size());
}

int main(int argc, char **argv) {
  ANSI::Init();

  printf(AWHITE("Noperts:") "\n");
  PrintAll();

  return 0;
}
