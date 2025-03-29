
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
#include "arcfour.h"
#include "base/stringprintf.h"
#include "polyhedra.h"
#include "smallest-sphere.h"
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
  std::map<std::string, std::set<int>> soltypes;

  printf("datatype method = ");
  for (int i = SolutionDB::FIRST_METHOD;
       i <= SolutionDB::LAST_METHOD;
       i++) {
    if (i != SolutionDB::FIRST_METHOD) {
      printf(" | ");
    }
    printf("%s", SolutionDB::MethodName(i));
  }
  printf("\n");

  printf("fun method-eq (m1, m2) =\n"
         "  case (m1, m2) of\n");
  for (int i = SolutionDB::FIRST_METHOD;
       i <= SolutionDB::LAST_METHOD;
       i++) {
    printf ("    | (%s, %s) => true\n",
            SolutionDB::MethodName(i),
            SolutionDB::MethodName(i));
  }
  printf("    | _ => false\n\n");

  printf("val solution-table =\n");


  {
    SolutionDB db;
    std::vector<Solution> sols = db.GetAllSolutions();

    for (const Solution &sol : sols) {
      names.insert(sol.polyhedron);
      solmap[sol.polyhedron].push_back(sol);

      if (sol.ratio < 0.999999 &&
          sol.clearance > 0.000001) {
        soltypes[sol.polyhedron].insert(sol.method);
      }
    }
  }

  for (const std::string &name : names) {
    if (Util::StartsWith(name, "nopert_"))
      continue;

    Polyhedron poly = PolyhedronByName(name);

    if (filter.empty() || filter.contains(name)) {

      std::string nickname = PolyhedronShortName(name);
      std::string human_name = PolyhedronHumanName(name);

      int vertices = (int)poly.vertices.size();
      int faces = (int)poly.faces->v.size();
      int edges = 0;
      for (const std::vector<int> &face : poly.faces->v) {
        edges += (int)face.size();
      }
      // Each edge is on exactly two faces.
      CHECK(edges % 2 == 0);
      edges >>= 1;

      const std::vector<Solution> &sols = solmap[name];

      std::vector<std::string> st;
      for (int m : soltypes[name]) {
        st.push_back(SolutionDB::MethodName(m));
      }
      st.push_back("nil");

      printf("  (\"%s\", %d, %d, %d, ",
             nickname.c_str(), vertices, edges, faces);

      if (sols.empty() ||
          // Assuming the solutions are not valid if they are
          // for wishlist polyhedra
          Wishlist().contains(name)) {
        printf("NONE");
      } else {

        // Get the best ratio and best clearance.
        Solution best_ratio = sols[0];
        Solution best_clearance = sols[0];

        for (const Solution &s : sols) {
          if (s.ratio < best_ratio.ratio) best_ratio = s;
          if (s.clearance > best_ratio.clearance) best_clearance = s;
        }

        // Normalize the clearance so that it doesn't depend on the
        // polyhedron's scale.
        ArcFour rc("table");
        const auto &[center, radius] =
          SmallestSphere::Smallest(&rc, poly.vertices);
        double norm_clearance = best_clearance.clearance / radius;
        printf("SOME(%s, %s, %s)",
               Ftos(best_ratio.ratio).c_str(),
               Ftos(norm_clearance).c_str(),
               Util::Join(st, " :: ").c_str());
      }

      printf(") ::\n");
    }

    delete poly.faces;
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
