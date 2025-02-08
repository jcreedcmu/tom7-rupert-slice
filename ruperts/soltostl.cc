
// Generates STL of the residue from a solution.

#include "ansi.h"

#include <cstdio>
#include <format>
#include <string_view>
#include <utility>
#include <vector>
#include <string>

#include "base/stringprintf.h"
#include "polyhedra.h"
#include "rendering.h"
#include "solutions.h"
#include "yocto_matht.h"
#include "csg.h"
#include "big-csg.h"

using frame3 = yocto::frame<double, 3>;

using Solution = SolutionDB::Solution;

static std::pair<frame3, frame3> GetSolution(std::string_view name) {
  SolutionDB db;
  Solution sol = db.GetBestSolutionFor(name);
  return std::make_pair(sol.outer_frame, sol.inner_frame);
}

static void RenderAny(std::string_view name) {
  Polyhedron polyhedron = PolyhedronByName(name);
  const auto &[outer_frame, inner_frame] = GetSolution(name);

  Polyhedron outer = Rotate(polyhedron, outer_frame);

  Polyhedron inner = Rotate(polyhedron, inner_frame);
  Mesh2D sinner = Shadow(inner);
  std::vector<int> hull = QuickHull(sinner.vertices);

  std::vector<vec2> polygon;
  polygon.reserve(hull.size());
  for (int i : hull) polygon.push_back(sinner.vertices[i]);
  // XXX test
  // for (vec2 &v : polygon) v *= 0.5;
  Mesh3D residue = BigMakeHole(outer, polygon);

  std::string filename = std::format("{}-residue.stl", name);
  SaveAsSTL(residue, filename);
  printf("Wrote %s\n", filename.c_str());
}

int main(int argc, char **argv) {
  ANSI::Init();

  CHECK(argc == 2) << "./soltostl.exe polyhedronname";

  RenderAny(argv[1]);

  return 0;
}
