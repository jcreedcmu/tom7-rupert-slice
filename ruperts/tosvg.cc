
// Even though this is called "tosvg", it just generates PNGs.
// That was enough for me to debug my problem!

#include "ansi.h"

#include <string_view>
#include <utility>
#include <vector>
#include <string>

#include "base/stringprintf.h"
#include "polyhedra.h"
#include "rendering.h"
#include "solutions.h"
#include "yocto_matht.h"

using frame3 = yocto::frame<double, 3>;

using Solution = SolutionDB::Solution;

std::pair<frame3, frame3> GetSolution(std::string_view name) {
  SolutionDB db;
  std::vector<Solution> sols = db.GetSolutionsFor(std::string(name));
  CHECK(!sols.empty()) << "There are no solutions for " << name;
  return std::make_pair(sols[0].outer_frame, sols[0].inner_frame);
}

static void RenderAny(std::string_view name) {
  Polyhedron polyhedron = PolyhedronByName(name);
  const auto &[outer_frame, inner_frame] = GetSolution(name);

  Polyhedron outer = Rotate(polyhedron, outer_frame);
  Mesh2D souter = Shadow(outer);

  Polyhedron inner = Rotate(polyhedron, inner_frame);
  Mesh2D sinner = Shadow(inner);

  {
    Rendering rendering(polyhedron, 3840 * 2, 2160 * 2);
    rendering.RenderMesh(souter);
    rendering.DarkenBG();

    rendering.RenderMesh(sinner);
    rendering.RenderBadPoints(sinner, souter);

    rendering.Save(StringPrintf("tosvg-%s.png", std::string(name).c_str()));
  }

  std::vector<int> outer_hull = QuickHull(souter.vertices);
  std::vector<int> inner_hull = QuickHull(sinner.vertices);

  {
    Rendering rendering(polyhedron, 3840 * 2, 2160 * 2);
    rendering.RenderHullDistance(souter, outer_hull);
    rendering.RenderHull(souter, outer_hull, 0x0000FFFF);
    rendering.RenderHull(sinner, inner_hull, 0x00FF0077);
    rendering.Save(StringPrintf("tosvg-%s-hulls.png",
                                std::string(name).c_str()));
  }
}


int main(int argc, char **argv) {
  ANSI::Init();

  CHECK(argc == 2) << "./tosvg.exe polyhedronname";

  RenderAny(argv[1]);

  return 0;
}
