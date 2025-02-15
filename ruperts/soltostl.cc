
// Generates STL of the residue from a solution.

#include <cstdio>
#include <format>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "ansi.h"
#include "base/stringprintf.h"
#include "big-csg.h"
#include "mesh.h"
#include "polyhedra.h"
#include "rendering.h"
#include "solutions.h"
#include "timer.h"
#include "util.h"
#include "yocto_matht.h"

using frame3 = yocto::frame<double, 3>;

using Solution = SolutionDB::Solution;

inline constexpr bool USE_CLEARANCE = true;

static Solution GetSolution(std::string_view name) {
  SolutionDB db;
  return db.GetBestSolutionFor(name, USE_CLEARANCE);
}

static void RenderPNG(const Polyhedron &polyhedron,
                      const std::pair<frame3, frame3> &frames,
                      std::string_view name) {
  const auto &[outer_frame, inner_frame] = frames;

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

    rendering.Save(StringPrintf("soltostl-%s.png", std::string(name).c_str()));
  }

  std::vector<int> outer_hull = QuickHull(souter.vertices);
  std::vector<int> inner_hull = QuickHull(sinner.vertices);

  {
    Rendering rendering(polyhedron, 3840 * 2, 2160 * 2);
    rendering.RenderHullDistance(souter, outer_hull);
    rendering.RenderHull(souter, outer_hull, 0x0000FFFF);
    rendering.RenderHull(sinner, inner_hull, 0x00FF0077);
    rendering.Save(StringPrintf("soltostl-%s-hulls.png",
                                std::string(name).c_str()));
  }
}


static void RenderAny(std::string_view name) {
  Timer timer;
  Polyhedron polyhedron = PolyhedronByName(name);

  const Solution &sol = GetSolution(name);
  const auto &outer_frame = sol.outer_frame;
  const auto &inner_frame = sol.inner_frame;
  printf("Rendering solution " AYELLOW("%d") " from " ACYAN("%s") "\n",
         sol.id,
         Util::FormatTime("%d %b %Y - %H:%M:%S", sol.createdate).c_str());

  std::optional<double> ratio =
    GetRatio(polyhedron, outer_frame, inner_frame);
  std::optional<double> clearance =
    GetClearance(polyhedron, outer_frame, inner_frame);

  if (!ratio.has_value() || !clearance.has_value()) {
    printf(ARED("Not valid!!") " Solution #%d\n", sol.id);
    return;
  }

  RenderPNG(polyhedron, std::make_pair(outer_frame, inner_frame),
            name);

  Polyhedron outer = Rotate(polyhedron, outer_frame);

  Polyhedron inner = Rotate(polyhedron, inner_frame);
  Mesh2D sinner = Shadow(inner);
  std::vector<int> hull = QuickHull(sinner.vertices);

  std::vector<vec2> polygon;
  polygon.reserve(hull.size());
  for (int i : hull) polygon.push_back(sinner.vertices[i]);
  TriangularMesh3D residue = BigMakeHole(outer, polygon);

  std::string filename = std::format("{}-residue.stl", name);
  SaveAsSTL(residue, filename);
  printf("Took %s. Wrote %s\n",
         ANSI::Time(timer.Seconds()).c_str(), filename.c_str());
}

int main(int argc, char **argv) {
  ANSI::Init();

  CHECK(argc == 2) << "./soltostl.exe polyhedronname";

  RenderAny(argv[1]);

  return 0;
}
