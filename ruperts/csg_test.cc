
#include "csg.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numbers>
#include <string>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>

#include "ansi.h"
#include "arcfour.h"
#include "base/logging.h"
#include "base/stringprintf.h"
#include "bounds.h"
#include "image.h"
#include "rendering.h"
#include "yocto_matht.h"
#include "polyhedra.h"
#include "periodically.h"

using vec2 = yocto::vec<double, 2>;

static inline double IsNear(double a, double b) {
  return std::abs(a - b) < 0.0000001;
}

#define CHECK_NEAR(f, g) do {                                           \
  const double fv = (f);                                                \
  const double gv = (g);                                                \
  const double e = std::abs(fv - gv);                                   \
  CHECK(e < 0.0000001) << "Expected " << #f << " and " << #g <<         \
    " to be close, but got: " <<                                        \
    StringPrintf("%.17g and %.17g, with err %.17g", fv, gv, e);         \
  } while (0)

static void StressTest() {
  Polyhedron poly = Cube();
  ArcFour rc("stress");
  Periodically status_per(1.0);
  for (int iters = 0; true; iters++) {
    std::vector<vec2> vertices;
    for (int i = 0; i < 3; i++) {
      vec2 p{RandDouble(&rc) * 1.8 - 0.90, RandDouble(&rc) * 1.8 - 0.90};
      vertices.push_back(p);
    }

    std::vector<int> hull = QuickHull(vertices);

    std::vector<vec2> polygon;
    for (int i : hull) polygon.push_back(vertices[i]);

    Mesh3D mesh = MakeHole(poly, polygon);
    if (status_per.ShouldRun()) {
      printf("%d iters\n", iters);
    }
  }
}

/*
  regression TODO:
  (-0.6368,0.2133)
  (-0.3222,-0.0660)
  (-0.2862,-0.1969)
  (-0.3380,-0.5656)
  (-0.6061,-0.0610)
*/

static void TestMakeHole() {
  Polyhedron polyhedron = Cube();

  // A small rectangular hole, not in any special
  // position (e.g. not on the cube's face diagonal).

  /*
  std::vector<vec2> hole = {
    {-0.22, -0.33},
    {-0.27, 0.36},
    {0.34, 0.31},
    {0.29, -0.35},
  };
  */

  std::vector<vec2> hole = {
    {-0.5983,0.6050},
    {0.5934,0.2455},
    {0.8957,-0.7372},
  };

  Mesh3D mesh = MakeHole(polyhedron, hole);
  // XXX check properties
  printf("Mesh vertices:\n");
  for (int i = 0; i < mesh.vertices.size(); i++) {
    const vec3 &v = mesh.vertices[i];
    printf("  #%d. %s\n", i, VecString(v).c_str());
  }
  printf("Triangles:\n");
  for (const auto &[a, b, c] : mesh.triangles) {
    printf("  %d %d %d\n", a, b, c);
  }
}

int main(int argc, char **argv) {
  ANSI::Init();
  printf("\n");

  TestMakeHole();
  // StressTest();

  printf("OK\n");
  return 0;
}
