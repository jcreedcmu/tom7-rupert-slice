
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

static void TestMakeHole() {
  Polyhedron polyhedron = Cube();

  // A small triangular hole, not in any special
  // position (e.g. not on the cube's face diagonal).
  std::vector<vec2> hole = {
    {-0.11, -0.44},
    {+0.33, -0.20},
    {0.0, 0.25},
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

  printf("OK\n");
  return 0;
}
