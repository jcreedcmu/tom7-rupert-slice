
#include "big-csg.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "ansi.h"
#include "arcfour.h"
#include "base/logging.h"
#include "mesh.h"
#include "randutil.h"
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
    printf("--- %d ----\n", iters);
    for (const vec2 &v : polygon) {
      printf("  (%.17g, %.17g)\n", v.x, v.y);
    }

    TriangularMesh3D mesh = BigMakeHole(poly, polygon);
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

static void MakeHoleRegression1() {
  Polyhedron polyhedron = Cube();

  // This hole doesn't intersect any triangles.
  // We used to incorrectly ignore them, and then
  // fail to do any splitting.
  std::vector<vec2> hole = {
    {-0.5983,0.6050},
    {0.5934,0.2455},
    {0.8957,-0.7372},
  };

  TriangularMesh3D mesh = BigMakeHole(polyhedron, hole);
  // XXX check properties
}

static void MakeHoleRegression2() {
  Polyhedron polyhedron = Cube();

  // This hole doesn't intersect any triangles.
  // We used to incorrectly ignore them, and then
  // fail to do any splitting.
  std::vector<vec2> hole = {
    {-0.55155376633860409,-0.40203764978889872},
    {-0.21766304180549348,-0.2152329697590897},
    {0.10901159242057352,-0.20790275443552875},
  };

  TriangularMesh3D mesh = BigMakeHole(polyhedron, hole);
  // XXX check properties
}

static void MakeHoleRegression3() {
  Polyhedron polyhedron = Cube();
  std::vector<vec2> hole = {
    {-0.75495612684800883, 0.099498339031475855},
    {-0.73322843361013357, 0.16597406143629345},
    {0.098365546849227034, -0.097417790196595272},
  };

  TriangularMesh3D mesh = BigMakeHole(polyhedron, hole);
  // XXX check properties
}

static void MakeHoleRegression4() {
  Polyhedron polyhedron = Cube();
  std::vector<vec2> hole = {
    {0.077060999292063143, -0.75358866758849841},
    {0.30529144067361724, 0.42228231883558182},
    {0.31918058913674846, 0.44125289037604137},
  };

  TriangularMesh3D mesh = BigMakeHole(polyhedron, hole);
  // XXX check properties
}

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

  TriangularMesh3D mesh = BigMakeHole(polyhedron, hole);
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

  // MakeHoleRegression1();
  // MakeHoleRegression2();
  // MakeHoleRegression3();
  MakeHoleRegression4();

  // TestMakeHole();
  // StressTest();

  printf("OK\n");
  return 0;
}
