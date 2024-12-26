#include "hull.h"

#include <cstdio>
#include <tuple>
#include <vector>

#include "yocto_matht.h"
#include "polyhedra.h"
#include "ansi.h"

#include "base/logging.h"

using vec3 = yocto::vec<double, 3>;

static void TestTetra() {
  std::vector<vec3> tetra = {
    // bottom
    {1.0, 0.0, 0.0},
    {0.0, 1.0, 0.0},
    {1.0, 1.0, 0.0},

    {0.0, 0.0, 1.0},
  };

  std::vector<std::tuple<int, int, int>> triangles =
    QuickHull3D::Hull(tetra);
  CHECK(triangles.size() == 4);
}

static void TestCube() {
  std::vector<vec3> cube_points = {
      {-1, -1, -1}, {-1, -1, 1}, {-1, 1, -1}, {-1, 1, 1},
      {1, -1, -1}, {1, -1, 1}, {1, 1, -1}, {1, 1, 1},
  };
  std::vector<std::tuple<int, int, int>> triangles =
      QuickHull3D::Hull(cube_points);
  CHECK(triangles.size() == 12);
}

static void TestPyramid() {
  std::vector<vec3> pyramid_points = {
      {0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0}, {0.5, 0.5, 1}};
  std::vector<std::tuple<int, int, int>> triangles =
      QuickHull3D::Hull(pyramid_points);
  CHECK(triangles.size() == 6) << triangles.size();
}

// Test with a tetrahedron, including an internal point.
static void TestTetrahedronInternal() {
  std::vector<vec3> tetrahedron_points = {
    {1, 1, 1}, {-1, -1, 1}, {-1, 1, -1}, {1, -1, -1},
    // Internal point
    {0, 0, 0}
  };
  std::vector<std::tuple<int, int, int>> triangles =
      QuickHull3D::Hull(tetrahedron_points);
  CHECK(triangles.size() == 4);
  for (const auto &[a, b, c] : triangles) {
    CHECK(a != 4);
    CHECK(b != 4);
    CHECK(c != 4);
  }
}

int main(int argc, char **argv) {
  ANSI::Init();

  TestTetra();
  TestCube();
  TestPyramid();
  TestTetrahedronInternal();

  printf("OK\n");
  return 0;
}
