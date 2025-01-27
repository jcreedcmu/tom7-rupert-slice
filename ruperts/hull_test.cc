#include "arcfour.h"
#include "randutil.h"
#include "hull.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <tuple>
#include <vector>

#include "yocto_matht.h"
#include "polyhedra.h"
#include "ansi.h"
#include "hull3d.h"

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

static void TestOctahedronRegression1() {
  std::vector<vec3> vertices{
    {-1.0000, 0.0000, 0.0000},
    {0.0000, -1.0000, 0.0000},
    {0.0335, 0.0128, 0.0709},
    {0.0000, 0.0000, -1.0000},
    {0.0000, 1.0000, 0.0000},
    {0.0000, 0.0000, 1.0000},
    {1.0000, 0.0000, 0.0000},
  };

  std::vector<int> hull = Hull3D::HullPoints(vertices);
  if (hull.size() != 6) {
    printf("Hull is wrong size (%d):\n", (int)hull.size());
    for (int i : hull) {
      printf("  #%d. %s\n", i, VecString(vertices[i]).c_str());
    }
    CHECK(hull.size() == 6);
  }
  for (int i : hull) {
    CHECK(i != 2) << "This point is definitely not in the hull!";
  }
}

static void TestOctahedron() {
  ArcFour rc("octahedron");
  static constexpr int VERBOSE = 0;

  const double sqrt6 = std::sqrt(6.0);
  for (int passes = 0; passes < 50; passes++) {
    std::vector<vec3> vertices;
    for (double s : {-1.0, 1.0}) {
      vertices.emplace_back(s, 0.0, 0.0);
      vertices.emplace_back(0.0, s, 0.0);
      vertices.emplace_back(0.0, 0.0, s);
    }

    // Now a bunch of random vertices strictly within the inscribed
    // sphere.
    for (int i = 0; i < 10000; i++) {
      vec3 v = normalize(vec3(RandDouble(&rc), RandDouble(&rc), RandDouble(&rc)));
      vertices.push_back((v * RandDouble(&rc) * 0.9999999) / sqrt6);
    }

    Shuffle(&rc, &vertices);

    if (VERBOSE > 1) {
      printf("Hull of:\n");
      for (int i = 0; i < vertices.size(); i++) {
        printf("  #%d: %s\n", i, VecString(vertices[i]).c_str());
      }
    }

    std::vector<int> hull = Hull3D::HullPoints(vertices);

    if (hull.size() != 6) {
      printf("Hull should be the octahedron, but got %d points:\n",
             (int)hull.size());
      for (int i : hull) {
        printf("  #%d: vertex %d, %s\n",
               i, hull[i], VecString(vertices[hull[i]]).c_str());
      }
      LOG(FATAL) << "It's wrong";
    }

    for (int i : hull) {
      const vec3 &v = vertices[i];
      if (std::abs(v.x) == 1.0) {
        CHECK(v.y == 0.0 && v.z == 0.0);
      } else if (std::abs(v.y) == 1.0) {
        CHECK(v.x == 0.0 && v.z == 0.0);
      } else if (std::abs(v.z) == 1.0) {
        CHECK(v.x == 0.0 && v.y == 0.0);
      } else {
        LOG(FATAL) << i << " not an octahedron vertex: " << VecString(v);
      }
    }

    if (VERBOSE) printf("ok\n");
  }
}

int main(int argc, char **argv) {
  ANSI::Init();

  TestTetra();
  TestCube();
  TestPyramid();
  TestTetrahedronInternal();
  TestOctahedronRegression1();
  TestOctahedron();

  printf("OK\n");
  return 0;
}
