
#include "mesh.h"

#include <cstdio>
#include <cstdlib>
#include <format>
#include <tuple>
#include <vector>

#include "ansi.h"
#include "base/logging.h"
#include "yocto_matht.h"

using vec3 = yocto::vec<double, 3>;

double IsNear(double a, double b) {
  return std::abs(a - b) < 0.0000001;
}

#define CHECK_NEAR(f, g) do {                                           \
  const double fv = (f);                                                \
  const double gv = (g);                                                \
  const double e = std::abs(fv - gv);                                   \
  CHECK(e < 0.0000001) << "Expected " << #f << " and " << #g <<         \
    " to be close, but got: " <<                                        \
    std::format("{:.17g} and {:.17g}, with err {:.17g}", fv, gv, e);    \
  } while (0)

static void VolumeOfCube() {

    //                  +y
    //      a------b     | +z
    //     /|     /|     |/
    //    / |    / |     0--- +x
    //   d------c  |
    //   |  |   |  |
    //   |  e---|--f
    //   | /    | /
    //   |/     |/
    //   h------g

  std::vector<vec3> vertices;
  auto Vertex = [&vertices](double x, double y, double z) {
      int idx = vertices.size();
      vertices.push_back(vec3{.x = x, .y = y, .z = z});
      return idx;
    };

  const int a = Vertex(0.0, 1.0, 1.0);
  const int b = Vertex(1.0, 1.0, 1.0);
  const int c = Vertex(1.0, 1.0, 0.0);
  const int d = Vertex(0.0, 1.0, 0.0);

  const int e = Vertex(0.0, 0.0, 1.0);
  const int f = Vertex(1.0, 0.0, 1.0);
  const int g = Vertex(1.0, 0.0, 0.0);
  const int h = Vertex(0.0, 0.0, 0.0);

  std::vector<std::tuple<int, int, int>> triangles;
  // top
  triangles.emplace_back(a, b, d);
  triangles.emplace_back(b, c, d);
  // right
  triangles.emplace_back(c, b, g);
  triangles.emplace_back(b, f, g);
  // left
  triangles.emplace_back(a, d, h);
  triangles.emplace_back(h, e, a);
  // front
  triangles.emplace_back(d, c, h);
  triangles.emplace_back(h, c, g);
  // back
  triangles.emplace_back(b, a, e);
  triangles.emplace_back(e, f, b);
  // bottom
  triangles.emplace_back(h, f, e);
  triangles.emplace_back(g, f, h);

  TriangularMesh3D cube{.vertices = vertices, .triangles = triangles};

  CHECK_NEAR(MeshVolume(cube), 1.0);

  OrientMesh(&cube);

  CHECK_NEAR(MeshVolume(cube), 1.0);
}


int main(int argc, char **argv) {
  ANSI::Init();

  VolumeOfCube();

  printf("OK\n");
  return 0;
}
