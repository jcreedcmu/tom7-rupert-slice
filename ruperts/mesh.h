#ifndef _RUPERTS_MESH_H
#define _RUPERTS_MESH_H

#include <vector>
#include <tuple>

#include "yocto_matht.h"

// A 3D triangle mesh; not necessarily convex (or even connected).
struct TriangularMesh3D {
  using vec3 = yocto::vec<double, 3>;
  std::vector<vec3> vertices;
  std::vector<std::tuple<int, int, int>> triangles;
};

struct Mesh3D {
  using vec3 = yocto::vec<double, 3>;
  std::vector<vec3> vertices;
  // Not necessarily oriented
  std::vector<std::vector<int>> faces;
};


#endif
