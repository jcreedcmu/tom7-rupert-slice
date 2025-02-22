#ifndef _RUPERTS_MESH_H
#define _RUPERTS_MESH_H

#include <string_view>
#include <vector>
#include <tuple>

#include "yocto_matht.h"

// A 3D triangle mesh; not necessarily convex. Some code expects that
// faces be consistently oriented, that it be a proper manifold, or
// connected, etc.
struct TriangularMesh3D {
  using vec3 = yocto::vec<double, 3>;
  std::vector<vec3> vertices;
  std::vector<std::tuple<int, int, int>> triangles;
};

// Mesh with polygonal faces.
struct Mesh3D {
  using vec3 = yocto::vec<double, 3>;
  std::vector<vec3> vertices;
  // Not necessarily oriented
  std::vector<std::vector<int>> faces;
};

TriangularMesh3D LoadSTL(std::string_view filename);

void SaveAsSTL(const TriangularMesh3D &mesh, std::string_view filename,
               std::string_view name = "");

// Orients triangles to have a consistent winding order. The input
// must be a connected manifold, since this is how we determine what
// that order should be!
void OrientMesh(TriangularMesh3D *mesh);

#endif
