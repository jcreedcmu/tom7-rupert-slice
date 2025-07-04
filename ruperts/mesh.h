#ifndef _RUPERTS_MESH_H
#define _RUPERTS_MESH_H

#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>
#include <tuple>

#include "hashing.h"
#include "image.h"
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
               std::string_view name = "", bool quiet = false);

// TODO: Facetize TriangularMesh3D into Mesh3D?

// Creates a mesh that is simply the concatenation of the argument meshes.
// The geometry is not merged (not even duplicate vertices). This can
// be used to save several objects in the same STL file, for example.
TriangularMesh3D ConcatMeshes(const std::vector<TriangularMesh3D> &meshes);

// Orients triangles to have a consistent winding order. The input
// must be a connected manifold, since this is how we determine what
// that order should be! You may want to check for negative volume
// and flip the normals afterwards.
void OrientMesh(TriangularMesh3D *mesh);

// Gets the signed volume of the mesh.
double MeshVolume(const TriangularMesh3D &mesh);

// Flip all of the normals.
void FlipNormals(TriangularMesh3D *mesh);

struct MeshView {
  using vec3 = yocto::vec<double, 3>;

  // perspective transformation.
  double fov = 1.0;
  double near_plane = 0.1;
  double far_plane = 1000.0;

  // looking at origin.
  vec3 camera_pos = {0, 0, 1};
  vec3 up_vector = {0, 1, 0};

  std::string ToString() const;

  static MeshView FromString(std::string_view s);
};

struct MeshEdgeInfo {
  // Must be manifold.
  explicit MeshEdgeInfo(const TriangularMesh3D &mesh);

  // Ordered a < b.
  using Edge = std::pair<int, int>;
  static inline Edge MakeEdge(int a, int b);

  // The edge a-b or b-a must exist, or this aborts!
  // It produces a nonnegative value in [0, π/2].
  double EdgeAngle(int a, int b) const;

 private:
  // Always non-negative.
  std::unordered_map<Edge, double, Hashing<Edge>> dihedral_angle;
  // The other points of the two triangles that touch this edge.
  std::unordered_map<Edge, std::pair<int, int>, Hashing<Edge>> other_points;
};


// Inline implementations follow.

inline MeshEdgeInfo::Edge MeshEdgeInfo::MakeEdge(int a, int b) {
  if (a > b) std::swap(a, b);
  return std::make_pair(a, b);
}

#endif
