
#include "symmetry-groups.h"

#include <vector>
#include <numbers>

#include "polyhedra.h"
#include "yocto_matht.h"

void SymmetryGroups::EdgeRotations(const Polyhedron &poly,
                                   std::vector<frame3> *rots) {

  for (int i = 0; i < poly.vertices.size(); ++i) {
    const vec3 &v0 = poly.vertices[i];
    for (int j : poly.faces->neighbors[i]) {
      CHECK(i != j);
      // Only consider the edge in one orientation.
      if (i < j) {
        const vec3 &v1 = poly.vertices[j];
        const vec3 mid = (v0 + v1) * 0.5;
        const vec3 axis = normalize(mid);

        // But also, we don't want to do this for both an edge and
        // its opposite edge, because that gives us an equivalent
        // rotation. So only do this when the axis is in one half
        // space. We can check the dot product with an arbitrary
        // reference axis. This only fails if the rotation axis is
        // perpendicular to the reference axis, so use one that we
        // know is not perpendicular to any of the rotation axes in
        // these regular polyhedra.
        constexpr vec3 half_space{1.23456789, 0.1133557799, 0.777555};

        if (dot(half_space, axis) > 0.0) {
          // Rotate 180 degrees about the axis.
          rots->push_back(yocto::rotation_frame(
              yocto::rotation_quat(axis, std::numbers::pi)));
        }
      }
    }
  }
}
