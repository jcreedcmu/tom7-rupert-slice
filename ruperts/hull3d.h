
#include "yocto_matht.h"

#include <tuple>
#include <vector>

struct Hull3D {
  using vec3 = yocto::vec<double, 3>;

  // Return the indices of points that are on the hull.
  static std::vector<int> HullPoints(const std::vector<vec3> &v);

  // Discard all the points that are not on the hull.
  static std::vector<vec3> ReduceToHull(const std::vector<vec3> &v);

  // Gets triangular faces (might be coplanar, of course, like
  // if computing the hull of a cube). The set of distinct
  // vertices in these triangles is the convex hull.
  static std::vector<std::tuple<int, int, int>> HullFaces(
      const std::vector<vec3> &v);
};
