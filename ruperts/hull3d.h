
#include "yocto_matht.h"

#include <vector>

struct Hull3D {
  using vec3 = yocto::vec<double, 3>;

  // Return the indices of points that are on the hull.
  static std::vector<int> HullPoints(const std::vector<vec3> &v);

  // Discard all the points that are not on the hull.
  static std::vector<vec3> ReduceToHull(const std::vector<vec3> &v);
};
