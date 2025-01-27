
#include "yocto_matht.h"

#include <vector>

struct Hull3D {
  using vec3 = yocto::vec<double, 3>;

  static std::vector<int> HullPoints(const std::vector<vec3> &v);
};
