
#ifndef _RUPERTS_HULL_H
#define _RUPERTS_HULL_H

#include "yocto_matht.h"

#include <tuple>
#include <vector>

struct QuickHull3D {
  using vec3 = yocto::vec<double, 3>;

  // Returns the triangular faces as indices into the original point
  // cloud. Note that
  static std::vector<std::tuple<int, int, int>>
  Hull(const std::vector<vec3> &points);
};

#endif  // _RUPERTS_HULL_H

