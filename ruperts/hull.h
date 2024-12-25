
#ifndef _RUPERTS_HULL_H
#define _RUPERTS_HULL_H

#include "yocto_matht.h"
#include <vector>

struct QuickHull3D {
  using vec3 = yocto::vec<double, 3>;

  // Returns the faces.
  static std::vector<std::vector<int>> Hull(const std::vector<vec3> &points);
};

#endif  // _RUPERTS_HULL_H

