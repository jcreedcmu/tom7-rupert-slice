
#ifndef _RUPERTS_HULL_H
#define _RUPERTS_HULL_H

#include "yocto_matht.h"

#include <tuple>
#include <vector>

// Note: Buggy?? See hull_test.cc.
// You can use hull3d.h.
struct QuickHull3D {
  using vec3 = yocto::vec<double, 3>;

  // Returns the triangular faces as indices into the original point
  // cloud.
  static std::vector<std::tuple<int, int, int>>
  Hull(const std::vector<vec3> &points);

  // Just the unique points on the hull.
  static std::vector<int>
  HullPoints(const std::vector<vec3> &points);

};

#endif  // _RUPERTS_HULL_H

