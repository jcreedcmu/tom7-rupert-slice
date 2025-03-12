
// To cc-lib!

#ifndef _SMALLEST_SPHERE_H
#define _SMALLEST_SPHERE_H

#include <utility>
#include <vector>

#include "yocto_matht.h"
#include "arcfour.h"

struct SmallestSphere {

  using vec3 = yocto::vec<double, 3>;

  // Compute the smallest sphere that contains all of the
  // points. This is a randomized algorithm, so it needs
  // an ArcFour instance.
  static std::pair<vec3, double> Smallest(
      ArcFour *rc,
      const std::vector<vec3> &pts);

  static bool verbose;
};

#endif
