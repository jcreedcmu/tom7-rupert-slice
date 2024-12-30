
#ifndef _RUPERTS_POINT_SET_H
#define _RUPERTS_POINT_SET_H

#include <vector>
#include <cstddef>
#include <utility>

#include "yocto_matht.h"

// SQDIST is the (squared) threshold below which two points are
// considered the "same." Note that this kind of approach does not
// actually yield an equivalence relation! I think better would
// be to say that we look up with respect to a distance, but we
// insert exact points (possibly overlapping balls).
template<double SQDIST = 0.000001>
struct PointSet {
  using vec3 = yocto::vec<double, 3>;

  size_t Size() const {
    return pts.size();
  }

  bool Contains(const vec3 &p) const {
    for (const vec3 &q : pts) {
      if (distance_squared(p, q) < SQDIST) {
        return true;
      }
    }

    return false;
  }

  // This allows points to overlap (or even be exactly coincident),
  // so check Contains() before inserting if you do not want to insert
  // such points. (But then the contents will be order-dependent!)
  void Add(const vec3 &q) {
    pts.push_back(q);
  }

  // Clearing the point set.
  std::vector<vec3> ExtractVector() {
    std::vector<vec3> ret = std::move(pts);
    pts.clear();
    return ret;
  }

 private:
  // PERF use kd-tree
  std::vector<vec3> pts;
};

#endif
