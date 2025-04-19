
#ifndef _RUPERTS_POINT_MAP_H
#define _RUPERTS_POINT_MAP_H

#include <optional>
#include <vector>
#include <cstddef>
#include <utility>

#include "yocto_matht.h"

// Below there is also PointSet3, which is a simple wrapper for
// the common case that you don't need interesting values.

// SQDIST is the (squared) threshold below which two points are
// considered the "same." Note that this kind of approach does not
// actually yield an equivalence relation! I think better would
// be to say that we look up with respect to a distance, but we
// insert exact points (possibly overlapping balls).
//
// Value is copied willy-nilly, so it should be something cheap
// like an int.
template<typename Value>
struct PointMap3 {
  using vec3 = yocto::vec<double, 3>;

  PointMap3() {}
  explicit PointMap3(double dist) : sqdist(dist * dist) {}

  // Note that this does *not* deduplicate points! It's just
  // the number of elements that have been inserted.
  size_t Size() const {
    return pts.size();
  }

  bool Contains(const vec3 &p) const {
    for (const auto &[q, v_] : pts) {
      if (distance_squared(p, q) < sqdist) {
        return true;
      }
    }

    return false;
  }

  std::optional<Value> Get(const vec3 &p) const {
    for (const auto &[q, v] : pts) {
      if (distance_squared(p, q) < sqdist) {
        return {v};
      }
    }

    return std::nullopt;
  }

  // This allows points to overlap (or even be exactly coincident),
  // so check Contains() before inserting if you do not want to insert
  // such points. (But then the contents will be order-dependent!)
  void Add(const vec3 &q, const Value &v) {
    pts.emplace_back(q, v);
  }

  std::vector<vec3> Points() const {
    std::vector<vec3> ret;
    ret.reserve(pts.size());
    for (const auto &[p, v] : pts) ret.push_back(p);
    return ret;
  }

 private:
  double sqdist = 0.000001;
  // perf use kd-tree
  std::vector<std::pair<vec3, Value>> pts;
};

// As in the 3d version.
template<typename Value>
struct PointMap2 {
  using vec2 = yocto::vec<double, 2>;

  PointMap2() {}
  explicit PointMap2(double dist) : sqdist(dist * dist) {}

  size_t Size() const {
    return pts.size();
  }

  bool Contains(const vec2 &p) const {
    for (const vec2 &q : pts) {
      if (distance_squared(p, q) < sqdist) {
        return true;
      }
    }

    return false;
  }

  std::optional<Value> Get(const vec2 &p) const {
    for (const auto &[q, v] : pts) {
      if (distance_squared(p, q) < sqdist) {
        return {v};
      }
    }

    return std::nullopt;
  }

  void Add(const vec2 &q, const Value &v) {
    pts.emplace_back(q, v);
  }

  std::vector<vec2> Points() const {
    std::vector<vec2> ret;
    ret.reserve(pts.size());
    for (const auto &[p, v] : pts) ret.push_back(p);
    return ret;
  }

 private:
  double sqdist = 0.000001;
  std::vector<std::pair<vec2, Value>> pts;
};

struct PointSet3 {
  using vec3 = yocto::vec<double, 3>;

  // Note that this does *not* deduplicate points! It's just
  // the number of elements that have been inserted.
  size_t Size() const {
    return m.Size();
  }

  bool Contains(const vec3 &p) const {
    return m.Contains(p);
  }

  // This allows points to overlap (or even be exactly coincident),
  // so check Contains() before inserting if you do not want to insert
  // such points. (But then the contents will be order-dependent!)
  void Add(const vec3 &q) {
    m.Add(q, Unit{});
  }

  std::vector<vec3> Points() const {
    return m.Points();
  }

 private:
  struct Unit { };
  PointMap3<Unit> m;
};

#endif
