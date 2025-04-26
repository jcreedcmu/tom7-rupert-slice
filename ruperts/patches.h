
// Utilities for working with view positions grouped into
// patches.

#ifndef _RUPERTS_PATCHES_H
#define _RUPERTS_PATCHES_H

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <optional>
#include <string>
#include <vector>

#include "arcfour.h"
#include "base/logging.h"
#include "big-polyhedra.h"
#include "polyhedra.h"
#include "yocto_matht.h"


inline vec3 QuaternionToSpherePoint(const quat4 &q) {
  // The z-column of the rotation matrix represents the rotated Z-axis.
  // PERF: You can skip computing most of this because of the zeroes.
  return transform_point(rotation_frame(normalize(q)), vec3{0, 0, 1});
}

using vec3 = yocto::vec<double, 3>;

struct Boundaries {
  // 1 bit means dot product is positive, 0 means negative.
  uint64_t GetCode(const BigVec3 &v) const;
  uint64_t GetCode(const BigQuat &q) const;

  // Appoximate.
  uint64_t GetCode(const vec3 &v) const;
  uint64_t GetCode(const quat4 &q) const;

  // Includes boundary. Approximate.
  bool HasCodeAssumingMask(uint64_t code, uint64_t mask,
                           const vec3 &v) const;

  std::string ColorMaskedBits(uint64_t code, uint64_t mask) const;

  explicit Boundaries(const BigPoly &poly);

  size_t Size() const { return big_planes.size(); }

  // Exact.
  std::vector<BigVec3> big_planes;
  BigPoly big_poly;

 private:
  std::vector<vec3> small_planes;
};

// For a given code, get the planes that it is sufficient
// to test in order to determine containment. There is
// not a unique answer (all 1s would always be correct)
// but we find something that's at least locally minimal.
//
// This shells out to Z3 dozens of times, so you should
// probably retain the result.
uint64_t GetCodeMask(const Boundaries &boundaries, uint64_t code);

// Compute an arbitrary rotation frame for a view position
// (not unit). The view position may not be on the z axis.
frame3 FrameFromViewPos(const vec3 &view);

// Random non-unit quaternion.
BigQuat RandomBigQuaternion(ArcFour *rc);

// Find some point in the patch, as a unit view position. Slow.
vec3 GetVec3InPatch(ArcFour *rc,
                    const Boundaries &boundaries,
                    uint64_t code, uint64_t mask = ~uint64_t{0});


// Find some point in the patch, as a non-unit view
// position. Slow.
BigVec3 GetBigVec3InPatch(const Boundaries &boundaries,
                          uint64_t code, uint64_t mask = ~uint64_t{0});

// Find some point in the patch, as a non-unit quaternion.
// Slow.
BigQuat GetBigQuatInPatch(const Boundaries &boundaries,
                          uint64_t code, uint64_t mask = ~uint64_t{0});

// Get the points on the hull when in this view patch. Exact.
// Clockwise winding order.
std::vector<int> ComputeHullForPatch(
    const Boundaries &boundaries,
    uint64_t code,
    uint64_t mask,
    // renders debug output if set (don't include extension)
    std::optional<std::string> render_hull_filebase);


#endif
