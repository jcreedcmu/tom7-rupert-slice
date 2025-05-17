
// Utilities for working with view positions grouped into
// patches.

#ifndef _RUPERTS_PATCHES_H
#define _RUPERTS_PATCHES_H

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <format>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "arcfour.h"
#include "base/logging.h"
#include "big-polyhedra.h"
#include "polyhedra.h"
#include "yocto_matht.h"

using vec3 = yocto::vec<double, 3>;

struct Boundaries {
  // 1 bit means dot product is positive, 0 means negative.
  uint64_t GetCode(const BigVec3 &v) const;
  uint64_t GetCode(const BigQuat &q) const;

  // Appoximate.
  uint64_t GetCode(const vec3 &v) const;
  uint64_t GetCode(const quat4 &q) const;

  bool HasCodeAssumingMask(uint64_t code, uint64_t mask,
                           const BigVec3 &v, bool include_boundary) const;
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

// a 3x3 matrix consisting that is a signed permutation (each column
// has a single 1 or -1). These are rotations when the determinant
// is 1.
struct SignedPermutation {
  // Default constructor is the identity.
  SignedPermutation() : data(GetPerm(0).data) {}
  // Must be a valid permutation (i.e. from ToWord).
  explicit SignedPermutation(uint16_t d) : data(d) {}
  uint16_t ToWord() const { return data; }

  // For each zero-based column, the zero-based index of the 1 (or if
  // the second pair is true, -1).
  inline std::pair<int, bool> ColIndex(int c) const {
    const uint8_t bits = ColBits(c);
    return std::make_pair(bits & 0b011, !!(bits & 0b100));
  }

  vec3 TransformPoint(const vec3 &v) const {
    // PERF: Whole point of this is that we can avoid multiplications
    // becuase the matrix is sparse and only has 1/-1.
    return ToMatrix() * v;
  }

  BigVec3 TransformPoint(const BigVec3 &v) const {
    // PERF: Here too!
    return ToBigMatrix() * v;
  }

  yocto::mat<double, 3> ToMatrix() const;
  BigMat3 ToBigMatrix() const;

  // for i in [0, 24), returns one of the distinct
  // values. These are the different rotations of the cube
  // or octahedron.
  static SignedPermutation GetPerm(int i);

 private:
  inline uint8_t ColBits(int c) const {
    return (data >> (c * 3)) & 0b111;
  }

  uint16_t data = 0;
};


struct PatchInfo {
  // One of the canonical codes.
  struct CanonicalPatch {
    uint64_t code;
    uint64_t mask;
    BigVec3 example;
  };

  // Maps a code to its canonical patch, including the way that
  // you transform coordinates.
  struct SamePatch {
    // For the patch itself.
    uint64_t code;
    uint64_t mask;

    // The canonical code to use instead.
    uint64_t canonical_code;
    // Transform a point in the patch to the corresponding
    // one in the canonical patch.
    SignedPermutation patch_to_canonical;
  };

  // Map every (inhabited) patch code to the canonical patch.
  std::unordered_map<uint64_t, SamePatch> all_codes;

  // Just the canonical patches.
  std::unordered_map<uint64_t, CanonicalPatch> canonical;
};

PatchInfo LoadPatchInfo(std::string_view filename);
void SavePatchInfo(const PatchInfo &info, std::string_view filename);
inline std::string TwoPatchFilename(uint64_t outer_code, uint64_t inner_code) {
  return std::format("{:x}-{:x}.nds", outer_code, inner_code);
}

// Find the set of patches (as their codes) that are non-empty, by
// shelling out to z3. This could be optimized a lot, but the set is a
// fixed property of the snub cube (given the ordering of vertices and
// faces), so we just need to enumerate them once.
PatchInfo EnumeratePatches(const BigPoly &poly);

// For a given code, get the planes that it is sufficient
// to test in order to determine containment. There is
// not a unique answer (all 1s would always be correct)
// but we find something that's at least locally minimal.
//
// This shells out to Z3 dozens of times, so you should
// probably retain the result.
uint64_t GetCodeMask(const Boundaries &boundaries, uint64_t code,
                     bool verbose = true);

// Compute an arbitrary rotation frame for a view position
// (not unit). The view position may not be on the z axis.
// The rotation frame transforms the view position *to*
// the Z axis (this matches how we treat a quaternion as a
// view position by applying the quaternion's rotation to
// the polyhedron).
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
