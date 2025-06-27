
#include "patches.h"

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <format>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "ansi.h"
#include "arcfour.h"
#include "base/logging.h"
#include "base/stringprintf.h"
#include "big-polyhedra.h"
#include "bignum/big.h"
#include "bounds.h"
#include "image.h"
#include "map-util.h"
#include "periodically.h"
#include "polyhedra.h"
#include "randutil.h"
#include "rendering.h"
#include "run-z3.h"
#include "status-bar.h"
#include "threadutil.h"
#include "timer.h"
#include "union-find.h"
#include "util.h"
#include "vector-util.h"
#include "yocto_matht.h"
#include "z3.h"

static const char *NONEMPTY_PATCHES_FILE = "scube-nonempty-patches.txt";
static const char *MASK_AND_EXAMPLE_FILE = "scube-patch-mask-ex.txt";


// 1 bit means dot product is positive, 0 means negative.
uint64_t Boundaries::GetCode(const BigVec3 &v) const {
  uint64_t code = 0;
  for (int i = 0; i < big_planes.size(); i++) {
    const BigVec3 &normal = big_planes[i];
    BigRat d = dot(v, normal);
    int sign = BigRat::Sign(d);
    CHECK(sign != 0) << "Points exactly on the boundary are not "
      "handled.";
    if (sign > 0) {
      code |= uint64_t{1} << i;
    }
  }
  return code;
}

bool Boundaries::HasCodeAssumingMask(
    uint64_t code, uint64_t mask,
    const BigVec3 &v, bool include_boundary) const {
  for (int i = 0; i < big_planes.size(); i++) {
    uint64_t pos = uint64_t{1} << i;
    if (pos & mask) {
      const BigVec3 &normal = big_planes[i];
      BigRat d = dot(v, normal);
      int s = BigRat::Sign(d);
      if (s == 0) {
        return include_boundary;
      }

      // Here we allow the point on the boundary.
      if (pos & code) {
        // Sign should be positive
        if (s != 1) return false;
      } else {
        // Sign should be negative.
        if (s != -1) return false;
      }
    }
  }
  return true;
}

bool Boundaries::HasCodeAssumingMask(uint64_t code, uint64_t mask,
                                     const vec3 &v) const {
  for (int i = 0; i < small_planes.size(); i++) {
    uint64_t pos = uint64_t{1} << i;
    if (pos & mask) {
      const vec3 &normal = small_planes[i];
      double d = dot(v, normal);
      // Here we allow the point on the boundary.
      if (pos & code) {
        // Sign should be positive
        if (d < 0.0) return false;
      } else {
        // Sign should be negative.
        if (d > 0.0) return false;
      }
    }
  }
  return true;
}


uint64_t Boundaries::GetCode(const BigQuat &q) const {
  BigVec3 view = ViewPosFromNonUnitQuat(q);
  return GetCode(view);
}

uint64_t Boundaries::GetCode(const vec3 &v) const {
  uint64_t code = 0;
  for (int i = 0; i < small_planes.size(); i++) {
    const vec3 &normal = small_planes[i];
    double d = dot(v, normal);
    CHECK(d != 0.0) << "Points exactly on the boundary are not "
      "handled.";
    if (d > 0.0) {
      code |= uint64_t{1} << i;
    }
  }
  return code;
}

uint64_t Boundaries::GetCodeSloppy(const vec3 &v) const {
  uint64_t code = 0;
  for (int i = 0; i < small_planes.size(); i++) {
    const vec3 &normal = small_planes[i];
    double d = dot(v, normal);
    // CHECK(d != 0.0) << "Points exactly on the boundary are not "
    // "handled.";
    if (d >= 0.0) {
      code |= uint64_t{1} << i;
    }
  }
  return code;
}

uint64_t Boundaries::GetCode(const quat4 &q) const {
  vec3 view = ViewPosFromNonUnitQuat(q);
  return GetCode(view);
}

std::string Boundaries::ColorMaskedBits(uint64_t code,
                                        uint64_t mask) const {
  const size_t num_bits = Size();
  std::string out;
  uint8_t prev = 0x2A;
  for (int i = num_bits - 1; i >= 0; i--) {
    uint64_t pos = uint64_t{1} << i;
    uint32_t cur = ((!!(code & pos)) << 1) | (!!(mask & pos));
    if (cur != prev) {
      if (mask & pos) {
        // forced bit.
        if (code & pos) {
          out.append(ANSI::ForegroundRGB32(0x76F5F3FF));
        } else {
          out.append(ANSI::ForegroundRGB32(0xB8BBF2FF));
        }
      } else {
        if (code & pos) {
          out.append(ANSI::ForegroundRGB32(0x023540FF));
        } else {
          out.append(ANSI::ForegroundRGB32(0x1A1F6EFF));
        }
      }
      prev = cur;
    }
    out.push_back((code & pos) ? '1' : '0');
  }
  out.append(ANSI_RESET);
  return out;
}


Boundaries::Boundaries(const BigPoly &poly) : big_poly(poly) {
  // Now, the boundaries are planes parallel to faces that pass
  // through the origin. First we find all of these planes
  // and give them ids. These planes need an orientation, too,
  // so a normal vector is a good representation. We can't make
  // this unit length, however.
  //
  // We could actually use integer vectors here! Scale by
  // multiplying by all the denominators, then divide by the GCD.
  // This representation is canonical up to sign flips.

  auto AlreadyHave = [&](const BigVec3 &n) {
      for (const BigVec3 &m : big_planes) {
        if (AllZero(cross(n, m))) {
          return true;
        }
      }
      return false;
    };

  for (const std::vector<int> &face : poly.faces->v) {
    CHECK(face.size() >= 3);
    const BigVec3 &a = poly.vertices[face[0]];
    const BigVec3 &b = poly.vertices[face[1]];
    const BigVec3 &c = poly.vertices[face[2]];
    BigVec3 normal = ScaleToMakeIntegral(cross(c - a, b - a));
    // printf("Normal: %s\n", VecString(normal).c_str());
    if (!AlreadyHave(normal)) {
      big_planes.push_back(normal);
    }
  }

  printf("There are %d distinct planes.\n",
         (int)big_planes.size());

  // You can switch to a larger word size for more complex
  // polyhedra.
  CHECK(big_planes.size() <= 64);

  for (const BigVec3 &v : big_planes) {
    small_planes.push_back(SmallVec(v));
  }
}

frame3 FrameFromViewPos(const vec3 &view) {
  CHECK(length(view) > 1e-9);

  vec3 frame_z = yocto::normalize(view);

  const vec3 up_z = {0.0, 0.0, 1.0};
  vec3 frame_x = normalize(cross(up_z, frame_z));

  frame3 frame{
    .x = frame_x,
    .y = cross(frame_z, frame_x),
    .z = frame_z,
    .o = vec3{0, 0, 0},
  };

  // The convention was opposite of what ViewPosFromNonUnitQuat
  // did, so invert this. (PERF)
  return inverse(frame);
}

BigQuat RandomBigQuaternion(ArcFour *rc) {
  quat4 s = RandomQuaternion(rc);
  return BigQuat(BigRat::ApproxDouble(s.x, 1000000),
                 BigRat::ApproxDouble(s.y, 1000000),
                 BigRat::ApproxDouble(s.z, 1000000),
                 BigRat::ApproxDouble(s.w, 1000000));
}

BigVec3 GetBigVec3InPatch(const Boundaries &boundaries,
                          uint64_t code, uint64_t mask) {
  ArcFour rc(std::format("point{}", code));
  for (;;) {
    vec3 s;
    std::tie(s.x, s.y, s.z) = RandomUnit3D(&rc);
    BigVec3 bs(BigRat::ApproxDouble(s.x, 1000000),
               BigRat::ApproxDouble(s.y, 1000000),
               BigRat::ApproxDouble(s.z, 1000000));

    // Try all the signs. At most one of these will be
    // in the patch, but this should increase the
    // efficiency (because knowing that other signs
    // are *not* in the patch increases the chance
    // that we are).
    for (uint8_t b = 0b000; b < 0b1000; b++) {
      BigVec3 bbs((b & 0b100) ? -bs.x : bs.x,
                  (b & 0b010) ? -bs.y : bs.y,
                  (b & 0b001) ? -bs.z : bs.z);
      if (boundaries.HasCodeAssumingMask(code, mask, bbs,
                                         // Don't include boundary.
                                         false)) {
        return bbs;
      }
    }
  }
}

// XXX use mask
BigQuat GetBigQuatInPatch(const Boundaries &boundaries,
                          uint64_t code, uint64_t mask) {
  ArcFour rc(std::format("point{}", code));

  for (;;) {
    // Work with doubles, but verify with rationals.
    BigQuat q = [&]() {
        for (;;) {
          quat4 smallq = RandomQuaternion(&rc);

          // Try all the signs. At most one of these will be in the patch,
          // but this should increase the efficiency (because knowing that
          // other signs are *not* in the patch increases the chance that we
          // are).
          //
          // PERF: We could integrate this with GetCode so that it
          // flips signs
          for (uint8_t b = 0b000; b < 0b1000; b++) {
            quat4 qsign((b & 0b100) ? -smallq.x : smallq.x,
                        (b & 0b010) ? -smallq.y : smallq.y,
                        (b & 0b001) ? -smallq.z : smallq.z,
                        // negating the whole quaternion
                        // is redundant, so wlog we leave
                        // the real part the same.
                        smallq.w);
            uint64_t sample_code = boundaries.GetCode(qsign);
            if (sample_code == code) {
              // Sample is in range.
              return BigQuat(BigRat::ApproxDouble(qsign.x, 1000000),
                             BigRat::ApproxDouble(qsign.y, 1000000),
                             BigRat::ApproxDouble(qsign.z, 1000000),
                             BigRat::ApproxDouble(qsign.w, 1000000));
            }
          }
        }
      }();

    if (boundaries.GetCode(q) == code) {
      return q;
    }
  }
}

vec3 GetVec3InPatch(ArcFour *rc,
                    const Boundaries &boundaries,
                    uint64_t code, uint64_t mask) {
  for (;;) {
    vec3 s;
    std::tie(s.x, s.y, s.z) = RandomUnit3D(rc);

    // Try all the signs. At most one of these will be
    // in the patch, but this should increase the
    // efficiency (because knowing that other signs
    // are *not* in the patch increases the chance
    // that we are).
    for (uint8_t b = 0b000; b < 0b1000; b++) {
      vec3 ss((b & 0b100) ? -s.x : s.x,
              (b & 0b010) ? -s.y : s.y,
              (b & 0b001) ? -s.z : s.z);
      if (boundaries.HasCodeAssumingMask(code, mask, ss)) {
        return ss;
      }
    }
  }
}



uint64_t GetCodeMask(const Boundaries &boundaries,
                     uint64_t code,
                     bool verbose) {
  Periodically status_per(1);
  StatusBar status(1);
  const int num_bits = boundaries.Size();

  std::vector<Z3Bool> bits;
  std::string setup;
  for (int b = 0; b < num_bits; b++) {
    // true = 1 = postive dot product
    bits.emplace_back(NewBool(&setup, std::format("bit{}", b)));
  }
  // The hypothesized point. If unsatisfiable, then the patch
  // is empty.
  Z3Vec3 v = NewVec3(&setup, "pt");

  // Inequalities for the bits.
  for (int b = 0; b < num_bits; b++) {
    Z3Vec3 normal{boundaries.big_planes[b]};
    AppendFormat(&setup,
                 "(assert (ite {} (> {} 0.0) (< {} 0.0)))\n",
                 bits[b].s,
                 Dot(v, normal).s,
                 Dot(v, normal).s);
  }

  // Start with all 1s.
  uint64_t mask = (uint64_t{1} << boundaries.Size()) - 1;

  for (int test_bit = 0; test_bit < num_bits; test_bit++) {
    if (verbose) {
      status_per.RunIf([&]() {
          status.Progressf(test_bit, num_bits, "%s",
                           std::format("Mask: {:b}", mask).c_str());
        });
    }

    std::string out = setup;

    // Now add constraints for each that is currently masked 1,
    // NOT including the test bit.
    for (int b = 0; b < num_bits; b++) {
      const uint64_t pos = uint64_t{1} << b;
      if (b != test_bit && !!(pos & mask)) {
        // Assert that the bit has the value from the code.
        AppendFormat(&out, "(assert (= {} {}))\n",
                     bits[b].s, Z3Bool((code & pos) ? true : false).s);
      }
    }

    // Now assert the *opposite* bit for the test bit.
    // Since this is the only change, if it is unsatisfiable
    // then we know that bit must have that value, and can
    // remove it from the mask.
    uint64_t test_pos = uint64_t{1} << test_bit;
    AppendFormat(&out,
                 ";; test bit\n"
                 "(assert (= {} {}))\n",
                 bits[test_bit].s,
                 Z3Bool(!(code & test_pos)).s);

    AppendFormat(&out, "(check-sat)\n");

    switch (RunZ3(out, {60.0})) {
    case Z3Result::SAT:
      // Bit is necessary, since it can feasibly have the opposite
      // value with the current mask.
      break;
    case Z3Result::UNSAT:
      // Bit is unnecessary; its value is forced with the current mask.
      mask &= ~test_pos;
      break;
    case Z3Result::UNKNOWN:
      // If we don't know, we have to keep it.
      printf(ARED("Z3 at its limits??") " Code %llu "
             "mask %llu\n", code, mask);
      break;
    }
  }

  if (verbose) {
    printf("Reduced mask from %d to %d bits.\n",
           (int)num_bits, std::popcount<uint64_t>(mask));
  }
  return mask;
}

static bool CanCodeContainZ(const Boundaries &boundaries,
                            uint64_t code) {
  const int num_bits = boundaries.Size();

  std::string out;

  // The hypothesized point on the z axis.
  Z3Vec3 v = NewVec3(&out, "pt");
  AppendFormat(&out,
               ";; is there a point on the z axis (not origin)?\n"
               "(assert (not (= {} 0.0)))\n"
               "(assert (not (= {} 0.0)))\n"
               "(assert (= {} 0.0))\n",
               v.x.s, v.y.s, v.z.s);

  AppendFormat(&out,
               "\n"
               ";; patch constraints\n");
  for (int b = 0; b < num_bits; b++) {
    Z3Vec3 normal{boundaries.big_planes[b]};
    const char *side = (code & (uint64_t{1} << b)) ? ">" : "<";
    AppendFormat(&out,
                 "(assert ({} {} 0.0))\n",
                 side,
                 Dot(v, normal).s);
  }

  AppendFormat(&out, "(check-sat)\n");

  switch (RunZ3(out, {60.0})) {
  case Z3Result::SAT:
    // Found a solution including the Z axis.
    return true;
  case Z3Result::UNSAT:
    // No solutions with Z axis.
    return false;
  case Z3Result::UNKNOWN:
    // Could consider this as having the z axis conservatively,
    // but we also expect Z3 to be able to succeed at these.
    printf(ARED("Z3 at its limits??") " Code %llu ", code);
    LOG(FATAL) << out;
    return true;
  }
}

static bool CanBeAllPositive(const Boundaries &boundaries,
                             uint64_t code) {
  const int num_bits = boundaries.Size();

  std::string out;

  // The hypothesized point that is all positive.
  Z3Vec3 v = NewVec3(&out, "pt");
  AppendFormat(&out,
               ";; is there a point with all positive coordinates?\n"
               "(assert (> {} 0.0)))\n"
               "(assert (> {} 0.0)))\n"
               "(assert (> {} 0.0)))\n",
               v.x.s, v.y.s, v.z.s);

  AppendFormat(&out,
               "\n"
               ";; patch constraints\n");
  for (int b = 0; b < num_bits; b++) {
    Z3Vec3 normal{boundaries.big_planes[b]};
    const char *side = (code & (uint64_t{1} << b)) ? ">" : "<";
    AppendFormat(&out,
                 "(assert ({} {} 0.0))\n",
                 side,
                 Dot(v, normal).s);
  }

  AppendFormat(&out, "(check-sat)\n");

  switch (RunZ3(out, {60.0})) {
  case Z3Result::SAT:
    return true;
  case Z3Result::UNSAT:
    return false;
  case Z3Result::UNKNOWN:
    printf(ARED("Z3 at its limits??") " Code %llu ", code);
    LOG(FATAL) << out;
    return false;
  }
}



std::vector<int> ComputeHullForPatch(
    const Boundaries &boundaries,
    uint64_t code,
    uint64_t mask,
    std::optional<std::string> render_hull_filebase) {
  BigQuat example_view = GetBigQuatInPatch(boundaries, code, mask);
  BigFrame frame = NonUnitRotationFrame(example_view);

  BigMesh2D full_shadow = Shadow(Rotate(frame, boundaries.big_poly));
  std::vector<int> hull = BigQuickHull(full_shadow.vertices);
  CHECK(hull.size() >= 3);

  if (render_hull_filebase.has_value()) {
    Rendering rendering(SmallPoly(boundaries.big_poly), 1920, 1080);
    auto small_shadow = SmallMesh(full_shadow);
    rendering.RenderMesh(small_shadow);
    rendering.RenderHull(small_shadow, hull);
    rendering.Save(std::format("{}-hull-{:b}.png",
                               render_hull_filebase.value(),
                               code));
  }

  BigRat area = SignedAreaOfHull(full_shadow, hull);
  if (render_hull_filebase.has_value()) {
    printf("Area for example hull: %.17g\n", area.ToDouble());
  }

  if (BigRat::Sign(area) == -1) {
    VectorReverse(&hull);
    area = SignedAreaOfHull(full_shadow, hull);
    if (render_hull_filebase.has_value()) {
      printf("Reversed hull to get area: %.17g\n", area.ToDouble());
    }
  }

  CHECK(BigRat::Sign(area) == 1);

  if (render_hull_filebase.has_value()) {
    const Polyhedron small_poly = SmallPoly(boundaries.big_poly);
    auto PlaceHull = [&](const frame3 &frame,
                         const std::vector<int> &hull) ->
      std::vector<vec2> {
      std::vector<vec2> out;
      out.resize(hull.size());
      for (int hidx = 0; hidx < hull.size(); hidx++) {
        int vidx = hull[hidx];
        const vec3 &v_in = small_poly.vertices[vidx];
        // PERF: Don't need z coordinate.
        const vec3 v_out = transform_point(frame, v_in);
        out[hidx] = vec2{v_out.x, v_out.y};
      }
      return out;
    };

    std::vector<vec2> phull = PlaceHull(SmallFrame(frame), hull);

    Bounds bounds;
    for (const auto &vec : phull) {
      bounds.Bound(vec.x, vec.y);
    }
    bounds.AddMarginFrac(0.05);

    ImageRGBA ref(1920, 1080);
    ref.Clear32(0x000000FF);
    Bounds::Scaler scaler =
      bounds.ScaleToFit(ref.Width(), ref.Height()).FlipY();
    for (int i = 0; i < phull.size(); i++) {
      const vec2 &v0 = phull[i];
      const vec2 &v1 = phull[(i + 1) % phull.size()];
      const auto &[x0, y0] = scaler.Scale(v0.x, v0.y);
      const auto &[x1, y1] = scaler.Scale(v1.x, v1.y);
      ref.BlendLine32(x0, y0, x1, y1, 0xFFFFFFAA);
    }
    ref.Save(std::format("{}-phull-{:b}.png",
                         render_hull_filebase.value(), code));
  }

  return hull;
}


// Encodes a 3x3 signed permutation matrix (in column-major order)
// into a uint16_t. The matrix is not checked!
static consteval uint16_t Encode(
    const std::array<int, 9> &matrix_col_major) {

  uint16_t data = 0;
  for (int col = 0; col < 3; ++col) {
    for (int row = 0; row < 3; ++row) {
      int val = matrix_col_major[col * 3 + row];
      if (val != 0) {
        uint16_t col_bits = ((val == -1) ? 0b100 : 0b000) | row;
        data |= col_bits << (col * 3);
      }
    }
  }

  return data;
}

SignedPermutation SignedPermutation::GetPerm(int i) {
  static constexpr uint16_t OCTAHEDRAL_GROUP[24] = {
    // Identity.
    Encode({1, 0, 0,  0, 1, 0,  0, 0, 1}),
    // Around X axis.
    Encode({1, 0, 0,  0,-1, 0,  0, 0,-1}),
    Encode({1, 0, 0,  0, 0, 1,  0,-1, 0}),
    Encode({1, 0, 0,  0, 0,-1,  0, 1, 0}),
    // Around Y axis.
    Encode({-1,0, 0,  0, 1, 0,  0, 0,-1}),
    Encode({0, 0,-1,  0, 1, 0,  1, 0, 0}),
    Encode({0, 0, 1,  0, 1, 0, -1, 0, 0}),
    // Around Z axis.
    Encode({-1,0, 0,  0,-1, 0,  0, 0, 1}),
    Encode({0, 1, 0, -1, 0, 0,  0, 0, 1}),
    Encode({0,-1, 0,  1, 0, 0,  0, 0, 1}),

    // Edge rotations.
    Encode({0, 1, 0,  1, 0, 0,  0, 0,-1}),
    Encode({0, 0, 1,  0,-1, 0,  1, 0, 0}),
    Encode({-1,0, 0,  0, 0, 1,  0, 1, 0}),
    Encode({0,-1, 0, -1, 0, 0,  0, 0,-1}),
    Encode({0, 0,-1,  0,-1, 0, -1, 0, 0}),
    Encode({-1,0, 0,  0, 0,-1,  0,-1, 0}),

    // Vertex rotations.
    Encode({0, 1, 0,  0, 0, 1,  1, 0, 0}),
    Encode({0, 0, 1,  1, 0, 0,  0, 1, 0}),
    Encode({0,-1, 0,  0, 0, 1, -1, 0, 0}),
    Encode({0, 0,-1, -1, 0, 0,  0, 1, 0}),
    Encode({0,-1, 0,  0, 0,-1,  1, 0, 0}),
    Encode({0, 0, 1, -1, 0, 0,  0,-1, 0}),
    Encode({0, 0,-1,  1, 0, 0,  0,-1, 0}),
    Encode({0, 1, 0,  0, 0,-1, -1, 0, 0}),
  };

  CHECK(i >= 0 && i < 24);
  return SignedPermutation(OCTAHEDRAL_GROUP[i]);
}

yocto::mat<double, 3> SignedPermutation::ToMatrix() const {
  yocto::mat<double, 3> m = {
    .x = {0, 0, 0},
    .y = {0, 0, 0},
    .z = {0, 0, 0},
  };

  auto [dx, sx] = ColIndex(0);
  m.x[dx] = sx ? -1 : 1;
  auto [dy, sy] = ColIndex(1);
  m.y[dy] = sy ? -1 : 1;
  auto [dz, sz] = ColIndex(2);
  m.z[dz] = sz ? -1 : 1;
  return m;
}

BigMat3 SignedPermutation::ToBigMatrix() const {
  BigVec3 zero{BigRat(0), BigRat(0), BigRat(0)};
  BigMat3 m = BigMat3{
    .x = zero,
    .y = zero,
    .z = zero,
  };
  auto [dx, sx] = ColIndex(0);
  m.x[dx] = BigRat(sx ? -1 : 1);
  auto [dy, sy] = ColIndex(1);
  m.y[dy] = BigRat(sy ? -1 : 1);
  auto [dz, sz] = ColIndex(2);
  m.z[dz] = BigRat(sz ? -1 : 1);
  return m;
}

namespace {
struct PatchEnumerator {
  PatchEnumerator(const Boundaries &boundaries) :
    boundaries(boundaries), status(1) {
    status.Statusf("Setup.");
    // Find patches that are non-empty.
    // Naively there are 2^31 of them, but the vast majority
    // are completely empty. Z3 is a good way to prove this.
    std::string out;

    int num_bits = boundaries.Size();
    for (int b = 0; b < num_bits; b++) {
      // true = 1 = postive dot product
      bits.emplace_back(NewBool(&out, std::format("bit{}", b)));
    }

    // The hypothesized point. If unsatisfiable, then the patch
    // is empty.
    Z3Vec3 v = NewVec3(&out, "pt");

    // Constrain v based on the bits.
    for (int b = 0; b < num_bits; b++) {
      Z3Vec3 normal{boundaries.big_planes[b]};
      AppendFormat(&out,
                   "(assert (ite {} (> {} 0.0) (< {} 0.0)))\n",
                   bits[b].s,
                   Dot(v, normal).s,
                   Dot(v, normal).s);
    }
    setup = std::move(out);
    status.Statusf("Setup done.");
  }

  std::string PartialCodeString(int depth, uint64_t code) {
    if (depth == 0) return AGREY("(empty)");
    std::string ret;
    for (int i = depth - 1; i >= 0; i--) {
      uint64_t bit = 1ULL << i;
      ret.append((code & bit) ? ACYAN("1") : ABLUE("0"));
    }
    return ret;
  }

  // Bits < depth have been assigned to the values in code.
  void EnumerateRec(int depth, uint64_t code) {
    // Is it possible at all?
    std::string out = setup;

    for (int b = 0; b < depth; b++) {
      AppendFormat(&out, "(assert (= {} {}))\n",
                   bits[b].s,
                   (code & (1UL << b)) ? "true" : "false");
    }

    // Don't even need to get the model here.
    AppendFormat(&out, "(check-sat)\n");

    z3calls++;
    status.Statusf("Z3: %s", std::format("{:b}", code).c_str());
    Z3Result z3result = RunZ3(out);
    status.Statusf("%lld Z3 calls. Depth %d\n", z3calls, depth);
    CHECK(z3result != Z3Result::UNKNOWN) << "Expecting a definitive "
      "answer here";

    if (z3result == Z3Result::UNSAT) {
      status.Print("Code {} is impossible.\n",
                   PartialCodeString(depth, code));
      return;
    }

    if (depth == boundaries.Size()) {
      // Then we have a complete code.
      status.Print(AGREEN("Nonempty") ": {}\n",
                   PartialCodeString(depth, code));
      nonempty_patches.push_back(code);
      return;
    }
    CHECK(depth < boundaries.Size());

    // Otherwise, we try extending with 0, and with 1.
    EnumerateRec(depth + 1, code);
    EnumerateRec(depth + 1, code | (1ULL << depth));
  }

  std::vector<uint64_t> Enumerate() {
    Timer timer;
    EnumerateRec(0, 0);
    status.Print("Computed {} patches in {} ({} z3 calls)\n",
                 nonempty_patches.size(),
                 ANSI::Time(timer.Seconds()),
                 z3calls);
    return nonempty_patches;

    #if 0
    std::string all_codes;
    for (uint64_t code : nonempty_patches) {
      AppendFormat(&all_codes, "{:b}\n", code);
    }
    Util::WriteFile(NONEMPTY_PATCHES_FILE, all_codes);
    #endif
  }

 private:
  Boundaries boundaries;
  std::vector<Z3Bool> bits;
  std::string setup;
  int64_t z3calls = 0;
  StatusBar status;
  std::vector<uint64_t> nonempty_patches;
};
}

static void WriteMaskAndExampleFile(const Boundaries &boundaries) {

  // PatchEnumerator enumerator(boundaries);
  // std::vector<uint64_t> all = enumerator.Enumerate();
  std::vector<uint64_t> all;
  for (const std::string &line :
         Util::ReadFileToLines(NONEMPTY_PATCHES_FILE)) {
    if (line.empty()) continue;
    std::optional<uint64_t> bo = Util::ParseBinary(line);
    CHECK(bo.has_value()) << "Bad line in " << NONEMPTY_PATCHES_FILE
                          << ": " << line;
    all.push_back(bo.value());
  }
  printf("Computed/loaded " ACYAN("%d") " patches.\n",
         (int)all.size());

  // Find an example vector for each patch.
  StatusBar status(1);
  Periodically status_per(1.0);


  std::mutex mu;
  std::unordered_set<int> outstanding;

  std::vector<uint64_t> masks =
    ParallelMapi(all,
                 [&](int idx, uint64_t code) {
                   {
                     MutexLock ml(&mu);
                     CHECK(!outstanding.contains(idx));
                     outstanding.insert(idx);
                   }

                   const uint64_t mask =
                     GetCodeMask(boundaries, code, false);

                   status.Print("{}. Code {:b} ok\n", idx, code);

                   {
                     MutexLock ml(&mu);
                     CHECK(outstanding.contains(idx));
                     outstanding.erase(idx);
                   }

                   status_per.RunIf([&]() {
                       std::string s;
                       for (int o : outstanding) {
                         AppendFormat(&s, " {}", o);
                       }
                       status.Progressf(idx, all.size(),
                                        "Get mask. Outstanding: %s",
                                        s.c_str());
                     });

                   return mask;
                 },
                 1);

  CHECK(masks.size() == all.size());


  std::unordered_map<uint64_t, BigVec3> examples;

  constexpr int NUM_SHOTGUN = 1'000'000;
  ParallelFan(8, [&](int thread_idx) {
      ArcFour rc(std::format("example{}", thread_idx));

      for (int i = 0; i < NUM_SHOTGUN; i++) {
        vec3 s;
        std::tie(s.x, s.y, s.z) = RandomUnit3D(&rc);
        BigVec3 bs(BigRat::ApproxDouble(s.x, 1000000),
                   BigRat::ApproxDouble(s.y, 1000000),
                   BigRat::ApproxDouble(s.z, 1000000));
        uint64_t code = boundaries.GetCode(bs);
        {
          MutexLock ml(&mu);
          if (!examples.contains(code)) {
            examples[code] = std::move(bs);
          }
        }

        if (thread_idx == 0) {
          status_per.RunIf([&]() {
              int num = WithLock(&mu, [&]() { return examples.size(); });
              status.Progressf(i, NUM_SHOTGUN,
                               "Shotgun: %d/%d examples",
                               num, (int)all.size());
            });
        }
      }

      // Then try solving the hard ones.
      for (int i = 0; i < all.size(); i++) {
        uint64_t code = all[i];
        uint64_t mask = masks[i];

        for (;;) {
          {
            MutexLock ml(&mu);
            if (examples.contains(code)) {
              break;
            }
          }

          for (int tries = 0; tries < 100; tries++) {
            vec3 s;
            std::tie(s.x, s.y, s.z) = RandomUnit3D(&rc);

            for (int p = 0; p < 24; p++) {
              vec3 ss = SignedPermutation::GetPerm(p).TransformPoint(s);
              if (boundaries.HasCodeAssumingMask(code, mask, ss)) {
                BigVec3 bs(BigRat::ApproxDouble(s.x, 100000000),
                           BigRat::ApproxDouble(s.y, 100000000),
                           BigRat::ApproxDouble(s.z, 100000000));
                if (boundaries.HasCodeAssumingMask(code, mask, bs,
                                                   // Don't include boundary.
                                                   false)) {
                  MutexLock ml(&mu);
                  if (!examples.contains(code)) {
                    examples[code] = std::move(bs);
                  }
                  goto next_code;
                }
              }
            }
          }

          status_per.RunIf([&]() {
              int num = WithLock(&mu, [&]() { return examples.size(); });
              status.Progressf(i, NUM_SHOTGUN,
                               "Hard: %d/%d examples",
                               num, (int)all.size());
            });
        }

      next_code:;
      }
    });


  // Save to file.
  std::string contents;
  for (int i = 0; i < all.size(); i++) {
    const uint64_t code = all[i];
    const uint64_t mask = masks[i];
    auto ei = examples.find(code);
    CHECK(ei != examples.end());
    const BigVec3 &example = ei->second;
    AppendFormat(&contents,
                 "{:b} {:b} {} {} {}\n",
                 code, mask,
                 example.x.ToString(),
                 example.y.ToString(),
                 example.z.ToString());
  }

  Util::WriteFile(MASK_AND_EXAMPLE_FILE, contents);
}

// Internal.
void AddHulls(const Boundaries &boundaries,
              PatchInfo *info) {
  for (auto &[code, canon] : info->canonical) {
    CHECK(code == canon.code);
    canon.hull = ComputeHullForPatch(boundaries,
                                     canon.code, canon.mask,
                                     std::nullopt);
  }
}

PatchInfo EnumeratePatches(const BigPoly &poly) {
  Timer run_timer;
  Boundaries boundaries(poly);
  if (!Util::ExistsFile(MASK_AND_EXAMPLE_FILE)) {
    WriteMaskAndExampleFile(boundaries);
  }

  struct Patch {
    uint64_t code = 0;
    uint64_t mask = 0;
    BigVec3 example;
    bool can_be_canonical = false;
    bool can_be_positive = false;
  };

  StatusBar status(1);
  Periodically status_per(1.0);
  std::vector<Patch> all;
  for (const std::string &raw_line :
         Util::ReadFileToLines(MASK_AND_EXAMPLE_FILE)) {
    std::string line = Util::NormalizeWhitespace(raw_line);
    if (line.empty()) continue;
    std::optional<uint64_t> ocode = Util::ParseBinary(Util::chop(line));
    std::optional<uint64_t> omask = Util::ParseBinary(Util::chop(line));
    std::string sx = Util::chop(line);
    std::string sy = Util::chop(line);
    std::string sz = Util::chop(line);
    CHECK(ocode.has_value());
    CHECK(omask.has_value());
    CHECK(!sz.empty());

    BigVec3 v{BigRat(sx), BigRat(sy), BigRat(sz)};
    CHECK(!AllZero(v));
    all.push_back(Patch{
        .code = ocode.value(),
        .mask = omask.value(),
        .example = std::move(v),
        .can_be_canonical = false,
        .can_be_positive = false,
      });
  }

  CHECK(all.size() == 848);

  printf("(re)loaded patches.\n");

  std::unordered_map<uint64_t, int> idx_from_code;
  for (int i = 0; i < all.size(); i++) idx_from_code[all[i].code] = i;
  auto IdxFromCode = [&](uint64_t code) {
      auto it = idx_from_code.find(code);
      CHECK(it != idx_from_code.end());
      return it->second;
    };

  // Now, figure out which patches are actually the same, assuming
  // octahedral (rotational) symmetry.

  auto PatchString = [&](const Patch &patch) {
      return boundaries.ColorMaskedBits(patch.code, patch.mask);
    };

  // We can choose any one of the patches in an equivalence class as
  // the canonical one, but it is useful to assume that it does *not*
  // contain the z axis (so that e.g. we can use polar coordinates
  // without singularities).
  for (int idx = 0; idx < all.size(); idx++) {
    Patch &patch = all[idx];
    bool can_contain_z = CanCodeContainZ(boundaries, patch.code);
    patch.can_be_canonical = !can_contain_z;
    patch.can_be_positive = CanBeAllPositive(boundaries, patch.code);
    if (can_contain_z) {
      printf("Patch %s can contain z axis.\n",
             PatchString(patch).c_str());
    }

    status_per.RunIf([&]() {
        status.Progressf(idx, all.size(), "Compute properties");
      });
  }

  // In the first pass, get the equivalence classes.
  Timer eq_timer;
  UnionFind uf(all.size());
  for (int idx = 0; idx < all.size(); idx++) {
    Patch &patch = all[idx];
    // Transform it into canonical patch.
    for (int i = 0; i < 24; i++) {
      SignedPermutation perm = SignedPermutation::GetPerm(i);
      BigVec3 v = perm.TransformPoint(patch.example);

      uint64_t code = boundaries.GetCode(v);
      uf.Union(idx, IdxFromCode(code));
    }
  }
  printf("Equivalence classes in %s\n",
         ANSI::Time(eq_timer.Seconds()).c_str());

  // Get the equivalence classes in the vectors.
  // The representative here is chosen by Union-Find; it's
  // not necessarily what we want to use as the canonical patch.
  CHECK(all.size() == uf.Size());
  std::unordered_map<int, std::vector<int>> eq_classes;
  for (int idx = 0; idx < all.size(); idx++) {
    eq_classes[uf.Find(idx)].push_back(idx);
  }

  printf("There are %d equivalence classes.\n",
         (int)eq_classes.size());
  for (const auto &[_, values] : eq_classes) {
    for (int idx : values) {
      printf(" %d", idx);
    }
    printf("\n");
  }

  // Choose a canonical one. All we require is that it
  // not contain the z axis, but we prefer to find one
  // that is also all positive.
  auto ChooseCanonical = [&](const std::vector<int> &indices) {
      for (int idx : indices) {
        const Patch &patch = all[idx];
        if (patch.can_be_canonical &&
            patch.can_be_positive) {
          return idx;
        }
      }

      for (int idx : indices) {
        const Patch &patch = all[idx];
        if (patch.can_be_canonical) {
          return idx;
        }
      }

      LOG(FATAL) << "There was no canonical patch available!?";
    };

  auto GetPermTo = [&](const BigVec3 &example, uint64_t canonical_code) {
      for (int p = 0; p < 24; p++) {
        SignedPermutation perm = SignedPermutation::GetPerm(p);
        BigVec3 v = perm.TransformPoint(example);
        if (boundaries.GetCode(v) == canonical_code) {
          return perm;
        }
      }

      LOG(FATAL) << "Bug: Couldn't reproduce rotation to canonical "
        "code?";
    };

  PatchInfo info;
  for (const auto &[_, values] : eq_classes) {
    int canonical_idx = ChooseCanonical(values);
    const Patch &canonical_patch = all[canonical_idx];
    {
      PatchInfo::CanonicalPatch cp;
      cp.code = canonical_patch.code;
      cp.mask = canonical_patch.mask;
      cp.example = canonical_patch.example;
      info.canonical[canonical_patch.code] = std::move(cp);
    }

    for (int idx : values) {
      const Patch &patch = all[idx];
      SignedPermutation perm =
        GetPermTo(patch.example, canonical_patch.code);
      PatchInfo::SamePatch same{
        .code = patch.code,
        .mask = patch.mask,
        .canonical_code = canonical_patch.code,
        .patch_to_canonical = perm,
      };
      info.all_codes[patch.code] = same;
    }

    // Now for every one (including itself), add it and the
    // rotation needed to reach the canonical.
    printf("%s Canonical: %d (size %d)\n",
           canonical_patch.can_be_positive ? "[" ACYAN("+") "]" : "[-]",
           canonical_idx, (int)values.size());
  }

  printf("Computed patch info in %s\n",
         ANSI::Time(run_timer.Seconds()).c_str());

  AddHulls(boundaries, &info);

  return info;
}

static std::string BigVecString(const BigVec3 &v) {
  return std::format("{} {} {}",
                     v.x.ToString(),
                     v.y.ToString(),
                     v.z.ToString());
}

void SavePatchInfo(const PatchInfo &info, std::string_view filename) {

  std::string contents;
  for (const auto &[code, canon] : MapToSortedVec(info.canonical)) {
    CHECK(code == canon.code);
    AppendFormat(&contents, "c {:b} {:b} {}",
                 canon.code, canon.mask, BigVecString(canon.example));
    for (int v : canon.hull) {
      AppendFormat(&contents, " {}", v);
    }
    contents.append("\n");
  }

  for (const auto &[code, same] : MapToSortedVec(info.all_codes)) {
    CHECK(code == same.code);
    AppendFormat(&contents, "a {:b} {:b} {:b} {}\n",
                 same.code, same.mask, same.canonical_code,
                 same.patch_to_canonical.ToWord());
  }

  Util::WriteFile(filename, contents);
}



PatchInfo LoadPatchInfo(std::string_view filename) {
  PatchInfo info;
  for (const std::string &raw_line :
         Util::NormalizeLines(Util::ReadFileToLines(filename))) {
    std::string line = raw_line;
    std::string cmd = Util::chop(line);
    if (cmd == "c") {
      std::optional<uint64_t> ocode = Util::ParseBinary(Util::chop(line));
      std::optional<uint64_t> omask = Util::ParseBinary(Util::chop(line));
      std::string sx = Util::chop(line);
      std::string sy = Util::chop(line);
      std::string sz = Util::chop(line);
      CHECK(ocode.has_value());
      CHECK(omask.has_value());
      CHECK(!sz.empty());

      BigVec3 v{BigRat(sx), BigRat(sy), BigRat(sz)};
      CHECK(!AllZero(v));

      // Read the rest of the line as hull points.

      std::vector<std::string> shull = Util::Tokenize(line, ' ');
      std::vector<int> hull;
      hull.reserve(shull.size());
      for (const std::string &sv : shull) {
        hull.push_back(atoi(sv.c_str()));
      }

      info.canonical[ocode.value()] = PatchInfo::CanonicalPatch{
          .code = ocode.value(),
          .mask = omask.value(),
          .example = std::move(v),
          .hull = std::move(hull),
        };

    } else if (cmd == "a") {
      std::optional<uint64_t> ocode = Util::ParseBinary(Util::chop(line));
      std::optional<uint64_t> omask = Util::ParseBinary(Util::chop(line));
      std::optional<uint64_t> ocano = Util::ParseBinary(Util::chop(line));
      uint16_t perm = Util::stoi(line);
      CHECK(ocode.has_value());
      CHECK(omask.has_value());
      CHECK(ocano.has_value());
      CHECK(perm > 0) << "Bad (probably short) line; perm cannot be zero";

      info.all_codes[ocode.value()] = PatchInfo::SamePatch{
          .code = ocode.value(),
          .mask = omask.value(),
          .canonical_code = ocano.value(),
          .patch_to_canonical = SignedPermutation(perm),
        };

    } else {
      LOG(FATAL) << "Unexpected line: " << line;
    }
  }

  for (const auto &[_, same] : info.all_codes) {
    CHECK(info.canonical.contains(same.canonical_code)) << filename;
  }

  return info;
}
