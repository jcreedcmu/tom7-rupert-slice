
#include "patches.h"

#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <format>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

#include "ansi.h"
#include "arcfour.h"
#include "base/logging.h"
#include "base/stringprintf.h"
#include "big-polyhedra.h"
#include "bignum/big.h"
#include "periodically.h"
#include "polyhedra.h"
#include "randutil.h"
#include "run-z3.h"
#include "status-bar.h"
#include "yocto_matht.h"
#include "z3.h"

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
    printf("Normal: %s\n", VecString(normal).c_str());
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

  return frame;
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
      uint64_t sample_code = boundaries.GetCode(bbs);
      if (sample_code == code) {
        // Sample is in range.
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
                     uint64_t code) {
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
    status_per.RunIf([&]() {
        status.Progressf(test_bit, num_bits, "%s",
                         std::format("Mask: {:b}", mask).c_str());
      });

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

    switch (RunZ3(out)) {
    case Z3Result::SAT:
      // Bit is necessary, since it can feasibly have the opposite
      // value with the current mask.
      break;
    case Z3Result::UNSAT:
      // Bit is unnecessary; its value is forced with the current mask.
      mask &= ~test_pos;
      break;
    case Z3Result::UNKNOWN:
      LOG(FATAL) << "Z3 at its limits?? " << out;
      break;
    }
  }

  printf("Reduced mask from %d to %d bits.\n",
         (int)num_bits, std::popcount<uint64_t>(mask));
  return mask;
}
