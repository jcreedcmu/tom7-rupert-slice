#ifndef _RUPERTS_DYSON_H
#define _RUPERTS_DYSON_H

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <utility>
#include <vector>

#include "yocto_matht.h"

struct Dyson {
  using vec3 = yocto::vec<double, 3>;
  using frame3 = yocto::frame<double, 3>;
  using quat4 = yocto::quat<double, 4>;
  using mat3 = yocto::mat<double, 3>;

  // Compute the intersection between the segment p1-p2 and
  // the unit cube. Nullopt if none. Otherwise, the interpolants
  // (i.e. each in [0, 1]) tmin and tmax where the segment
  // overlaps.
  inline static std::optional<std::pair<double, double>>
  SegmentIntersectsUnitCube(const vec3 &p1, const vec3 &p2) {
    const vec3 cube_min(0.0, 0.0, 0.0);
    const vec3 cube_max(1.0, 1.0, 1.0);

    // Direction vector of the segment
    vec3 d = p2 - p1;
    double t_min = -std::numeric_limits<double>::infinity();
    double t_max = std::numeric_limits<double>::infinity();

    constexpr double EPSILON = 1e-10;

    // Iterate through the three axes (x, y, z).
    for (int i = 0; i < 3; i++) {
      if (std::abs(d[i]) < EPSILON) {
        // Parallel segment.
        if (p1[i] < cube_min[i] || p1[i] > cube_max[i]) {
          return std::nullopt;
        }
      } else {
        // Compute intersection parameters with the near and far planes.
        double t1 = (cube_min[i] - p1[i]) / d[i];
        double t2 = (cube_max[i] - p1[i]) / d[i];

        // Swap so t1 <= t2.
        if (t1 > t2) {
          std::swap(t1, t2);
        }

        // Clip the intersection interval.
        t_min = std::max(t_min, t1);
        t_max = std::min(t_max, t2);

        // If the interval becomes empty, no intersection.
        if (t_min > t_max) {
          return std::nullopt;
        }
      }
    }

    if (t_min > 1.0) return std::nullopt;
    if (t_max < 0.0) return std::nullopt;

    // Restrict to the segment.
    t_min = std::max(t_min, 0.0);
    t_max = std::min(t_max, 1.0);

    // Intersection.
    return {std::make_pair(t_min, t_max)};
  }

  static std::vector<vec3> CubesToPoints(std::span<frame3> cubes) {
    std::vector<vec3> all_points;
    all_points.reserve(cubes.size() * 8);
    for (const frame3 &cube : cubes) {
      auto Vertex = [&](double x, double y, double z) {
          vec3 v = transform_point(cube,
                                   vec3{.x = x, .y = y, .z = z});
          all_points.push_back(v);
        };

      for (uint8_t b = 0b000; b < 0b1000; b++) {
        Vertex(b & 0b100, b & 0b010, b & 0b001);
      }
    }
    return all_points;
  }

  // Extract just the orientation from the rigid frame.
  static quat4 rotation_quat(const frame3 &frame) {
    // TODO PERF: These formulas actually wanted a row-major
    // matrix. We can just fix the formulas below...
    mat3 m = transpose(rotation(frame));
    double trace = m.x.x + m.y.y + m.z.z;
    if (trace > 0.0) {
      double s = 2.0 * std::sqrt(trace + 1.0);
      return quat4{
          .x = (m.z.y - m.y.z) / s,
          .y = (m.x.z - m.z.x) / s,
          .z = (m.y.x - m.x.y) / s,
          .w = 0.25 * s,
      };
    } else if (m.x.x > m.y.y && m.x.x > m.z.z) {
      double s = 2.0 * std::sqrt(1.0 + m.x.x - m.y.y - m.z.z);
      return quat4{
        .x = 0.25 * s,
        .y = (m.y.x + m.x.y) / s,
        .z = (m.z.x + m.x.z) / s,
        .w = (m.z.y - m.y.z) / s,
      };
    } else if (m.y.y > m.z.z) {
      auto s = 2.0 * std::sqrt(1.0 + m.y.y - m.x.x - m.z.z);
      return quat4{
        .x = (m.y.x + m.x.y) / s,
        .y = 0.25 * s,
        .z = (m.z.y + m.y.z) / s,
        .w = (m.x.z - m.z.x) / s,
      };
    } else {
      auto s = 2.0 * std::sqrt(1.0 + m.z.z - m.x.x - m.y.y);
      return quat4{
        .x = (m.x.z + m.z.x) / s,
        .y = (m.y.z + m.z.y) / s,
        .z = 0.25 * s,
        .w = (m.y.x - m.x.y) / s,
      };
    }
  }
};

#endif
