
#include "base/stringprintf.h"
#include "polyhedra.h"
#include "smallest-sphere.h"

#include <cstdint>
#include <cstdio>
#include <format>
#include <string>
#include <vector>

#include "ansi.h"
#include "arcfour.h"
#include "base/logging.h"
#include "periodically.h"
#include "randutil.h"
#include "status-bar.h"
#include "yocto_matht.h"

using namespace yocto;

using vec3 = vec<double, 3>;
using frame3 = frame<double, 3>;
using quat4 = quat<double, 4>;

static void TestRandom() {
  // Only use this for generating the points
  // to test, so that we can repeat failures if needed.
  ArcFour rc_seq("test");
  // ... but the algorithm also needs an RNG.
  ArcFour rc_algo("algo");

  double total_sec = 0.0;
  Periodically status_per(1.0);
  StatusBar status(1);
  // XXX increase limit after fixing bugs
  int LIMIT = 100000;
  for (int i = 0; i < LIMIT; i++) {
    int n = RandTo(&rc_seq, 100);
    std::vector<vec3> pts;
    pts.reserve(n);
    for (int j = 0; j < n; j++) {
      double x = RandDouble(&rc_seq) * 4 - 2.0;
      double y = RandDouble(&rc_seq) * 4 - 2.0;
      double z = RandDouble(&rc_seq) * 4 - 2.0;
      pts.emplace_back(x, y, z);
    }

    Timer run_timer;
    const auto &[o, r] = SmallestSphere::Smallest(&rc_algo, pts);
    total_sec += run_timer.Seconds();
    CHECK(r >= 0.0);
    for (const vec3 &v : pts) {
      double vr = distance(v, o);
      // We allow it to be slightly outside to account for numerical
      // issues.
      CHECK(r - vr > -1.0e-10) << "Want " << vr << " <= " << r;
    }

    if (i % 1024 == 0) {
      status_per.RunIf([&]() {
          status.Progressf(i, LIMIT, "Testing random point clouds");
        });
    }
  }

  printf("Random OK (avg time: %s).\n",
         ANSI::Time(total_sec / LIMIT).c_str());
}

// Three points is a special case, which is a bit hard to encounter
// with fully random points. So test it explicitly.
static void TestRandom3() {
  // Only use this for generating the points
  // to test, so that we can repeat failures if needed.
  ArcFour rc_seq("test");
  // ... but the algorithm also needs an RNG.
  ArcFour rc_algo("algo");

  Periodically status_per(1.0);
  StatusBar status(1);
  static constexpr int LIMIT = 10'000'000;
  for (int i = 0; i < LIMIT; i++) {
    std::vector<vec3> pts;
    pts.reserve(3);
    for (int j = 0; j < 3; j++) {
      double x = RandDouble(&rc_seq) * 4 - 2.0;
      double y = RandDouble(&rc_seq) * 4 - 2.0;
      double z = RandDouble(&rc_seq) * 4 - 2.0;
      pts.emplace_back(x, y, z);
    }

    const auto &[o, r] = SmallestSphere::Smallest(&rc_algo, pts);
    CHECK(r >= 0.0);
    for (const vec3 &v : pts) {
      double vr = distance(v, o);
      // We allow it to be slightly outside to account for numerical
      // issues.
      CHECK(r - vr > -1.0e-10) << "Want " << vr << " <= " << r;
    }

    status_per.RunIf([&]() {
        status.Progressf(i, LIMIT, "Random3");
      });
  }

  printf("Random3 OK.\n");
}

static void TestRandomCubes() {
  ArcFour rc("test");

  constexpr int64_t LIMIT = 10'000'000;
  double total_sec = 0.0;
  Periodically status_per(1.0);
  StatusBar status(1);
  for (int64_t i = 0; i < LIMIT; i++) {
    // if (i == 20356) SmallestSphere::verbose = true;

    // int num_cubes = 2 + RandTo(&rc, 3);
    const int num_cubes = 1;
    std::vector<vec3> all_points;
    all_points.reserve(num_cubes * 8);

    auto ShowPoints = [&all_points]() {
        std::string ret;
        AppendFormat(&ret, "There are {} points:\n",
                     (int)all_points.size());
        for (const vec3 &v : all_points) {
          AppendFormat(&ret, "({}, {}, {})\n", v.x, v.y, v.z);
        }
        return ret;
      };

    // Make a unit cube.
    for (int cube_idx = 0; cube_idx < num_cubes; cube_idx++) {
      frame3 frame = yocto::rotation_frame(RandomQuaternion(&rc));
      frame.o.x = RandDouble(&rc) * num_cubes * 2 - num_cubes;
      frame.o.y = RandDouble(&rc) * num_cubes * 2 - num_cubes;
      frame.o.z = RandDouble(&rc) * num_cubes * 2 - num_cubes;

      auto Vertex = [&all_points, &frame](double x, double y, double z) {
          all_points.push_back(
              transform_point(frame, vec3{.x = x, .y = y, .z = z}));
        };

      for (uint8_t b = 0b000; b < 0b1000; b++) {
        Vertex(!!(b & 0b100), !!(b & 0b010), !! (b & 0b001));
      }
    }

    Timer run_timer;
    const auto &[center, radius] = SmallestSphere::Smallest(&rc, all_points);
    total_sec += run_timer.Seconds();

    // Now check it.
    for (const vec3 &v : all_points) {
      double d = distance(center, v);
      CHECK(radius - d > -1.0e-10) << "Want " << d << " <= " << radius << "\n"
                                   << "On iter " << i << "\nPoints:\n"
                                   << ShowPoints()
                                   // << "Got sphere:\n"
                                   << std::format("sphere(({}, {}, {}), {})\n",
                                                  center.x,
                                                  center.y,
                                                  center.z,
                                                  radius);
    }

    if (i % 1024 == 0) {
      status_per.RunIf([&]() {
          status.Progressf(i, LIMIT, "cubes");
        });
    }
  }

  printf("Random cubes OK (avg time: %s).\n",
         ANSI::Time(total_sec / LIMIT).c_str());
}


int main(int argc, char **argv) {
  ANSI::Init();

  TestRandom3();
  TestRandom();
  TestRandomCubes();

  return 0;
}
