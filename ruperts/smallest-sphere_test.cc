
#include "smallest-sphere.h"

#include <cstdio>
#include <vector>

#include "ansi.h"
#include "arcfour.h"
#include "base/logging.h"
#include "periodically.h"
#include "randutil.h"
#include "status-bar.h"
#include "yocto_matht.h"

using vec3 = yocto::vec<double, 3>;

static void TestRandom() {
  // Only use this for generating the points
  // to test, so that we can repeat failures if needed.
  ArcFour rc_seq("test");
  // ... but the algorithm also needs an RNG.
  ArcFour rc_algo("algo");

  Periodically status_per(1.0);
  StatusBar status(1);
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

    const auto &[o, r] = SmallestSphere::Smallest(&rc_algo, pts);
    CHECK(r >= 0.0);
    for (const vec3 &v : pts) {
      double vr = distance(v, o);
      // We allow it to be slightly outside to account for numerical
      // issues.
      CHECK(r - vr > -1.0e10) << "Want " << vr << " <= " << r;
    }

    status_per.RunIf([&]() {
        status.Progressf(i, LIMIT, "Testing random point clouds");
      });
  }

  printf("Random OK.\n");
}

// Three points is a special case, which is a bit hard to encounter
// with fully random points. So test it explicitly.
static void TestRandom3() {
  // Only use this for generating the points
  // to test, so that we can repeat failures if needed.
  ArcFour rc_seq("test");
  // ... but the algorithm also needs an RNG.
  ArcFour rc_algo("algo");

  StatusBar status(1);
  for (int i = 0; i < 1000; i++) {
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
      CHECK(r - vr > -1.0e10) << "Want " << vr << " <= " << r;
    }

  }

  printf("Random3 OK.\n");
}


int main(int argc, char **argv) {
  ANSI::Init();

  TestRandom3();
  TestRandom();
  return 0;
}
