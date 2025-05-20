
#include "arcfour.h"
#include "patches.h"

#include <set>
#include <cstdint>
#include <cstdio>
#include <tuple>
#include <unordered_set>

#include "ansi.h"
#include "base/do-not-optimize.h"
#include "base/logging.h"
#include "big-polyhedra.h"
#include "randutil.h"
#include "timer.h"
#include "yocto_matht.h"

struct CompareVec3 {
  bool operator()(const vec3 &a, const vec3 &b) const {
    // This is not really a proper inequality, but it should
    // be fine for the points we are comparing.
    if (distance(a, b) < 0.01) return false;
    if (a.x < b.x) return true;
    if (a.x > b.x) return false;
    if (a.y < b.y) return true;
    if (a.y > b.y) return false;
    return a.z < b.z;
  }
};

static void TestSignedPerms() {

  {
    std::unordered_set<uint16_t> all;
    for (int i = 0; i < 24; i++) {
      all.insert(SignedPermutation::GetPerm(i).ToWord());
    }
    CHECK(all.size() == 24);
  }

  for (int i = 0; i < 24; i++) {
    yocto::mat<double, 3> m = SignedPermutation::GetPerm(i).ToMatrix();
    double det = determinant(m);
    CHECK(det == 1.0) << i << " got " << det;
  }

  {
    // Orbit check.

    // Should find all the points in a single pass.
    std::set<vec3, CompareVec3> seen1;
    for (int i = 0; i < 24; i++) {
      vec3 v = SignedPermutation::GetPerm(i).TransformPoint(vec3{1, 2, 3});
      CHECK(!seen1.contains(v));
      seen1.insert(v);
    }

    CHECK(seen1.size() == 24);

    // There should be no new points.
    std::set<vec3, CompareVec3> seen2;
    for (const vec3 &v : seen1) {
      for (int i = 0; i < 24; i++) {
        vec3 vv = SignedPermutation::GetPerm(i).TransformPoint(v);
        CHECK(seen1.contains(vv));
        seen2.insert(vv);
      }
    }

    CHECK(seen2.size() == 24);
  }
}

// Even though it is not smart about redundant plane-side tests,
// GetCodeSloppy is extremely fast... 108ns each.
static void BenchGetCode() {
  PatchInfo patch_info = LoadPatchInfo("scube-patchinfo.txt");
  Boundaries boundaries(BigScube(24));
  ArcFour rc("deterministic");

  Timer timer;
  static constexpr int ITERS = 10'000'000;
  uint64_t out = 0xBEEF;
  for (int i = 0; i < ITERS; i++) {
    vec3 v;
    std::tie(v.x, v.y, v.z) = RandomUnit3D(&rc);
    uint64_t code = boundaries.GetCodeSloppy(v);
    DoNotOptimize(code);
    out += code;
  }
  double sec = timer.Seconds();

  printf("Out: " ACYAN("%llx") "\n", out);
  printf("Ran %d in %s (%s ea.)\n",
         ITERS,
         ANSI::Time(sec).c_str(),
         ANSI::Time(sec / ITERS).c_str());
}

int main(int argc, char **argv) {
  ANSI::Init();

  TestSignedPerms();
  BenchGetCode();

  printf("OK\n");
  return 0;
}
