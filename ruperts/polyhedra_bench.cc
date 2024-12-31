
#include "polyhedra.h"

#include <cstdio>
#include <limits>
#include <vector>

#include "ansi.h"
#include "arcfour.h"
#include "base/do-not-optimize.h"
#include "base/logging.h"
#include "randutil.h"
#include "timer.h"
#include "yocto_matht.h"

static double Sample(const Polyhedron &poly, ArcFour *rc) {
  const frame3 outer_frame = yocto::rotation_frame(RandomQuaternion(rc));

  // Note: Does not include translation
  const frame3 inner_frame =
    yocto::rotation_frame(RandomQuaternion(rc));

  Mesh2D souter = Shadow(Rotate(poly, outer_frame));
  Mesh2D sinner = Shadow(Rotate(poly, inner_frame));

  // Does every vertex in inner fall inside the outer shadow?
  double error = 0.0;
  int errors = 0;
  for (const vec2 &iv : sinner.vertices) {
    if (!InMesh(souter, iv)) {
      // slow :(
      error += DistanceToMesh(souter, iv);
      errors++;
    }
  }

  if (error == 0.0 && errors > 0) [[unlikely]] {
    // If they are not in the mesh, don't return an actual zero.
    return std::numeric_limits<double>::min() * errors;
  } else {
    return error;
  }
}

static double SampleWithHull(const Polyhedron &poly, ArcFour *rc) {
  const frame3 outer_frame = yocto::rotation_frame(RandomQuaternion(rc));

  // Note: Does not include translation
  const frame3 inner_frame =
    yocto::rotation_frame(RandomQuaternion(rc));

  Mesh2D souter = Shadow(Rotate(poly, outer_frame));
  Mesh2D sinner = Shadow(Rotate(poly, inner_frame));

  // For snub cube, Graham Scan seems to be fastest.
  const std::vector<int> outer_hull = GrahamScan(souter.vertices);
  HullCircle circle(souter.vertices, outer_hull);

  // Does every vertex in inner fall inside the outer shadow?
  double error = 0.0;
  int errors = 0;
  for (const vec2 &iv : sinner.vertices) {
    // Quick test.
    if (circle.DefinitelyInside(iv))
      continue;

    if (!InHull(souter, outer_hull, iv)) {
      // slow :(
      error += DistanceToHull(souter.vertices, outer_hull, iv);
      errors++;
    }
  }

  if (error == 0.0 && errors > 0) [[unlikely]] {
    // If they are not in the mesh, don't return an actual zero.
    return std::numeric_limits<double>::min() * errors;
  } else {
    return error;
  }
}


static void BenchSample() {
  Polyhedron poly = SnubCube();

  ArcFour rc("deterministic");
  printf("Benchmark Samples:\n");
  Timer timer;
  constexpr int ITERS = 1'000'000;
  for (int i = 0; i < ITERS; i++) {
    DoNotOptimize(SampleWithHull(poly, &rc));
  }
  const double total_sec = timer.Seconds();

  printf("Did " AWHITE("%d") " iters in %s (%.2f ips)\n",
         ITERS, ANSI::Time(total_sec).c_str(),
         ITERS / total_sec);

  delete poly.faces;
}

int main(int argc, char **argv) {
  ANSI::Init();

  BenchSample();

  return 0;
}
