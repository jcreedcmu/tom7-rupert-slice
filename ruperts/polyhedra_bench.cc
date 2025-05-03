
#include "periodically.h"
#include "polyhedra.h"

#include <cstdio>
#include <limits>
#include <optional>
#include <vector>

#include "ansi.h"
#include "arcfour.h"
#include "base/do-not-optimize.h"
#include "base/logging.h"
#include "status-bar.h"
#include "timer.h"
#include "yocto_matht.h"

[[maybe_unused]]
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
  HullInscribedCircle circle(souter.vertices, outer_hull);

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

static std::vector<vec2> GetPoly(const std::vector<vec2> &vertices,
                                      const std::vector<int> &hull) {
  std::vector<vec2> poly;
  poly.reserve(hull.size());
  for (int idx : hull) poly.push_back(vertices[idx]);
  return poly;
}

static void BenchPolyTester() {
  Polyhedron poly = SnubCube();


  ArcFour rc("deterministic");
  printf("Benchmark Samples:\n");
  constexpr int OUTER_ITERS = 10'000;
  constexpr int INNER_ITERS = 10'000;
  double total_query_sec = 0.0;
  Timer all_timer;
  StatusBar status(1);
  Periodically status_per(1);
  for (int o = 0; o < OUTER_ITERS; o++) {

    const frame3 outer_frame = yocto::rotation_frame(RandomQuaternion(&rc));
    // Note: Does not include translation
    const frame3 inner_frame = yocto::rotation_frame(RandomQuaternion(&rc));

    Mesh2D souter = Shadow(Rotate(poly, outer_frame));
    Mesh2D sinner = Shadow(Rotate(poly, inner_frame));

    // For snub cube, Graham Scan seems to be fastest.
    const std::vector<int> outer_hull = GrahamScan(souter.vertices);
    const std::vector<int> inner_hull = GrahamScan(sinner.vertices);

    std::vector<vec2> outer_poly = GetPoly(souter.vertices, outer_hull);
    std::vector<vec2> inner_poly = GetPoly(sinner.vertices, inner_hull);

    PolyTester2D tester(outer_poly);

    double err = 0.0;
    Timer inner_timer;
    for (int i = 0; i < INNER_ITERS; i++) {
      for (const vec2 &pt : inner_poly) {
        std::optional<double> sqd = tester.SquaredDistanceOutside(pt);
        if (sqd.has_value()) err += sqd.value();
      }
    }
    DoNotOptimize(err);
    total_query_sec += inner_timer.Seconds();
    status_per.RunIf([&]() {
        status.Progressf(o, OUTER_ITERS, "Benchmarking...");
      });
  }

  printf("Total time: %s\n"
         "Total query time: %s\n"
         "Time per query: %s\n",
         ANSI::Time(all_timer.Seconds()).c_str(),
         ANSI::Time(total_query_sec).c_str(),
         ANSI::Time(total_query_sec / (OUTER_ITERS * INNER_ITERS)).c_str());
}

int main(int argc, char **argv) {
  ANSI::Init();

  // BenchSample();
  BenchPolyTester();

  return 0;
}
