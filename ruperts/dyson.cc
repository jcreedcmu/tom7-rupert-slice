
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <format>
#include <initializer_list>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "ansi.h"
#include "arcfour.h"
#include "atomic-util.h"
#include "base/logging.h"
#include "base/stringprintf.h"
#include "mesh.h"
#include "opt/opt.h"
#include "periodically.h"
#include "randutil.h"
#include "timer.h"
#include "yocto_matht.h"
#include "polyhedra.h"

DECLARE_COUNTERS(iters, attempts);

using namespace yocto;

using vec3 = vec<double, 3>;
using frame3 = frame<double, 3>;
using quat4 = quat<double, 4>;

// We just represent the unit cube (0,0,0)-(1,1,1) as its
// rigid transformation.
constexpr int NUM_CUBES = 5;

constexpr double SPHERE_RADIUS = 10.0;
constexpr double SPHERE_SQ_RADIUS = SPHERE_RADIUS * SPHERE_RADIUS;

static void CubesToSTL(const std::array<frame3, NUM_CUBES> &cubes,
                       std::string_view filename) {
  std::vector<TriangularMesh3D> all;
  for (const frame3 &frame : cubes) {
    TriangularMesh3D mesh;
    auto Vertex = [&frame, &mesh](double x, double y, double z) {
        int id = mesh.vertices.size();
        mesh.vertices.push_back(
            transform_point(frame, vec3{.x = x, .y = y, .z = z}));
        return id;
      };

    auto AddFace = [&mesh](int a, int b, int c, int d) {
        //  a-b
        //  |/|
        //  d-c
        mesh.triangles.emplace_back(a, b, d);
        mesh.triangles.emplace_back(b, c, d);
      };

    int a = Vertex(0.0, 1.0, 1.0);
    int b = Vertex(1.0, 1.0, 1.0);
    int c = Vertex(1.0, 1.0, 0.0);
    int d = Vertex(0.0, 1.0, 0.0);

    int e = Vertex(0.0, 0.0, 1.0);
    int f = Vertex(1.0, 0.0, 1.0);
    int g = Vertex(1.0, 0.0, 0.0);
    int h = Vertex(0.0, 0.0, 0.0);

    // top
    AddFace(a, b, c, d);
    // bottom
    AddFace(e, f, g, h);
    // left
    AddFace(a, e, h, d);
    // right
    AddFace(b, f, g, c);
    // front
    AddFace(d, c, g, h);
    // back
    AddFace(a, b, f, e);

    OrientMesh(&mesh);
    all.push_back(std::move(mesh));
  }

  all = {ApproximateSphere(1)};

  SaveAsSTL(ConcatMeshes(all), filename, "dyson");
}

// Compute the intersection between the segment p1-p2 and
// the unit cube. Nullopt if none. Otherwise, the interpolants
// (i.e. each in [0, 1]) tmin and tmax where the segment
// overlaps.
static std::optional<std::pair<double, double>>
SegmentIntersectsUnitCube(const vec3 &p1, const vec3 &p2) {
  const vec3 cube_min(0.0, 0.0, 0.0);
  const vec3 cube_max(1.0, 1.0, 1.0);

  // Direction vector of the segment
  vec3 d = p2 - p1;
  double t_min = -std::numeric_limits<double>::infinity();
  double t_max = std::numeric_limits<double>::infinity();

  constexpr double EPSILON = 1e-100;

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

  // Intersection.
  return {std::make_pair(t_min, t_max)};
}

namespace {
struct Eval {
  int num_edge_overlaps = 0;
  int num_points_outside = 0;
  // Non-negative costs
  double edge_overlap_sum = 0.0;
  double outside_sum = 0.0;
};
}

static Eval Evaluate(const std::array<frame3, NUM_CUBES> &cubes) {
  Eval eval;

  for (int idx1 = 0; idx1 < NUM_CUBES; idx1++) {
    const frame3 &cube1 = cubes[idx1];
    // cube1's vertices, after transformation.

    //                  +y
    //      a------b     | +z
    //     /|     /|     |/
    //    / |    / |     0--- +x
    //   d------c  |
    //   |  |   |  |
    //   |  e---|--f
    //   | /    | /
    //   |/     |/
    //   h------g

    auto Vertex = [&cube1](double x, double y, double z) {
        return transform_point(cube1, vec3{.x = x, .y = y, .z = z});
      };
    std::array<vec3, 8> v1;

    constexpr int a = 0, b = 1, c = 2, d = 3;
    constexpr int e = 4, f = 5, g = 6, h = 7;

    v1[a] = Vertex(0.0, 1.0, 1.0);
    v1[b] = Vertex(1.0, 1.0, 1.0);
    v1[c] = Vertex(1.0, 1.0, 0.0);
    v1[d] = Vertex(0.0, 1.0, 0.0);

    v1[e] = Vertex(0.0, 0.0, 1.0);
    v1[f] = Vertex(1.0, 0.0, 1.0);
    v1[g] = Vertex(1.0, 0.0, 0.0);
    v1[h] = Vertex(0.0, 0.0, 0.0);

    // Test if any vertices are outside the unit sphere.
    for (int i = 0; i < 8; i++) {
      double d = dot(v1[i], v1[i]);
      if (d >= SPHERE_SQ_RADIUS) {
        eval.num_points_outside++;
        eval.outside_sum += (d - SPHERE_SQ_RADIUS);
      }
    }

    std::initializer_list<std::pair<int, int>> edges = {
      {a, b},
      {a, d},
      {a, e},
      {c, d},
      {c, b},
      {c, g},
      {h, d},
      {h, e},
      {h, g},
      {f, b},
      {f, e},
      {f, g},
    };

    for (int idx2 = idx1 + 1; idx2 < NUM_CUBES; idx2++) {
      const frame3 &cube2 = cubes[idx2];

      // Put cube1's vertices in cube2's coordinate system.
      std::array<vec3, 8> v2;
      for (int i = 0; i < 8; i++) {
        v2[i] = transform_point_inverse(cube2, v1[i]);
      }

      // Now test each edge for intersection with the unit cube.
      for (const auto &[n0, n1] : edges) {
        if (std::optional<std::pair<double, double>> to =
            SegmentIntersectsUnitCube(v2[n0], v2[n1])) {
          eval.num_edge_overlaps++;
          const auto &[tmin, tmax] = to.value();
          CHECK(tmin <= tmax);
          eval.edge_overlap_sum += tmax - tmin;
        }
      }
    }
  }

  return eval;
}

static void Optimize() {
  ArcFour rc(std::format("dyson.{}", time(nullptr)));
  Timer run_time;

  static constexpr int ARGS_PER_CUBE = 7;
  static constexpr int NUM_ARGS = ARGS_PER_CUBE * NUM_CUBES;

  Periodically status_per(5.0);
  double best = std::numeric_limits<double>::infinity();

  // while (run_time.Seconds() < 60.0 * 60.0) {
  for (;;) {

    std::vector<quat4> initial_rot;
    for (int i = 0; i < NUM_CUBES; i++) {
      initial_rot.push_back(RandomQuaternion(&rc));
    }

    auto SetCubes = [&](const std::array<double, NUM_ARGS> &args,
                        std::array<frame3, NUM_CUBES> *cubes) {
        for (int i = 0; i < NUM_CUBES; i++) {
          const int base = ARGS_PER_CUBE * i;
          quat4 tweaked_rot = normalize(quat4{
              .x = initial_rot[i].x + args[base + 0],
              .y = initial_rot[i].y + args[base + 1],
              .z = initial_rot[i].z + args[base + 2],
              .w = initial_rot[i].w + args[base + 3],
            });
          frame3 frame = yocto::rotation_frame(tweaked_rot);
          frame.o.x = args[base + 4];
          frame.o.y = args[base + 5];
          frame.o.z = args[base + 6];
          (*cubes)[i] = frame;
        }
      };

    constexpr double Q = 0.15;

    std::array<double, NUM_ARGS> lb, ub;
    {
      int idx = 0;
      for (int i = 0; i < NUM_CUBES; i++) {
        for (int q = 0; q < 4; q++) {
          lb[idx] = -Q;
          ub[idx] = +Q;
          idx++;
        }

        for (int o = 0; o < 3; o++) {
          lb[idx] = -std::sqrt(SPHERE_SQ_RADIUS);
          ub[idx] = +std::sqrt(SPHERE_SQ_RADIUS);
          idx++;
        }
      }
      CHECK(idx == NUM_ARGS);
    }

    auto Loss = [&](const std::array<double, NUM_ARGS> &args) {
        std::array<frame3, NUM_CUBES> cubes;
        SetCubes(args, &cubes);
        Eval eval = Evaluate(cubes);
        attempts++;
        double loss =
          ((eval.num_points_outside + eval.num_edge_overlaps) * 1000.0) +
          eval.edge_overlap_sum + eval.outside_sum;
        return loss;
      };

    const auto &[args, error] =
      Opt::Minimize<NUM_ARGS>(Loss, lb, ub, 1000, 2);

    best = std::min(error, best);

    if (error <= 0.0) {
      printf("Solved!\n");
      std::array<frame3, NUM_CUBES> cubes;
      SetCubes(args, &cubes);

      for (int i = 0; i < NUM_CUBES; i++) {
        printf("Cube %d:\n%s\n", i, FrameString(cubes[i]).c_str());
      }

      CubesToSTL(cubes, "dyson.stl");

      return;
    }

    status_per.RunIf([&]() {
        printf("%lld iters, %lld attempts. Best: %.17g\n",
               iters.Read(), attempts.Read(), best);
      });
  }
}

int main(int argc, char **argv) {
  ANSI::Init();

  Optimize();

  return 0;
}
