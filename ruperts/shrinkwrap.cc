
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <format>
#include <functional>
#include <initializer_list>
#include <limits>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "ansi.h"
#include "arcfour.h"
#include "atomic-util.h"
#include "base/logging.h"
#include "base/stringprintf.h"
#include "mesh.h"
#include "opt/large-optimizer.h"
#include "opt/opt.h"
#include "periodically.h"
#include "polyhedra.h"
#include "randutil.h"
#include "status-bar.h"
#include "threadutil.h"
#include "timer.h"
#include "yocto_matht.h"

DECLARE_COUNTERS(iters, attempts);

using namespace yocto;

using vec3 = vec<double, 3>;
using frame3 = frame<double, 3>;
using quat4 = quat<double, 4>;

// We just represent the unit cube (0,0,0)-(1,1,1) as its
// rigid transformation.
constexpr int NUM_CUBES = 3;

// XXX factor it out
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

  // all = {ApproximateSphere(1)};

  SaveAsSTL(ConcatMeshes(all), filename, "shrinkwrap", true);
}

// Compute the intersection between the segment p1-p2 and
// the unit cube. Nullopt if none. Otherwise, the interpolants
// (i.e. each in [0, 1]) tmin and tmax where the segment
// overlaps.
// TODO: Factor this out
static std::optional<std::pair<double, double>>
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

  // Intersection.
  return {std::make_pair(t_min, t_max)};
}

namespace {
struct Eval {
  int num_edge_overlaps = 0;
  // Non-negative costs
  double edge_overlap_sum = 0.0;

  double max_sq_distance = 0.0;
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
      double dist = dot(v1[i], v1[i]);
      // eval.max_sq_distance = std::max(eval.max_sq_distance, dist);
      eval.max_sq_distance += dist;
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

struct DelayedWrite {
  explicit DelayedWrite(double sec) : delay_time(sec) {}

  void Tick() {
    std::unique_lock ml(m);
    if (f && timer.Seconds() > delay_time) {
      f();
      f = std::function<void()>();
    }
  }

  void Delay(std::function<void()> ff) {
    std::unique_lock ml(m);
    f = std::move(ff);
    timer.Reset();
  }

  std::mutex m;
  double delay_time = 1.0;
  Timer timer;
  std::function<void()> f;
};

static void Optimize() {

  static constexpr int ARGS_PER_CUBE = 7;
  static constexpr int NUM_ARGS = ARGS_PER_CUBE * NUM_CUBES;
  constexpr int NUM_THREADS = 6;
  StatusBar status(NUM_THREADS + 1);

  DelayedWrite writer(15.0);
  Periodically status_per(5.0);
  std::mutex mu;
  double best_error = std::numeric_limits<double>::infinity();
  Timer run_timer;

  {
    std::vector<std::string> lines(NUM_THREADS, "?");
    lines.push_back("Start");
    status.EmitStatus(lines);
  }

  // Must hold lock.
  auto MaybeStatus = [&]() {
      status_per.RunIf([&]() {
          status.LineStatusf(
              NUM_THREADS,
              "%lld iters, %lld attempts. Best: %.17g [%s]\n",
              iters.Read(), attempts.Read(), best_error,
              ANSI::Time(run_timer.Seconds()).c_str());
          writer.Tick();
        });
    };


  // while (run_timer.Seconds() < 60.0 * 60.0) {
  ParallelFan(
      NUM_THREADS,
      [&](int thread_idx) {
        ArcFour rc(std::format("{}.shrinkwrap.{}",
                               thread_idx, time(nullptr)));

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

                constexpr vec3 center = {0.5, 0.5, 0.5};
                frame3 opt_frame = yocto::rotation_frame(tweaked_rot);
                opt_frame.o.x = args[base + 4];
                opt_frame.o.y = args[base + 5];
                opt_frame.o.z = args[base + 6];

                // Normally we would translate, rotate, and translate
                // back. But since the optimization contains its own
                // translation, we just bake that in.
                (*cubes)[i] = translation_frame(-center) * opt_frame;
              }
            };

          constexpr double Q = 0.25;

          std::array<double, NUM_ARGS> lb, ub;
          {
            double radius = std::min(best_error, (double)NUM_CUBES);
            int idx = 0;
            for (int i = 0; i < NUM_CUBES; i++) {
              for (int q = 0; q < 4; q++) {
                lb[idx] = -Q;
                ub[idx] = +Q;
                idx++;
              }

              for (int o = 0; o < 3; o++) {
                lb[idx] = 0.5 + -NUM_CUBES; // radius;
                ub[idx] = 0.5 + +NUM_CUBES; // +radius;
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
                (eval.num_edge_overlaps * 1000.0) +
                eval.edge_overlap_sum +
                eval.max_sq_distance;
              return loss;
            };


          static constexpr bool USE_LARGE_OPTIMIZER = true;

          std::array<double, NUM_ARGS> args;
          double error;

          if (USE_LARGE_OPTIMIZER) {
            using LOpt = LargeOptimizer<false>;

            LargeOptimizer lopt([&](const std::vector<double> &vargs) {
                std::array<double, NUM_ARGS> aargs;
                for (int i = 0; i < NUM_ARGS; i++) aargs[i] = vargs[i];
                return std::make_pair(Loss(aargs), true);
              }, NUM_ARGS, Rand64(&rc));

            {
              std::vector<double> sample_args(NUM_ARGS, 0.0);
              lopt.Sample(sample_args);
            }

            std::vector<LOpt::arginfo> largs(NUM_ARGS);
            for (int i = 0; i < NUM_ARGS; i++) {
              largs[i] = LOpt::Double(lb[i], ub[i]);
            }

            std::vector<bool> pos_mask(NUM_ARGS, false);
            for (int i = 0; i < NUM_CUBES; i++) {
              pos_mask[i * ARGS_PER_CUBE + 4 + 0] = true;
              pos_mask[i * ARGS_PER_CUBE + 4 + 1] = true;
              pos_mask[i * ARGS_PER_CUBE + 4 + 2] = true;
            }

            Timer opt_timer;
            int passes = 0;
            while (opt_timer.Seconds() < 12.0) {
              // First, optimize only positions.
              Timer pos_timer;
              lopt.Run(largs,
                       std::nullopt,
                       std::nullopt,
                       // max seconds
                       {1.0},
                       std::nullopt,
                       // parameters
                       NUM_CUBES * 3,
                       pos_mask);
              double pos_seconds = pos_timer.Seconds();

              // Then, optimize random subsets.
              Timer sub_timer;
              lopt.Run(largs,
                       std::nullopt,
                       std::nullopt,
                       // max seconds
                       {2.0},
                       std::nullopt,
                       // parameters
                       12);
              double sub_seconds = sub_timer.Seconds();

              // Then, optimize everything.
              Timer every_timer;
              lopt.Run(largs,
                       std::nullopt,
                       std::nullopt,
                       // max seconds
                       {1.0},
                       std::nullopt,
                       // parameters
                       NUM_ARGS);
              double every_seconds = every_timer.Seconds();

              Timer best_timer;
              {
                // Then, optimize the cube that's furthest away.
                auto best = lopt.GetBest();
                CHECK(best.has_value()) << "Everything is \"feasible\".";
                std::array<double, NUM_ARGS> args;
                for (int i = 0; i < NUM_ARGS; i++)
                  args[i] = best.value().first[i];
                std::array<frame3, NUM_CUBES> cubes;
                SetCubes(args, &cubes);
                std::optional<int> besti;
                double furthest = 0.0;
                for (int i = 0; i < NUM_CUBES; i++) {
                  auto Vertex = [&](double x, double y, double z) {
                      vec3 v = transform_point(cubes[i],
                                               vec3{.x = x, .y = y, .z = z});
                      double sqdist = dot(v, v);
                      if (!besti.has_value() || sqdist > furthest) {
                        besti = {i};
                        furthest = sqdist;
                      }
                    };

                  for (uint8_t b = 0b000; b < 0b1000; b++) {
                    Vertex(b & 0b100, b & 0b010, b & 0b001);
                  }
                }
                CHECK(besti.has_value());

                std::vector<bool> cube_mask(NUM_ARGS, false);
                for (int i = 0; i < ARGS_PER_CUBE; i++)
                  cube_mask[besti.value() * ARGS_PER_CUBE + i] = true;
                lopt.Run(largs,
                         std::nullopt,
                         std::nullopt,
                         // max seconds
                         {1.0},
                         std::nullopt,
                         // parameters
                         ARGS_PER_CUBE,
                         cube_mask);
              }
              double best_seconds = best_timer.Seconds();

              passes++;
              auto best = lopt.GetBest();
              CHECK(best.has_value());
              status.LineStatusf(
                  thread_idx,
                  "%d" ABLUE("×") " best %.6g, "
                  "%s + %s + %s + %s\n",
                  passes,
                  best.value().second,
                  // ANSI::Time(opt_timer.Seconds()).c_str(),
                  ANSI::Time(pos_seconds).c_str(),
                  ANSI::Time(sub_seconds).c_str(),
                  ANSI::Time(every_seconds).c_str(),
                  ANSI::Time(best_seconds).c_str());

              {
                MutexLock ml(&mu);
                MaybeStatus();
              }
              writer.Tick();
            }

            {
              auto best = lopt.GetBest();
              CHECK(best.has_value());
              const auto &[vargs, e] = best.value();
              for (int i = 0; i < NUM_ARGS; i++) args[i] = vargs[i];
              error = e;
            }

          } else {

            std::tie(args, error) =
              Opt::Minimize<NUM_ARGS>(Loss, lb, ub, 100000, 3);

          }

          {
            MutexLock ml(&mu);
            if (error < best_error) {
              best_error = error;
              status.Printf("New best! %.17g\n", best_error);
              std::array<frame3, NUM_CUBES> cubes;
              SetCubes(args, &cubes);
              writer.Delay([&, c = std::move(cubes)]() {
                  for (int i = 0; i < NUM_CUBES; i++) {
                    status.Printf(
                        "Cube %d:\n%s\n", i, FrameString(c[i]).c_str());
                  }
                  std::string filename = "shrinkwrap.stl";
                  CubesToSTL(c, filename);
                  status.Printf("Wrote " AGREEN("%s") "\n",
                                filename.c_str());
                });
            }

            MaybeStatus();
          }
          iters++;
        }
      });

  printf("Exit.\n");
}

int main(int argc, char **argv) {
  ANSI::Init();

  Optimize();

  return 0;
}
