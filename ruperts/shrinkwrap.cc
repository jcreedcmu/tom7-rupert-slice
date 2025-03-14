
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
#include <utility>
#include <vector>

#include "ansi.h"
#include "arcfour.h"
#include "atomic-util.h"
#include "base/logging.h"
#include "base/stringprintf.h"
#include "dyson.h"
#include "mesh.h"
#include "opt/large-optimizer.h"
#include "opt/opt.h"
#include "periodically.h"
#include "polyhedra.h"
#include "randutil.h"
#include "shrinklutions.h"
#include "smallest-sphere.h"
#include "status-bar.h"
#include "threadutil.h"
#include "timer.h"
#include "yocto_matht.h"

DECLARE_COUNTERS(iters, attempts, invalid);

using namespace yocto;

using vec3 = vec<double, 3>;
using frame3 = frame<double, 3>;
using quat4 = quat<double, 4>;

// We just represent the unit cube (0,0,0)-(1,1,1) as its
// rigid transformation.
static constexpr int NUM_CUBES = 6;

[[maybe_unused]]
static std::vector<std::array<frame3, 3>> Manual3() {
  std::vector<std::array<frame3, 3>> v;
  frame3 cube1 = translation_frame(vec3{0, 0, 0});
  frame3 cube2 = translation_frame(vec3{1, 0, 0});
  frame3 cube3 = translation_frame(vec3{0.5, 0, 1});
  v.push_back(std::array<frame3, 3>(
                  {cube1, cube2, cube3}));
  return v;
}

[[maybe_unused]]
static std::vector<std::array<frame3, 4>> Manual4() {
  std::vector<std::array<frame3, 4>> v;

  {
    frame3 cube1 = translation_frame(vec3{0, 0, 0});
    frame3 cube2 = translation_frame(vec3{1, 0, 0});
    frame3 cube3 = translation_frame(vec3{0, 0, 1});
    frame3 cube4 = translation_frame(vec3{1, 0, 1});
    v.push_back(std::array<frame3, 4>(
                    {cube1, cube2, cube3, cube4}));
  }

  {
    frame3 cube1 = translation_frame(vec3{0, 0, 0});
    frame3 cube2 = translation_frame(vec3{1, 0, 0});
    frame3 cube3 = translation_frame(vec3{0.5, 0, 1});
    frame3 cube4 = translation_frame(vec3{0.5, 1.0, 0.333});
    v.push_back(std::array<frame3, 4>(
                    {cube1, cube2, cube3, cube4}));
  }

  return v;
}

[[maybe_unused]]
static std::vector<std::array<frame3, 5>> Manual5() {
  std::vector<std::array<frame3, 5>> v;

  // This is just a previous "good" solution
  frame3 cube1 =
    frame3{.x = vec3(0.61211709447829854, 0.15825284178178359, -0.77477009539310127),
    .y = vec3(-0.74299986347763747, -0.22026657940597877, -0.63200778228515631),
    .z = vec3(-0.2706729863131111, 0.96251684248369351, -0.017247098757847829),
    .o = vec3(0.39441862282094664, 0.46851371084117571, 0.13174313866636145)};
  frame3 cube2 =
    frame3{.x = vec3(0.43229774385230235, 0.19108955498876951, -0.88125106674227838),
    .y = vec3(-0.33618938046497532, 0.94098142162553888, 0.039123709156422271),
    .z = vec3(0.83671701376438246, 0.27935415896273014, 0.4710264246806547),
    .o = vec3(-0.65637438738788523, 1.1701364532038565, 0.35730899514265996)};
  frame3 cube3 =
    frame3{.x = vec3(0.53819386971636152, 0.27092628120546897, -0.79808916090365678),
    .y = vec3(-0.73801526919160232, -0.30582621247584646, -0.60150128030015437),
    .z = vec3(-0.40703909030924107, 0.91272628860729832, 0.035353939601859986),
    .o = vec3(0.43150042578266656, 0.42246780560695302, 1.5670040993203462)};
  frame3 cube4 =
    frame3{.x = vec3(0.74680790405057973, 0.2152719120160779, 0.62923442240910721),
    .y = vec3(0.56683394945785859, 0.28880227065978198, -0.7715520217093339),
    .z = vec3(-0.3478178089022852, 0.93287258098772341, 0.093656390341061893),
    .o = vec3(0.34853819868403635, 0.4375799903007862, 0.20496087038554472)};
  frame3 cube5 =
    frame3{.x = vec3(-0.36975367394300884, 0.9260670062271833, -0.07537984202072566),
    .y = vec3(0.60412404295338262, 0.17798393501889964, -0.77675984680005705),
    .z = vec3(-0.7059152649796625, -0.33274858204447777, -0.62526955772377302),
    .o = vec3(-0.28950678218380221, 0.2338036220003018, 0.83321855365986708)};

  v.push_back(std::array<frame3, 5>(
                  {cube1, cube2, cube3, cube4, cube5}));

  return v;
}

static auto Manual() {
  if constexpr (NUM_CUBES == 3) {
    return Manual3();
  } else if constexpr (NUM_CUBES == 4) {
    return Manual4();
  } else if constexpr (NUM_CUBES == 5) {
    return Manual5();
  } else {
    return std::vector<std::array<frame3, NUM_CUBES>>();
  }
}


// XXX factor it out
static void CubesToSTL(const std::array<frame3, NUM_CUBES> &cubes,
                       const std::optional<std::pair<vec3, double>> &sphere,
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

  if (sphere.has_value()) {
    const auto &[center, radius] = sphere.value();
    TriangularMesh3D geo = ApproximateSphere(2);
    for (vec3 &v : geo.vertices) {
      v *= radius;
      v += center;
    }
    all.push_back(geo);
  }

  // Also a tiny sphere at the origin.
  {
    TriangularMesh3D origin = ApproximateSphere(0);
    for (vec3 &v : origin.vertices) v *= 0.01;
    all.push_back(origin);
  }

  SaveAsSTL(ConcatMeshes(all), filename, "shrinkwrap", true);
}

namespace {
struct Eval {
  int num_edge_overlaps = 0;
  // Non-negative costs
  double edge_overlap_sum = 0.0;

  // Smallest sphere
  std::pair<vec3, double> sphere = {vec3{0, 0, 0}, 0};
};
}

static Eval Evaluate(ArcFour *rc,
                     const std::array<frame3, NUM_CUBES> &cubes) {
  Eval eval;

  std::vector<vec3> all_points;
  all_points.reserve(8 * NUM_CUBES);

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
      all_points.push_back(v1[i]);
      // double dist = dot(v1[i], v1[i]);
      // eval.max_sq_distance = std::max(eval.max_sq_distance, dist);
      // eval.max_sq_distance += dist;
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

    // PERF: We actually have to check both directions, since
    // one cube can intersect the other without any of *its*
    // edges intersecting (point into face). A faster thing
    // would be to just check whether any vertex is contained
    // (in one direction) or an edge is contained in the other.
    for (int idx2 = 0; idx2 < NUM_CUBES; idx2++) {
      if (idx1 == idx2) continue;

      const frame3 &cube2 = cubes[idx2];

      // Put cube1's vertices in cube2's coordinate system.
      std::array<vec3, 8> v2;
      for (int i = 0; i < 8; i++) {
        v2[i] = transform_point_inverse(cube2, v1[i]);
      }

      // Now test each edge for intersection with the unit cube.
      for (const auto &[n0, n1] : edges) {
        if (std::optional<std::pair<double, double>> to =
            Dyson::SegmentIntersectsUnitCube(v2[n0], v2[n1])) {
          eval.num_edge_overlaps++;
          const auto &[tmin, tmax] = to.value();
          CHECK(tmin <= tmax);
          eval.edge_overlap_sum += tmax - tmin;
        }
      }
    }
  }

  eval.sphere = SmallestSphere::Smallest(rc, all_points);
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

  double SecondsSinceImprovement() {
    std::unique_lock ml(m);
    return timer.Seconds();
  }

  std::mutex m;
  double delay_time = 1.0;
  Timer timer;
  std::function<void()> f;
};

struct Good {
  double radius = 0.0;
  std::array<frame3, NUM_CUBES> cubes;
};

static void Optimize() {
  static constexpr int ARGS_PER_CUBE = 7;
  static constexpr int NUM_ARGS = ARGS_PER_CUBE * NUM_CUBES;
  constexpr int NUM_THREADS = 6;

  StatusBar status(NUM_THREADS + 1);
  std::mutex mu;
  ShrinklutionDB db;
  static constexpr int METHOD = ShrinklutionDB::METHOD_RANDOM;
  static constexpr int MAX_GOOD = 10;
  std::vector<Good> good;
  DelayedWrite writer(15.0);
  Periodically status_per(5.0);
  double best_error = std::numeric_limits<double>::infinity();
  Timer run_timer;

  {
    std::vector<std::array<frame3, NUM_CUBES>> manual = Manual();
    ArcFour rc("manual");
    for (int m = 0; m < manual.size(); m++) {
      const std::array<frame3, NUM_CUBES> &cubes = manual[m];
      Eval eval = Evaluate(&rc, cubes);
      status.Printf("Manual solution #%d has radius: %.11g\n", m,
                    eval.sphere.second);
      std::string filename =
        std::format("shrinkwrap-manual{}-{}.stl", m, NUM_CUBES);
      CubesToSTL(cubes, {eval.sphere}, filename);
      status.Printf("Wrote " AGREEN("%s") "\n",
                    filename.c_str());
      good.push_back(Good{.radius = eval.sphere.second, .cubes = cubes});
    }
  }


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
              "%lld iters, %s attempts. Best: %.17g "
              " %s ago. "
              "Invalid: " AORANGE("%lld") " [%s]\n",
              iters.Read(), FormatNum(attempts.Read()).c_str(),
              best_error,
              ANSI::Time(writer.SecondsSinceImprovement()).c_str(),
              invalid.Read(),
              ANSI::Time(run_timer.Seconds()).c_str());
          writer.Tick();
        });
    };

  auto InitializeFromGood = [&](const Good &good,
                                std::vector<quat4> *initial_rot,
                                std::array<double, NUM_ARGS> *initial_args) {
      initial_rot->clear();
      for (int c = 0; c < NUM_CUBES; c++) {
        const frame3 &cube = good.cubes[c];
        initial_rot->push_back(Dyson::rotation_quat(cube));
        // four quaternion components, initially zero
        for (int i = 0; i < 4; i++)
          (*initial_args)[c * ARGS_PER_CUBE + i] = 0.0;
        // translation
        vec3 t = translation(cube);
        for (int i = 3; i < 3; i++)
          (*initial_args)[c * ARGS_PER_CUBE + 4 + i] = t[i];
      }
  };


  // while (run_timer.Seconds() < 60.0 * 60.0) {
  ParallelFan(
      NUM_THREADS,
      [&](int thread_idx) {
        ArcFour rc(std::format("{}.shrinkwrap.{}",
                               thread_idx, time(nullptr)));
        Periodically my_status_per(5.0);

        for (;;) {
          std::vector<quat4> initial_rot;
          std::array<double, NUM_ARGS> initial_args;

          const bool can_reuse = [&]{
              MutexLock ml(&mu);
              return good.size() > std::max(MAX_GOOD >> 1, 1);
            }();

          bool did_reuse = false;
          if (can_reuse && (rc.Byte() & 1)) {
            did_reuse = true;
            MutexLock ml(&mu);
            int idx = RandTo(&rc, good.size());
            InitializeFromGood(good[idx], &initial_rot, &initial_args);

          } else {
            for (int i = 0; i < NUM_CUBES; i++) {
              initial_rot.push_back(RandomQuaternion(&rc));
            }

            for (double &a : initial_args) a = 0.0;
          }


          // PERF:
          // Normally we would translate, rotate, and translate
          // back. But since the optimization contains its own
          // translation, we could just bake that in.

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

                (*cubes)[i] =
                  translation_frame(center) *
                  opt_frame *
                  translation_frame(-center);
              }
            };

          constexpr double Q = 0.25;

          std::array<double, NUM_ARGS> lb, ub;
          {
            // double radius = std::min(best_error, (double)NUM_CUBES);
            int idx = 0;
            for (int i = 0; i < NUM_CUBES; i++) {
              for (int q = 0; q < 4; q++) {
                lb[idx] = -Q;
                ub[idx] = +Q;
                idx++;
              }

              for (int o = 0; o < 3; o++) {
                lb[idx] = -NUM_CUBES; // radius;
                ub[idx] = +NUM_CUBES; // +radius;
                idx++;
              }
            }
            CHECK(idx == NUM_ARGS);
          }

          auto Loss = [&](const std::array<double, NUM_ARGS> &args) {
              std::array<frame3, NUM_CUBES> cubes;
              SetCubes(args, &cubes);
              Eval eval = Evaluate(&rc, cubes);
              attempts++;
              double loss =
                (eval.num_edge_overlaps * 1000.0) +
                eval.edge_overlap_sum +
                eval.sphere.second;
                // eval.max_sq_distance;
              return loss;
            };


          // static constexpr bool USE_LARGE_OPTIMIZER = true;
          const bool USE_LARGE_OPTIMIZER = (thread_idx % 3) != 0;

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
              std::vector<double> sample_args(NUM_ARGS);
              for (int i = 0; i < NUM_ARGS; i++) {
                sample_args[i] = initial_args[i];
              }
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

            double time_limit = 9.0 + RandDouble(&rc) * 120.0;

            while (opt_timer.Seconds() < time_limit) {
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
                // Then, optimize the cube that's furthest away
                // from the minimal sphere's center.
                auto best = lopt.GetBest();
                CHECK(best.has_value()) << "Everything is \"feasible\".";
                std::array<double, NUM_ARGS> args;
                for (int i = 0; i < NUM_ARGS; i++)
                  args[i] = best.value().first[i];
                std::array<frame3, NUM_CUBES> cubes;
                SetCubes(args, &cubes);
                std::vector<vec3> all_points;
                all_points.reserve(NUM_CUBES * 8);
                for (int i = 0; i < NUM_CUBES; i++) {
                  auto Vertex = [&](double x, double y, double z) {
                      vec3 v = transform_point(cubes[i],
                                               vec3{.x = x, .y = y, .z = z});
                      all_points.push_back(v);
                    };

                  for (uint8_t b = 0b000; b < 0b1000; b++) {
                    Vertex(b & 0b100, b & 0b010, b & 0b001);
                  }
                }

                Eval eval = Evaluate(&rc, cubes);

                std::optional<int> besti;
                double furthest = 0.0;
                for (int i = 0; i < NUM_CUBES; i++) {
                  for (int j = 0; j < 8; j++) {
                    const vec3 &v = all_points[i * 8 + j];
                    double sqdist = distance_squared(eval.sphere.first, v);
                    if (!besti.has_value() || sqdist > furthest) {
                      besti = {i};
                      furthest = sqdist;
                    }
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
                  "% 3d" ABLUE("×") " %s best %.6g, "
                  "%s + %s + %s + %s / %s\n",
                  passes,
                  did_reuse ? APURPLE("∞") : " ",
                  best.value().second,
                  // ANSI::Time(opt_timer.Seconds()).c_str(),
                  ANSI::Time(pos_seconds).c_str(),
                  ANSI::Time(sub_seconds).c_str(),
                  ANSI::Time(every_seconds).c_str(),
                  ANSI::Time(best_seconds).c_str(),
                  ANSI::Time(time_limit).c_str());

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

            Timer opt_timer;
            std::tie(args, error) =
              Opt::Minimize<NUM_ARGS>(Loss, lb, ub, 10000, 3);
            double opt_sec = opt_timer.Seconds();

            my_status_per.RunIf([&]() {
                status.LineStatusf(
                    thread_idx,
                    AGREY("---") ACYAN("o") "   best %.6g, "
                    "%s ea.\n",
                    best_error,
                    // ANSI::Time(opt_timer.Seconds()).c_str(),
                    ANSI::Time(opt_sec).c_str());
              });

          }

          {
            MutexLock ml(&mu);
            if (error < best_error) {
              std::array<frame3, NUM_CUBES> cubes;
              SetCubes(args, &cubes);
              Eval eval = Evaluate(&rc, cubes);
              // Sometimes the sphere is wrong (and we get a different
              // answer here). This is presumably a numerical issue in
              // smallest-sphere, like where we return 0 in some cases.
              // Just reject it if it's not consistent. (Probably fixed
              // now.)
              if (eval.sphere.second < 0.99 * error ||
                  eval.num_edge_overlaps > 0) {
                status.Printf(ARED("Invalid") "?!\n");
                invalid++;
              } else {
                best_error = error;
                status.Printf("New best! %.17g\n", best_error);

                good.emplace_back(Good{
                    .radius = eval.sphere.second,
                    .cubes = cubes
                  });

                std::sort(good.begin(), good.end(),
                          [](const auto &a, const auto &b) {
                            return a.radius < b.radius;
                          });
                if (good.size() > MAX_GOOD)
                  good.resize(MAX_GOOD);

                writer.Delay([&, c = std::move(cubes), eval]() {
                    for (int i = 0; i < NUM_CUBES; i++) {
                      status.Printf(
                          "Cube %d:\n%s\n",
                          i, FrameString(c[i]).c_str());
                    }
                    status.Printf(ABLUE("Radius") ": %.11g\n",
                                  eval.sphere.second);
                    std::string filename =
                      std::format("shrinkwrap{}.stl", NUM_CUBES);
                    CubesToSTL(c, {eval.sphere}, filename);
                    status.Printf("Wrote " AGREEN("%s") "\n",
                                  filename.c_str());
                    db.AddSolution<NUM_CUBES>(cubes, METHOD, 0,
                                              eval.sphere.second);
                  });
              }
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
