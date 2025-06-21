
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
#include <numbers>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include "ansi.h"
#include "arcfour.h"
#include "atomic-util.h"
#include "base/logging.h"
#include "dyson.h"
#include "factorization.h"
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

static constexpr bool VERBOSE = false;

using namespace yocto;

using vec3 = vec<double, 3>;
using frame3 = frame<double, 3>;
using quat4 = quat<double, 4>;

// We just represent the unit cube (0,0,0)-(1,1,1) as its
// rigid transformation.
static constexpr int NUM_CUBES = 6;

static constexpr int NUM_THREADS = 8;
static StatusBar status = StatusBar(NUM_THREADS + 2);

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

  {
    // 2x2 with one centered above
    frame3 cube1 = translation_frame(vec3{0, 0, 0});
    frame3 cube2 = translation_frame(vec3{1, 0, 0});
    frame3 cube3 = translation_frame(vec3{0, 1, 0});
    frame3 cube4 = translation_frame(vec3{1, 1, 0});
    frame3 cube5 = translation_frame(vec3{0.5, 0.5, 1.0});

    v.push_back(std::array<frame3, 5>(
                    {cube1, cube2, cube3, cube4, cube5}));
  }

  {
    // [ | ]
    //  [ ]
    // with one centered above and below.
    frame3 cube1 = translation_frame(vec3{0, 0, 0});
    frame3 cube2 = translation_frame(vec3{1, 0, 0});
    frame3 cube3 = translation_frame(vec3{0.5, 1, 0});
    // above and below
    frame3 cube4 = translation_frame(vec3{0.5, 0.5, 1});
    frame3 cube5 = translation_frame(vec3{0.5, 0.5, -1});

    v.push_back(std::array<frame3, 5>(
                    {cube1, cube2, cube3, cube4, cube5}));
  }

  return v;
}

[[maybe_unused]]
static std::vector<std::array<frame3, 6>> Manual6() {
  std::vector<std::array<frame3, 6>> v;

  const frame3 z45 =
    translation_frame(vec3{.5, .5, 0}) *
    rotation_frame(vec3{0, 0, 1}, 0.785398) *
    translation_frame(vec3{-.5, -.5, 0});

  {
    // 2x2 with one centered above and below
    frame3 cube1 = translation_frame(vec3{0, 0, 0});
    frame3 cube2 = translation_frame(vec3{1, 0, 0});
    frame3 cube3 = translation_frame(vec3{0, 1, 0});
    frame3 cube4 = translation_frame(vec3{1, 1, 0});
    frame3 cube5 = translation_frame(vec3{0.5, 0.5, 1.0});
    frame3 cube6 = translation_frame(vec3{0.5, 0.5, -1.0});

    v.push_back(std::array<frame3, 6>(
                    {cube1, cube2, cube3, cube4, cube5, cube6}));
  }

  {
    // 2x2 with two above
    frame3 cube1 = translation_frame(vec3{0, 0, 0});
    frame3 cube2 = translation_frame(vec3{1, 0, 0});
    frame3 cube3 = translation_frame(vec3{0, 1, 0});
    frame3 cube4 = translation_frame(vec3{1, 1, 0});
    frame3 cube5 = translation_frame(vec3{0, 0.5, 1.0});
    frame3 cube6 = translation_frame(vec3{1, 0.5, 1.0});

    v.push_back(std::array<frame3, 6>(
                    {cube1, cube2, cube3, cube4, cube5, cube6}));
  }

  {
    frame3 cube1 = translation_frame(vec3{0, 0, 0});
    frame3 cube2 = translation_frame(vec3{0, 1, 0});
    frame3 cube3 = translation_frame(vec3{1, -.1, 0});
    frame3 cube4 = translation_frame(vec3{1, 0.9, 0.5});
    frame3 cube5 = translation_frame(vec3{0.3, 0.3, -1}) * z45;
    frame3 cube6 = translation_frame(vec3{1.25, 1.15, -0.5}) * z45;

    v.push_back(std::array<frame3, 6>(
                    {cube1, cube2, cube3, cube4, cube5, cube6}));
  }

  return v;
}

[[maybe_unused]]
static std::vector<std::array<frame3, 7>> Manual7() {
  std::vector<std::array<frame3, 7>> v;

  {
    // 2x2 with three above
    frame3 cube1 = translation_frame(vec3{0, 0, 0});
    frame3 cube2 = translation_frame(vec3{1, 0, 0});
    frame3 cube3 = translation_frame(vec3{0, 1, 0});
    frame3 cube4 = translation_frame(vec3{1, 1, 0});

    frame3 cube5 = translation_frame(vec3{0, 0, 1});
    frame3 cube6 = translation_frame(vec3{1, 0, 1});
    frame3 cube7 = translation_frame(vec3{0.5, 1, 1});

    v.push_back(std::array<frame3, 7>(
                    {cube1, cube2, cube3, cube4, cube5, cube6, cube7}));
  }

  {
    // plus shape
    frame3 cube1 = translation_frame(vec3{0, 0, 0});
    frame3 cube2 = translation_frame(vec3{1, 0, 0});
    frame3 cube3 = translation_frame(vec3{0, 1, 0});
    frame3 cube4 = translation_frame(vec3{0, 0, 1});
    frame3 cube5 = translation_frame(vec3{-1, 0, 0});
    frame3 cube6 = translation_frame(vec3{0, -1, 0});
    frame3 cube7 = translation_frame(vec3{0, 0, -1});
    v.push_back(std::array<frame3, 7>(
                    {cube1, cube2, cube3, cube4, cube5, cube6, cube7}));
  }

  return v;
}

[[maybe_unused]]
static std::vector<std::array<frame3, 8>> Manual8() {
  std::vector<std::array<frame3, 8>> v;

  {
    // 2x2x2
    frame3 cube1 = translation_frame(vec3{0, 0, 0});
    frame3 cube2 = translation_frame(vec3{1, 0, 0});
    frame3 cube3 = translation_frame(vec3{0, 1, 0});
    frame3 cube4 = translation_frame(vec3{1, 1, 0});
    frame3 cube5 = translation_frame(vec3{0, 0, 1});
    frame3 cube6 = translation_frame(vec3{1, 0, 1});
    frame3 cube7 = translation_frame(vec3{0, 1, 1});
    frame3 cube8 = translation_frame(vec3{1, 1, 1});


    v.push_back(std::array<frame3, 8>(
                    {cube1, cube2, cube3, cube4, cube5, cube6, cube7, cube8}));
  }

  return v;
}


static auto Manual() {
  if constexpr (NUM_CUBES == 1) {
    return {translation_frame(vec3{0, 0, 0})};
  } else if constexpr (NUM_CUBES == 3) {
    return Manual3();
  } else if constexpr (NUM_CUBES == 4) {
    return Manual4();
  } else if constexpr (NUM_CUBES == 5) {
    return Manual5();
  } else if constexpr (NUM_CUBES == 6) {
    return Manual6();
  } else if constexpr (NUM_CUBES == 7) {
    return Manual7();
  } else if constexpr (NUM_CUBES == 8) {
    return Manual8();
  } else {
    CHECK(NUM_CUBES > 1);

    std::vector<std::pair<uint64_t, int>> factors =
      Factorization::Factorize(NUM_CUBES);

    // We will create a JxKxL cube of cubes; find J, K, L.
    int j = 1, k = 1, l = 1;
    for (int idx = factors.size() - 1; idx >= 0; idx--) {
      while (factors[idx].second > 0) {
        if (j <= k && j <= l) {
          j *= factors[idx].first;
        } else if (k <= j && k <= l) {
          k *= factors[idx].first;
        } else {
          l *= factors[idx].first;
        }

        factors[idx].second--;
      }
    }

    status.Print("Manual cube: {}x{}x{}\n", j, k, l);

    std::array<frame3, NUM_CUBES> arr;
    int idx = 0;
    for (int jj = 0; jj < j; jj++) {
      for (int kk = 0; kk < k; kk++) {
        for (int ll = 0; ll < l; ll++) {
          arr[idx++] = translation_frame(vec3(jj, kk, ll));
        }
      }
    }

    return std::vector<std::array<frame3, NUM_CUBES>>{arr};
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

  Timer smallest_timer;
  eval.sphere = SmallestSphere::Smallest(rc, all_points);
  double t = smallest_timer.Seconds();
  if (t > 1) {
    status.Print("Slow sphere call ({}):\n", ANSI::Time(t).c_str());
    for (const vec3 &pt : all_points) {
      status.Print("{{{:.17g}, {:.17g}, {:.17g}}},\n",
                   pt.x, pt.y, pt.z);
    }
  }
  return eval;
}

struct DelayedWrite {
  explicit DelayedWrite(double sec) : delay_time(sec) {}

  void Tick() {
    MutexLock ml(&m);
    if (f && timer.Seconds() > delay_time) {
      f();
      f = std::function<void()>();
    }
  }

  void Delay(std::function<void()> ff) {
    MutexLock ml(&m);
    f = std::move(ff);
    timer.Reset();
  }

  double SecondsSinceImprovement() {
    MutexLock ml(&m);
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

// TODO: This could be a general purpose utility in cc-lib/opt.
// But it would want to save more metadata and be smarter
// about noise. We assume here that samples might be way over
// (processor contention) and approach from the bottom.
struct OptimizeTuner {
  struct Params {
    int iters = 1000;
    int depth = 2;
    int attempts = 1;
  };

  OptimizeTuner(double goal_sec) : goal_sec(goal_sec) {}

  Params GetParams() const {
    return cur;
  }

  void Tune(double took_sec) {
    // First try increasing depth.
    if (cur.depth < 3 &&
        took_sec < std::sqrt(goal_sec)) {
      cur.depth++;
      cur.iters = 1000;
      cur.attempts = 1;
    } else if (took_sec < goal_sec &&
               cur.iters < 10000) {
      cur.iters += 200;
    } else if (took_sec < goal_sec) {
      int64_t total_calls = cur.iters * cur.attempts;
      cur.attempts++;
      cur.iters = total_calls / cur.attempts;
    } else if (took_sec > goal_sec * 1.1) {
      if (cur.attempts > 1) {
        cur.attempts--;
      } else if (cur.iters > 1100) {
        cur.iters -= 100;
      } else if (cur.depth > 1) {
        cur.depth--;
      }
    }
  }

  std::string InfoString() {
    return std::format("({} it {} d {} at)",
                       cur.iters, cur.depth, cur.attempts);
  }

  double goal_sec = 1.0;
  Params cur = {};
};


struct Shrinkwrap {
  static constexpr int ARGS_PER_CUBE = 7;
  static constexpr int NUM_ARGS = ARGS_PER_CUBE * NUM_CUBES;
  std::vector<std::thread> threads;

  std::mutex mu;
  ShrinklutionDB db;
  static constexpr int METHOD = ShrinklutionDB::METHOD_RANDOM;
  static constexpr int MAX_GOOD = 10;
  std::vector<Good> good;
  DelayedWrite writer = DelayedWrite(15.0);
  Periodically status_per = Periodically(5.0);
  double best_error = std::numeric_limits<double>::infinity();
  Timer run_timer;
  double best_manual = std::numeric_limits<double>::infinity();

  Shrinkwrap() {
    std::vector<std::array<frame3, NUM_CUBES>> manual = Manual();
    ArcFour rc("manual");
    for (int m = 0; m < manual.size(); m++) {
      const std::array<frame3, NUM_CUBES> &cubes = manual[m];
      Eval eval = Evaluate(&rc, cubes);
      status.Print("Manual solution #{} has radius: {:.11g}\n", m,
                   eval.sphere.second);
      std::string filename =
        std::format("shrinkwrap-manual{}-{}.stl", m, NUM_CUBES);
      CubesToSTL(cubes, {eval.sphere}, filename);
      status.Print("Wrote " AGREEN("{}") "\n",
                   filename.c_str());
      double radius = eval.sphere.second;
      if (!db.HasSolutionWithRadius(NUM_CUBES, radius)) {
        status.Print("Add to database.\n");
        db.AddSolution(cubes, ShrinklutionDB::METHOD_MANUAL, 0, radius);
      }

      best_manual = std::min(radius, radius);
      // Since the manual solutions are actually touching, explode
      // everything a little so that optimization has some space
      // to work with.
      std::array<frame3, NUM_CUBES> ecubes = cubes;
      for (frame3 &f : ecubes) {
        // Put it at the origin.
        f.x -= eval.sphere.first.x;
        f.y -= eval.sphere.first.y;
        f.z -= eval.sphere.first.z;

        // Spread 'em out.
        f.x *= 1.001;
        f.y *= 1.001;
        f.z *= 1.001;
      }
      good.push_back(Good{.radius = radius, .cubes = cubes});
    }
  }

  void Run() {
    std::vector<std::string> lines;
    for (int i = 0; i < NUM_THREADS; i++) {
      threads.emplace_back(&Shrinkwrap::OptimizeThread, this, i);
      lines.push_back(std::format("Start thread {}", i));
    }

    lines.push_back("Start");
    lines.push_back("Start");
    status.EmitStatus(lines);
    for (std::thread &t : threads) {
      t.join();
    }
  }

  // Must hold lock.
  void MaybeStatus() {
    status_per.RunIf([&]() {
        status.LineStatusf(
            NUM_THREADS,
            "%lld it, %s att.  "
            "Invalid: " AORANGE("%lld") " [%s]",
            iters.Read(), FormatNum(attempts.Read()).c_str(),
            invalid.Read(),
            ANSI::Time(run_timer.Seconds()).c_str());
        status.LineStatusf(
            NUM_THREADS + 1,
            "Best: %.14g "
            " %s ago. Best manual: %.14g",
            best_error,
            ANSI::Time(writer.SecondsSinceImprovement()).c_str(),
            best_manual);
        writer.Tick();
      });
  }

  // Add a good solution, but keep only the best ones.
  // Must hold lock.
  void AddGood(Good g) {
    good.emplace_back(std::move(g));
    std::sort(good.begin(), good.end(),
              [](const auto &a, const auto &b) {
                return a.radius < b.radius;
              });
    if (good.size() > MAX_GOOD) {
      good.resize(MAX_GOOD);
    }
  }

  // Holding lock.
  static void InitializeFromGood(const Good &good,
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
  }

  // PERF:
  // Normally we would translate, rotate, and translate
  // back. But since the optimization contains its own
  // translation, we could just bake that in.
  void SetCubes(const std::vector<quat4> &initial_rot,
                const std::array<double, NUM_ARGS> &args,
                std::array<frame3, NUM_CUBES> *cubes) const {
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
  }

  void OptimizeThread(int thread_idx) {
    ArcFour rc(std::format("{}.shrinkwrap.{}",
                           thread_idx, time(nullptr)));
    Periodically my_status_per(5.0);

    // Try to optimize for ten seconds per pass.
    OptimizeTuner plain_tuner(10);
    OptimizeTuner sa_tuner(10);

    for (int num_this_thread = 0; true; num_this_thread++) {

      // Pick a method randomly.
      // static constexpr bool USE_LARGE_OPTIMIZER = true;
      const bool USE_LARGE_OPTIMIZER = NUM_CUBES > 8;
      // false; // (thread_idx % 3) != 0;
      const int method = [&]() {
          switch (rc.Byte() & 7) {
          default:
          case 0:
          case 1:
          case 2:
          case 3:
          case 4:
            return ShrinklutionDB::METHOD_RANDOM;
          case 5:
          case 6:
            return ShrinklutionDB::METHOD_SAME_ANGLE;
          }
        }();

      std::vector<quat4> initial_rot;
      std::array<double, NUM_ARGS> initial_args;

      // Output of below.
      std::array<double, NUM_ARGS> args;
      double error = 0.0;

      if (method == ShrinklutionDB::METHOD_RANDOM) {

        if (VERBOSE)
          status.LineStatusf(thread_idx, "random %d", num_this_thread);

        const bool can_reuse = [&]{
            MutexLock ml(&mu);
            return good.size() > std::max(MAX_GOOD >> 1, 1);
          }();

        bool did_reuse = false;
        if (can_reuse && method == ShrinklutionDB::METHOD_RANDOM &&
            (rc.Byte() & 1)) {
          did_reuse = true;
          MutexLock ml(&mu);
          int idx = RandTo(&rc, good.size());
          InitializeFromGood(good[idx], &initial_rot, &initial_args);

        } else {
          for (int i = 0; i < NUM_CUBES; i++) {
            initial_rot.push_back(RandomQuaternion(&rc));
          }

          for (double &a : initial_args) {
            a = 0.0;
          }
        }

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

        auto Loss = [this, &initial_rot, &rc](
            const std::array<double, NUM_ARGS> &args) -> double {
            std::array<frame3, NUM_CUBES> cubes;
            SetCubes(initial_rot, args, &cubes);
            Eval eval = Evaluate(&rc, cubes);
            attempts++;
            double loss =
              (eval.num_edge_overlaps * 1000.0) +
              eval.edge_overlap_sum +
              eval.sphere.second;
            // eval.max_sq_distance;
            return loss;
          };

        if (USE_LARGE_OPTIMIZER) {
          using LOpt = LargeOptimizer<false>;

          status.LineStatusf(thread_idx, "Thread %d lopt [#%d]",
                             thread_idx, num_this_thread);

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

          // square scale to prefer shorter times
          double scale = RandDouble(&rc);
          double time_limit = 9.0 + scale * scale * 120.0;

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
              SetCubes(initial_rot, args, &cubes);
              std::vector<vec3> all_points =
                Dyson::CubesToPoints(cubes);

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
          }

          {
            auto best = lopt.GetBest();
            CHECK(best.has_value());
            const auto &[vargs, e] = best.value();
            for (int i = 0; i < NUM_ARGS; i++) args[i] = vargs[i];
            error = e;
          }

        } else {

          std::string tuner_info = plain_tuner.InfoString();

          Timer opt_timer;
          auto params = plain_tuner.GetParams();

          if (VERBOSE)
            status.LineStatusf(thread_idx, "opt #%d with: i %d d %d a %d",
                               num_this_thread,
                               params.iters,
                               params.depth,
                               params.attempts);

          std::tie(args, error) =
            Opt::Minimize<NUM_ARGS>(
                Loss, lb, ub,
                params.iters, params.depth, params.attempts);

          double opt_sec = opt_timer.Seconds();
          plain_tuner.Tune(opt_sec);

          my_status_per.RunIf([&]() {
              status.LineStatusf(
                  thread_idx,
                  AGREY(" opt")
                  "   "
                  "last %.6g, "
                  "%s ea. %s\n",
                  error,
                  ANSI::Time(opt_sec).c_str(),
                  tuner_info.c_str());
            });
        }

      } else if (method == ShrinklutionDB::METHOD_SAME_ANGLE) {

        if (VERBOSE)
          status.LineStatusf(thread_idx, "same angle");

        // Optimize all positions (but one), and only one angle
        // shared by all.
        static constexpr int SA_NUM_ARGS =
          // The angle (quaternion params).
          4 +
          // Pos, but leave one at its original location.
          (NUM_CUBES - 1) * 3;

        quat4 initial = RandomQuaternion(&rc);
        for (int i = 0; i < NUM_CUBES; i++) {
          initial_rot.push_back(initial);
        }

        std::array<double, SA_NUM_ARGS> lb, ub;
        {
          // One angle.
          int idx = 0;
          for (int i = 0; i < 4; i++) {
            lb[idx] = -0.25;
            ub[idx] = +0.25;
            idx++;
          }

          for (int i = 0; i < NUM_CUBES - 1; i++) {
            for (int o = 0; o < 3; o++) {
              lb[idx] = -NUM_CUBES; // radius;
              ub[idx] = +NUM_CUBES; // +radius;
              idx++;
            }
          }
          CHECK(idx == SA_NUM_ARGS);
        }

        auto SaMakeArgs =
          [&](const std::array<double, SA_NUM_ARGS> &sa_args) ->
          std::array<double, NUM_ARGS> {
          std::array<double, NUM_ARGS> args;
          for (double &d : args) d = -100000;

          // Every cube with the same angle.
          for (int i = 0; i < NUM_CUBES; i++) {
            for (int q = 0; q < 4; q++) {
              args[i * ARGS_PER_CUBE + q] = sa_args[q];
            }
          }

          // First cube.
          for (int o = 0; o < 3; o++)
            args[4 + o] = 0.0;

          // Skip first cube, which has fixed position.
          for (int i = 1; i < NUM_CUBES; i++) {
            for (int o = 0; o < 3; o++) {
              int idx = i * ARGS_PER_CUBE + 4 + o;
              CHECK(idx < NUM_ARGS);
              int sidx = 4 + (i - 1) * 3 + o;
              CHECK(sidx < SA_NUM_ARGS);
              args[idx] = sa_args[sidx];
            }
          }

          return args;
        };

        auto SaLoss = [&](const std::array<double, SA_NUM_ARGS> &sa_args) {

            std::array<double, NUM_ARGS> args = SaMakeArgs(sa_args);

            std::array<frame3, NUM_CUBES> cubes;
            SetCubes(initial_rot, args, &cubes);
            Eval eval = Evaluate(&rc, cubes);
            attempts++;
            double loss =
              (eval.num_edge_overlaps * 1000.0) +
              eval.edge_overlap_sum +
              eval.sphere.second;
            return loss;
          };

        std::string tuner_info = sa_tuner.InfoString();

        Timer opt_timer;
        auto params = sa_tuner.GetParams();
        const auto &[sa_args, err] =
          Opt::Minimize<SA_NUM_ARGS>(
              SaLoss, lb, ub,
              params.iters, params.depth, params.attempts);
        double opt_sec = opt_timer.Seconds();
        sa_tuner.Tune(opt_sec);

        args = SaMakeArgs(sa_args);
        error = err;

        my_status_per.RunIf([&]() {
            status.LineStatusf(
                thread_idx,
                AGREY("sa ") APURPLE("∢")
                "   "
                "last %.6g, "
                "%s ea. %s\n",
                error,
                ANSI::Time(opt_sec).c_str(),
                tuner_info.c_str());
          });

      }

      if (VERBOSE)
        status.LineStatusf(thread_idx, "observe");
      ObserveSolution(initial_rot, args, error, method);
    }
  }

  void ObserveSolution(const std::vector<quat4> &initial_rot,
                       const std::array<double, NUM_ARGS> &args,
                       double error,
                       int method) {

    MutexLock ml(&mu);
    if (error < best_error) {
      ArcFour rc("sol");
      std::array<frame3, NUM_CUBES> cubes;
      SetCubes(initial_rot, args, &cubes);
      Eval eval = Evaluate(&rc, cubes);

      double radius = eval.sphere.second;
      double volume = (4.0 * std::numbers::pi / 3.0) *
        radius * radius * radius;

      // If our best solution was't valid (overlap), then
      // reject it.
      if (eval.num_edge_overlaps > 0 ||
          // Impossible!
          volume < NUM_CUBES ||
          error < 0.001 ||
          // The issue I was checking for is likely fixed now,
          // but it is cheap to continue sanity checking that
          // the radius matches what we computed. Maybe better
          // would be to actually compute that all the points
          // are inside.
          radius < 0.99 * error) {
        status.Print(ARED("Invalid") " Got radius {:.6g} "
                     "but error {:.6g} from method {}", radius, error,
                     ShrinklutionDB::MethodName(method));
        for (int i = 0; i < NUM_CUBES; i++) {
          status.Print("Cube {}:\n{}\n", i, FrameString(cubes[i]));
        }
        invalid++;
      } else {
        best_error = error;
        status.Print("New best! {:.17g} with method " APURPLE("{}") "\n",
                     best_error,
                     ShrinklutionDB::MethodName(method));

        AddGood(Good{.radius = eval.sphere.second, .cubes = cubes});

        writer.Delay([this, method, c = std::move(cubes), eval]() {
          for (int i = 0; i < NUM_CUBES; i++) {
            status.Print("Cube {}:\n{}\n", i, FrameString(c[i]));
          }
          status.Print(ABLUE("Radius") ": {:.11g}\n", eval.sphere.second);
          std::string filename = std::format("shrinkwrap{}.stl", NUM_CUBES);
          CubesToSTL(c, {eval.sphere}, filename);
          status.Print("Wrote " AGREEN("{}") "\n", filename);
          db.AddSolution<NUM_CUBES>(c, method, 0, eval.sphere.second);
        });
      }
    }

    MaybeStatus();

    iters++;
  }

};

int main(int argc, char **argv) {
  ANSI::Init();

  Shrinkwrap shrink;
  shrink.Run();

  printf("Exit.\n");
  return 0;
}
