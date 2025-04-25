
// Like churro, but for distortions of the snub cube.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <format>
#include <functional>
#include <limits>
#include <mutex>
#include <numbers>
#include <optional>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include "ansi.h"
#include "arcfour.h"
#include "atomic-util.h"
#include "base/stringprintf.h"
#include "big-csg.h"
#include "big-polyhedra.h"
#include "bignum/big.h"
#include "geom/tree-3d.h"
#include "mesh.h"
#include "nd-solutions.h"
#include "opt/opt.h"
#include "periodically.h"
#include "polyhedra.h"
#include "randutil.h"
#include "rendering.h"
#include "status-bar.h"
#include "threadutil.h"
#include "timer.h"
#include "util.h"
#include "yocto_matht.h"

DECLARE_COUNTERS(solve_attempts, solved, hard, noperts, footballs);

static constexpr int NUM_THREADS = 16;
static constexpr int ADDL_STATUS_LINES = 3;

static StatusBar *status = nullptr;
static std::mutex thread_status_m;
static std::array<std::string, NUM_THREADS> thread_status;

using vec2 = yocto::vec<double, 2>;
using vec3 = yocto::vec<double, 3>;
using vec4 = yocto::vec<double, 4>;
using mat4 = yocto::mat<double, 4>;
using quat4 = yocto::quat<double, 4>;
using frame3 = yocto::frame<double, 3>;

static std::string ConfigString(
    double theta, double phi, double stretch) {
  return StringPrintf("%.6g" ABLUE("θ")
                      "%.6g" ACYAN("φ")
                      "%.6g" APURPLE("ρ") " ",
                      theta, phi, stretch);

}

// Return the number of iterations taken, or nullopt if we exceeded
// the limit. If solved and the arguments are non-null, sets the outer
// frame and inner frame to some solution.
// static constexpr int NOPERT_ITERS = 100000;
// This is about 30 minutes of optimization.
static constexpr int NOPERT_ITERS = 10000;
static constexpr int MIN_VERBOSE_ITERS = 2000;

// Returns best solution, iters, and clearance.
static std::optional<std::tuple<frame3, frame3, int64_t, double>>
DoSolve(int thread_idx,
        // Just used for debug print.
        double theta, double phi, double stretch,
        ArcFour *rc, const Polyhedron &poly) {
  CHECK(!poly.faces->v.empty());
  Timer solve_timer;
  {
    MutexLock ml(&thread_status_m);
    thread_status[thread_idx] =
      std::format("[" AYELLOW("{}") "] {}",
                  thread_idx,
                  ConfigString(theta, phi, stretch));
  }

  for (int iter = 0; iter < NOPERT_ITERS; iter++) {

    if (iter > 0 && (iter % 1000) == 0) {
      MutexLock ml(&thread_status_m);
      thread_status[thread_idx] =
        std::format("[" AYELLOW("{}") "] {} "
                    "{}"
                    ": " AWHITE("{}") " it " AGREY("({:.1f}%)") " {}",
                    thread_idx,
                    ConfigString(theta, phi, stretch),
                    (iter >= 20000) ? ARED("hard") : AORANGE("run"),
                    iter,
                    (100.0 * iter) / (double)NOPERT_ITERS,
                    ANSI::Time(solve_timer.Seconds()));

      if (iter == 2000) {
        hard++;
      }
    }

    // four params for outer rotation, four params for inner
    // rotation, two for 2d translation of inner.
    static constexpr int D = 10;

    // TODO: Favor the stretch axis for the inner one.
    const quat4 initial_outer_rot = RandomQuaternion(rc);
    const quat4 initial_inner_rot = RandomQuaternion(rc);

    // Get the frames from the appropriate positions in the
    // argument.

    auto OuterFrame = [&initial_outer_rot](
        const std::array<double, D> &args) {
        const auto &[o0, o1, o2, o3,
                     i0_, i1_, i2_, i3_, dx_, dy_] = args;
        quat4 tweaked_rot = normalize(quat4{
            .x = initial_outer_rot.x + o0,
            .y = initial_outer_rot.y + o1,
            .z = initial_outer_rot.z + o2,
            .w = initial_outer_rot.w + o3,
          });
        return yocto::rotation_frame(tweaked_rot);
      };

    auto InnerFrame = [&initial_inner_rot](
        const std::array<double, D> &args) {
        const auto &[o0_, o1_, o2_, o3_,
                     i0, i1, i2, i3, dx, dy] = args;
        quat4 tweaked_rot = normalize(quat4{
            .x = initial_inner_rot.x + i0,
            .y = initial_inner_rot.y + i1,
            .z = initial_inner_rot.z + i2,
            .w = initial_inner_rot.w + i3,
          });
        frame3 frame = yocto::rotation_frame(tweaked_rot);
        // frame3 translate = yocto::translation_frame(
        // vec3{.x = dx, .y = dy, .z = 0.0});
        // return rotate * translate;
        frame.o.x = dx;
        frame.o.y = dy;
        return frame;
      };

    std::function<double(const std::array<double, D> &)> Loss =
      [&poly, &OuterFrame, &InnerFrame](
          const std::array<double, D> &args) {
        solve_attempts++;
        // All snub footballs include the origin.
        return FullLossContainsOrigin(
            poly, OuterFrame(args), InnerFrame(args));
      };

    constexpr double Q = 0.20;

    const std::array<double, D> lb =
      {-Q, -Q, -Q, -Q,
       -Q, -Q, -Q, -Q, -1.0, -1.0};
    const std::array<double, D> ub =
      {+Q, +Q, +Q, +Q,
       +Q, +Q, +Q, +Q, +1.0, +1.0};

    const int seed = RandTo(rc, 0x7FFFFFFE);
    const auto &[args, error] =
      Opt::Minimize<D>(Loss, lb, ub, 1000, 2, 1, seed);

    if (error <= 0.0) {

      // XXX would be best to improve the solution here, especially
      // since the previous optimization had limits on the
      // angle

      if (iter > MIN_VERBOSE_ITERS) {
        status->Printf("%s " AYELLOW("solved") " after "
                       AWHITE("%d") " iters (%s).\n",
                       ConfigString(theta, phi, stretch).c_str(),
                       iter,
                       ANSI::Time(solve_timer.Seconds()).c_str());
      }
      return std::make_tuple(OuterFrame(args),
                             InnerFrame(args),
                             iter,
                             -error);
    }
  }

  return std::nullopt;
}

// Snub cube that has been stretched along the given axis.
// theta (inclination): [0, pi]
// phi (azimuth): [0, 2pi]
// stretch: [1, ∞]
static Polyhedron Football(double theta, double phi, double stretch) {
  CHECK(stretch >= 1.0);

  Polyhedron poly = SnubCube();
  // Stretch
  vec3 axis_dir = normalize(vec3{
    sin(theta) * yocto::cos(phi),
    sin(theta) * yocto::sin(phi),
    cos(theta)
    });

  for (vec3 &v : poly.vertices) {
    // Project point onto axis.
    double proj = dot(v, axis_dir);
    vec3 parallel_change = (stretch - 1.0) * proj * axis_dir;
    v += parallel_change;
  }

  return poly;
}

static BigPoly BigFootball(double theta, double phi, double stretch,
                           int digits) {

  BigPoly poly = BigScube(digits);
  // Stretch
  BigVec3 axis_dir = BigVec3{
    BigRat::FromDouble(sin(theta) * yocto::cos(phi)),
    BigRat::FromDouble(sin(theta) * yocto::sin(phi)),
    BigRat::FromDouble(cos(theta)),
  };

  BigVec3 stretch_axis =
    (BigRat::FromDouble(stretch) - BigRat(1)) * axis_dir;

  for (BigVec3 &v : poly.vertices) {
    // Project point onto axis.
    BigRat proj = dot(v, axis_dir);
    BigVec3 parallel_change = stretch_axis * proj;
    v = v + parallel_change;
  }

  return poly;
}

// num_points is the number on each side.
static std::optional<std::tuple<frame3, frame3, double>>
ComputeMinimumClearance(
    int thread_idx,
    ArcFour *rc,
    double theta, double phi, double stretch) {
  Polyhedron poly = Football(theta, phi, stretch);

  const auto result = DoSolve(thread_idx,
                              theta, phi, stretch,
                              rc, poly);
  delete poly.faces;

  if (!result.has_value()) {
    noperts++;
    return std::nullopt;

  } else {
    solved++;
    const auto &[outer, inner, iters, clearance] = result.value();
    return {std::make_tuple(outer, inner, clearance)};
  }
}

Tree3D<double, bool> GetDoneTree(NDSolutions<3> &sols) {
  ArcFour rc("getdone");
  std::vector<std::tuple<double, double, double>> done_xyz;
  for (int64_t i = 0; i < sols.Size(); i++) {
    const auto &[pos, clearance, outer, inner] = sols[i];
    done_xyz.emplace_back(pos[0], pos[1], pos[2]);
  }

  // Insert in a random order, because Tree3D doesn't
  // rebalance yet.
  Shuffle(&rc, &done_xyz);
  Tree3D<double, bool> done;
  for (const auto &[x, y, z] : done_xyz) {
    done.Insert(x, y, z, true);
  }

  return done;
}

static void DoFootball() {
  solve_attempts.Reset();
  solved.Reset();
  footballs.Reset();

  NDSolutions<3> sols("football.nds");
  if (sols.Size() > 0) {
    status->Printf("Continuing from " AWHITE("%lld") " sols.",
                   sols.Size());
  }

  // Keep hard, noperts

  // Given a number of points n, we generate a prism with regular
  // n-gons as faces (dihedral symmetry). There is one
  // parameter, which is the depth of the prism. When it is very
  // shallow, we have a "manhole cover" (which goes through one way)
  // and when it is very long we have a "churro" (which goes through
  // another way) and we want to see where the crossover point is.

  // Each trial is independent. We're mostly interested in the minimum
  // clearance at each depth. For a given depth we find solutions and
  // then minimize their clearance.
  std::mutex m;

  Timer run_timer;
  Periodically status_per(1.0);
  Periodically save_per(600.0, false);

  const int64_t THETA_SAMPLES = 100;
  const int64_t PHI_SAMPLES = 100;
  const int64_t STRETCH_SAMPLES =
    140;
    // 2000;
  const int64_t TOTAL_SAMPLES =
    THETA_SAMPLES * PHI_SAMPLES * STRETCH_SAMPLES;

  constexpr double MIN_THETA = 0.0;
  constexpr double MAX_THETA = std::numbers::pi / 2.0;
  constexpr double MIN_PHI = 0.0;
  constexpr double MAX_PHI = std::numbers::pi / 2.0;
  constexpr double MIN_STRETCH = 1.0;
  // constexpr double MAX_STRETCH = 1.05;
  constexpr double MAX_STRETCH = 1.00035;

  auto MaybeStatus = [&](int64_t num_left,
                         double theta, double phi, double stretch) {
      if (status_per.ShouldRun()) {
        MutexLock ml(&m);
        double total_time = run_timer.Seconds();
        const int64_t p = footballs.Read();
        double pps = p / total_time;

        ANSI::ProgressBarOptions options;
        options.include_frac = false;
        options.include_percent = true;

        std::string timing = std::format(
            AWHITE("{:.4f}") " footballs/s  "
            AGREY("|") "  " AWHITE("{}") " remain",
            pps,
            num_left);

        const double save_in = save_per.SecondsLeft();

        std::string msg =
          std::format(
              APURPLE("{}") AWHITE("s") " "
              AGREEN("{}") AWHITE("✔") " "
              AORANGE("{}") AWHITE("⚡") " "
              ARED("{}") AWHITE("⛔") " "
              "(save in {})",
              FormatNum(solve_attempts.Read()),
              FormatNum(solved.Read()),
              FormatNum(hard.Read()),
              noperts.Read(),
              ANSI::Time(save_in));

        std::string bar =
          ANSI::ProgressBar(TOTAL_SAMPLES - num_left, TOTAL_SAMPLES,
                            msg, total_time, options);

        std::vector<std::string> status_lines;
        status_lines.reserve(NUM_THREADS + ADDL_STATUS_LINES);
        status_lines.push_back(
            ANSI_GREY
            "——————————————————————————————————————————————————————————"
            ANSI_RESET);
        for (int i = 0; i < NUM_THREADS; i++) {
          status_lines.push_back(thread_status[i]);
        }
        status_lines.push_back(std::move(timing));
        status_lines.push_back(std::move(bar));
        status->EmitStatus(status_lines);
      }
    };

  auto MaybeSave = [&]() {
      save_per.RunIf([&]() {
          sols.Save();
          status->Printf("Saved " AWHITE("%lld") "\n",
                         sols.Size());
        });
    };

  auto Decode = [](int64_t work_idx) {
      int64_t ti = work_idx % THETA_SAMPLES;
      work_idx /= THETA_SAMPLES;

      int64_t pi = work_idx % PHI_SAMPLES;
      work_idx /= PHI_SAMPLES;

      CHECK(work_idx < STRETCH_SAMPLES);
      int64_t si = work_idx;

      double tf = ti / (double)THETA_SAMPLES;
      double theta = MIN_THETA + (tf * (MAX_THETA - MIN_THETA));

      double pf = pi / (double)PHI_SAMPLES;
      double phi = MIN_PHI + (pf * (MAX_PHI - MIN_PHI));

      // stretch more at the beginning.
      double sf = (1.0 - si / (double)STRETCH_SAMPLES);
      double stretch = MIN_STRETCH + (sf * (MAX_STRETCH - MIN_STRETCH));

      return std::make_tuple(theta, phi, stretch);
    };

  static constexpr double UNSOLVED = -1.0;

  Tree3D<double, bool> done_tree = GetDoneTree(sols);

  constexpr double CLOSE = 1.0e-6;

  // Collect all the work indices in memory. When running fresh, we
  // can just do these in sequence, but the process takes a long time
  // and it is nice to see samples from all throughout. Everything
  // has to fit in memory, anyway.
  std::vector<int64_t> work;
  for (int64_t work_idx = 0; work_idx < TOTAL_SAMPLES; work_idx++) {
    const auto &[theta, phi, stretch] = Decode(work_idx);
    if (!done_tree.Empty()) {
      const auto &[pos_, data_, dist] =
        done_tree.Closest(theta, phi, stretch);
      // Essentially looking for exact hits here.
      if (dist < CLOSE) {
        // Done, then.
        continue;
      }
    }
    work.push_back(work_idx);
  }

  // Get a picture of the full space by running these in random
  // order. It's just numbers; there are no memory locality downsides.
  {
    ArcFour rc("shuffle");
    Shuffle(&rc, &work);
  }

  ParallelFan(
      NUM_THREADS + 1,
      [&](int thread_idx) {
        if (thread_idx == NUM_THREADS) {
          for(;;) {
            using namespace std::chrono_literals;
            std::this_thread::sleep_for(1s);

            int64_t num_left = 0;
            {
              MutexLock ml(&m);
              if (work.empty())
                return;
              num_left = work.size();
            }
            MaybeStatus(num_left, 0, 0, 0);
          }

          return;
        }

        ArcFour rc(std::format("football.{}.{}", thread_idx,
                               time(nullptr)));
        for (;;) {
          int64_t work_idx = 0;
          int64_t num_left = 0;
          {
            MutexLock ml(&m);
            if (work.empty())
              return;
            work_idx = work.back();
            work.pop_back();
            num_left = work.size();
          }

          const auto &[theta, phi, stretch] = Decode(work_idx);

          const auto &oresult =
            ComputeMinimumClearance(thread_idx, &rc,
                                    theta, phi, stretch);
          footballs++;

          if (oresult.has_value()) {
            const auto &[outer, inner, clearance] = oresult.value();
            sols.Add({theta, phi, stretch}, clearance, outer, inner);
          } else {
            sols.Add({theta, phi, stretch}, UNSOLVED, frame3(), frame3());
          }

          MaybeStatus(num_left, theta, phi, stretch);
          MaybeSave();
        }
      });

  status->Printf("Done in %s.\n",
                 ANSI::Time(run_timer.Seconds()).c_str());
  sols.Save();
}

static void Plot() {
  NDSolutions<3> sols("football.nds");

  printf("Number of rows: %lld\n", sols.Size());

  #if 0
  for (int axis = 0; axis < 3; axis++) {
    sols.Plot1D(axis, 1920 * 2, 1080 * 2,
                std::format("football-axis-{}.png", axis));
  }
  #endif

  sols.Plot1DColor2(2, 0, 1, 1920 * 2, 1080 * 2,
                    std::format("football-axis-{}.png", 2));

  sols.Plot2D(2, 0, 1920 * 2, 1080 * 2,
              "football-2D-20.png");
  sols.Plot2D(2, 1, 1920 * 2, 1080 * 2,
              "football-2D-21.png");
}

static void STL() {
  Polyhedron poly = Football(0.0, 0.0, 1.0);
  SaveAsSTL(poly, "football001.stl");

  Polyhedron poly2 = Football(0.1, 0.1, 1.5);
  SaveAsSTL(poly2, "football115.stl");

  Polyhedron poly3 = Football(std::numbers::pi / 4.0,
                              std::numbers::pi / 4.0,
                              1.5);
  SaveAsSTL(poly3, "football445.stl");
}

static void GetSol() {
  StatusBar status(1);
  NDSolutions<3> sols("football.nds");
  size_t size = sols.Size();
  double best_stretch = 1000000.0;
  CHECK(sols.Size() > 0);
  int64_t best_idx = 0;
  int64_t valid = 0, invalid = 0;
  Timer findbest_timer;
  Periodically findbest_per(1.0);
  std::mutex m;
  auto vec = sols.GetVec();
  ParallelAppi(
      vec,
      [&](int64_t idx, const auto &sol) {
        double best = ReadWithLock(&m, &best_stretch);
        const auto &[key, score, outer, inner] = sol;

        if (score >= 0.0 && key[2] < best) {
          // Must be valid.
          const auto &[theta, phi, stretch] = key;
          BigPoly football = BigFootball(theta, phi, stretch, 100);
          bool is_valid = ValidateSolution(football, outer, inner, 100);

          {
            MutexLock ml(&m);
            if (is_valid) {
              best_stretch = key[2];
              best_idx = idx;
              valid++;
            } else {
              invalid++;
            }

            findbest_per.RunIf([&]() {
                status.Progressf(idx, size, "Finding best. "
                                  ARED("%lld") "+" AGREEN("%lld"),
                                  invalid, valid);
              });
          }
        }
      }, 8);

  status.Statusf("Done.\n");

  CHECK(valid > 0);
  status.Printf("Took %s. Saw " ARED("%lld") " invalid solutions\n"
                 "(and " AWHITE("%lld") " valid improving solutions)\n"
                 "on the way.\n",
                 ANSI::Time(findbest_timer.Seconds()).c_str(),
                 invalid, valid);

  const auto &[key, score, outer, inner] = sols[best_idx];

  const auto &[theta, phi, stretch] = key;
  BigPoly football = BigFootball(theta, phi, stretch, 100);
  CHECK(ValidateSolution(football, outer, inner, 100));
  status.Printf("Solution " AGREEN("OK") "!\n");

  status.Printf("Best stretch %.17g at %lld\n"
                 "Config: %s\n"
                 "Clearance %.17g.\n",
                 best_stretch, best_idx,
                 ConfigString(theta, phi, stretch).c_str(),
                 score);

  Polyhedron poly = Football(key[0], key[1], key[2]);
  Rendering rendering(poly, 1920, 1080);
  rendering.RenderSolution(poly, outer, inner);
  rendering.Save("football-sol.png");

  {
    Polyhedron opoly = Rotate(poly, outer);
    Polyhedron ipoly = Rotate(poly, inner);
    Mesh2D sinner = Shadow(ipoly);
    std::vector<int> hull = QuickHull(sinner.vertices);

    std::vector<vec2> polygon;
    polygon.reserve(hull.size());
    for (int i : hull) polygon.push_back(sinner.vertices[i]);
    TriangularMesh3D residue = BigMakeHole(opoly, polygon);
    OrientMesh(&residue);

    std::string filename = "football-residue.stl";
    SaveAsSTL(residue, filename);
    status.Printf("Wrote %s", filename.c_str());
  }
}

#if 0
static void Extrapolate() {
  // Load solutions from the main database. Working from the existing
  // solution, reduce both the shrink while maintaining a solution
  // with maximal clearance. Save these solutions to a new file so
  // that we can plot them.

  NDSolutions<3> main_sols("football.nds");

  // Find some good solutions to extrapolate.

  std::vector<std::tuple<std::array<double, 3>, double, frame3, frame3>>
    out_sols;
  for (const auto &[key, clearance, outer, inner] : main_sols.GetVec()) {
    if (clearance > 0.0) {
      out_sols.emplace_back(key, clearance, outer, inner);
    }
  }

  std::sort(out_sols.begin(),
            out_sols.end(),
            [](const auto &a, const auto &b) {
              return std::get<1>(a) < std::get<1>(b);
            });

  // Consider the best 10%.
  out_sols.resize(out_sols.size() * 0.10);

  CHECK(out_sols.empty()) << "No sols to extrapolate!";

  Timer run_timer;
  Periodically status_per(1.0);
  Periodically save_per(600.0, false);
  std::mutex m;
  NDSolutions<3> sols("footstrapolated.nds");

  auto MaybeStatus = [&](int64_t num_left) {
      if (status_per.ShouldRun()) {
        MutexLock ml(&m);
        double total_time = run_timer.Seconds();
        const int64_t p = footballs.Read();
        double pps = p / total_time;

        ANSI::ProgressBarOptions options;
        options.include_frac = false;
        options.include_percent = true;

        std::string timing = std::format(
            AWHITE("{:.4f}") " footballs/s  "
            AGREY("|") "  " AWHITE("{}") " remain",
            pps,
            num_left);

        const double save_in = save_per.SecondsLeft();

        std::string msg =
          std::format(
              APURPLE("{}") AWHITE("s") " "
              AGREEN("{}") AWHITE("✔") " "
              AORANGE("{}") AWHITE("⚡") " "
              ARED("{}") AWHITE("⛔") " "
              "(save in {})",
              FormatNum(solve_attempts.Read()),
              FormatNum(solved.Read()),
              FormatNum(hard.Read()),
              noperts.Read(),
              ANSI::Time(save_in));

        std::string bar =
          ANSI::ProgressBar(TOTAL_SAMPLES - num_left, TOTAL_SAMPLES,
                            msg, total_time, options);

        std::vector<std::string> status_lines;
        status_lines.reserve(NUM_THREADS + ADDL_STATUS_LINES);
        status_lines.push_back(
            ANSI_GREY
            "——————————————————————————————————————————————————————————"
            ANSI_RESET);
        for (int i = 0; i < NUM_THREADS; i++) {
          status_lines.push_back(thread_status[i]);
        }
        status_lines.push_back(std::move(timing));
        status_lines.push_back(std::move(bar));
        status->EmitStatus(status_lines);
      }
    };

  auto MaybeSave = [&]() {
      save_per.RunIf([&]() {
          sols.Save();
          status->Printf("Saved " AWHITE("%lld") "\n",
                         sols.Size());
        });
    };
  // Get a picture of the full space by running these in random
  // order. It's just numbers; there are no memory locality downsides.
  {
    ArcFour rc("shuffle");
    Shuffle(&rc, &work);
  }

  ArcFour rc(std::format("football.{}.{}", thread_idx,
                         time(nullptr)));

  ParallelFan(
      NUM_THREADS,
      [&](int thread_idx) {

        for (;;) {
          int64_t work_idx = 0;
          int64_t num_left = 0;
          {
            MutexLock ml(&m);
            if (work.empty())
              return;
            work_idx = work.back();
            work.pop_back();
            num_left = work.size();
          }

          const auto &[theta, phi, stretch] = Decode(work_idx);

          const auto &oresult =
            ComputeMinimumClearance(thread_idx, &rc,
                                    theta, phi, stretch);
          footballs++;

          if (oresult.has_value()) {
            const auto &[outer, inner, clearance] = oresult.value();
            sols.Add({theta, phi, stretch}, clearance, outer, inner);
          } else {
            sols.Add({theta, phi, stretch}, UNSOLVED, frame3(), frame3());
          }

          MaybeStatus(num_left, theta, phi, stretch);
          MaybeSave();
        }
      });

  status->Printf("Done in %s.\n",
                 ANSI::Time(run_timer.Seconds()).c_str());
  sols.Save();
}
#endif

int main(int argc, char **argv) {
  ANSI::Init();

  {
    MutexLock ml(&thread_status_m);
    for (int i = 0; i < NUM_THREADS; i++)
      thread_status[i] = "start";
  }
  status = new StatusBar(NUM_THREADS + ADDL_STATUS_LINES);

  // STL();

  Plot();
  // GetSol();

  // DoFootball();

  return 0;
}
