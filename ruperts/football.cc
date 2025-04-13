
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
#include <tuple>
#include <utility>
#include <vector>

#include "ansi.h"
#include "arcfour.h"
#include "atomic-util.h"
#include "base/stringprintf.h"
#include "big-csg.h"
#include "bounds.h"
#include "image.h"
#include "mesh.h"
#include "nd-solutions.h"
#include "opt/opt.h"
#include "periodically.h"
#include "polyhedra.h"
#include "randutil.h"
#include "rendering.h"
#include "solutions.h"
#include "status-bar.h"
#include "threadutil.h"
#include "timer.h"
#include "util.h"
#include "yocto_matht.h"

DECLARE_COUNTERS(solve_attempts, solved, hard, noperts, footballs);

static StatusBar *status = nullptr;

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
static constexpr int NOPERT_ITERS = 100000;
static constexpr int MIN_VERBOSE_ITERS = 10000;

// Returns best solution, iters, and clearance.
static std::optional<std::tuple<frame3, frame3, int64_t, double>>
DoSolve(int thread_idx,
        // Just used for debug print.
        double theta, double phi, double stretch,
        ArcFour *rc, const Polyhedron &poly) {
  CHECK(!poly.faces->v.empty());

  for (int iter = 0; iter < NOPERT_ITERS; iter++) {

    if (iter > 0 && (iter % 5000) == 0) {
      status->Printf("[" AYELLOW("%d") "] %s "
                     AFGCOLOR(190, 220, 190, "not solved")
                     " yet; " AWHITE("%d") " iters...\n",
                     thread_idx, ConfigString(theta, phi, stretch).c_str());

      if (iter == 20000) {
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
                       AWHITE("%d") " iters.\n",
                       ConfigString(theta, phi, stretch).c_str(),
                       iter);
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

  static constexpr int NUM_THREADS = 16;

  // Each trial is independent. We're mostly interested in the minimum
  // clearance at each depth. For a given depth we find solutions and
  // then minimize their clearance.
  std::mutex m;

  Timer run_timer;
  Periodically status_per(1.0);
  Periodically save_per(600.0, false);

  const int64_t THETA_SAMPLES = 100;
  const int64_t PHI_SAMPLES = 100;
  const int64_t STRETCH_SAMPLES = 2000;
  const int64_t TOTAL_SAMPLES =
    THETA_SAMPLES * PHI_SAMPLES * STRETCH_SAMPLES;

  constexpr double MIN_THETA = 0.0;
  constexpr double MAX_THETA = std::numbers::pi / 2.0;
  constexpr double MIN_PHI = 0.0;
  constexpr double MAX_PHI = std::numbers::pi / 2.0;
  constexpr double MIN_STRETCH = 1.0;
  constexpr double MAX_STRETCH = 1.05;

  auto MaybeStatus = [&](double theta, double phi, double stretch) {
      if (status_per.ShouldRun()) {
        MutexLock ml(&m);
        double total_time = run_timer.Seconds();
        const int64_t p = footballs.Read();
        double pps = p / total_time;

        ANSI::ProgressBarOptions options;
        options.include_frac = false;
        options.include_percent = true;

        std::string timing = std::format(
            "{:.4f} footballs/s  "
            AGREY("|") "  {}",
            pps,
            ConfigString(theta, phi, stretch));

        const double save_in = save_per.SecondsLeft();

        std::string msg =
          StringPrintf(
              APURPLE("%s") AWHITE("s") " "
              AGREEN("%s") AWHITE("✔") " "
              ARED("%lld") AWHITE("⛔") " "
              "(save in %s)",
              FormatNum(solve_attempts.Read()).c_str(),
              FormatNum(solved.Read()).c_str(),
              noperts.Read(),
              ANSI::Time(save_in).c_str());

        std::string bar =
          ANSI::ProgressBar(p, TOTAL_SAMPLES,
                            msg, total_time, options);

        status->Statusf(
            "%s\n"
            "%s\n",
            timing.c_str(),
            bar.c_str());
      }
    };

  auto MaybeSave = [&]() {
      save_per.RunIf([&]() {
          sols.Save();
          status->Printf("Saved " AWHITE("%lld") "\n",
                         sols.Size());
        });
    };

  #if 0
  for (int ti = 0; ti < THETA_SAMPLES; ti++) {
    double tf = ti / (double)THETA_SAMPLES;
    double theta = MIN_THETA + (tf * (MAX_THETA - MIN_THETA));

    for (int pi = 0; pi < PHI_SAMPLES; pi++) {
      double pf = pi / (double)PHI_SAMPLES;
      double phi = MIN_PHI + (pf * (MAX_PHI - MIN_PHI));

      for (int si = 0; si < STRETCH_SAMPLES; si++) {
        double sf = si / (double)STRETCH_SAMPLES;
        double stretch = MIN_STRETCH + (sf * (MAX_STRETCH - MIN_STRETCH));


      }
    }
  }
  #endif

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

  constexpr double CLOSE = 1.0e-6;
  // Binary search to find the next work index
  // that we have to do! The boundary is mostly
  // monotonic but might be a little ragged
  // because of multithreading. So we just try
  // to get close and then do a linear scan (which
  // could still miss samples if we get unlucky!).
  // We don't really care about duplicates or a
  // few missing samples, though; this is stochastic.
  int64_t next_work_idx = [&](){
    // all work below this is done
    int64_t lb = 0;
    // all work here and beyond is not done.
    int64_t ub = TOTAL_SAMPLES;
    for (;;) {
      status->Printf("[%lld, %lld]\n", lb, ub);
      CHECK(ub >= lb);
      if (ub - lb < 1000) {
        while (lb < ub) {
          const auto &[theta, phi, stretch] = Decode(lb);
          double dist = sols.Distance({theta, phi, stretch});
          if (dist > CLOSE)
            break;
          lb++;
        }
        return lb;
      }

      int64_t midpoint = (lb + ub) >> 1;

      const auto &[theta, phi, stretch] = Decode(midpoint);
      double dist = sols.Distance({theta, phi, stretch});

      if (dist < CLOSE) {
        lb = midpoint + 1;
      } else {
        ub = midpoint;
      }
    }
    }();

  status->Printf("Starting at idx %lld/%lld\n",
                 next_work_idx, TOTAL_SAMPLES);

  ParallelFan(
      NUM_THREADS,
      [&](int thread_idx) {
        ArcFour rc(std::format("football.{}.{}", thread_idx,
                               time(nullptr)));
        for (;;) {
          int64_t work_idx = 0;
          {
            MutexLock ml(&m);
            if (next_work_idx == TOTAL_SAMPLES)
              return;
            work_idx = next_work_idx;
            next_work_idx++;
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

          MaybeStatus(theta, phi, stretch);
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
  NDSolutions<3> sols("football.nds");
  size_t size = sols.Size();
  double best_stretch = 1000000.0;
  int64_t best_idx = 0;
  for (int64_t idx = 0; idx < size; idx++) {
    const auto &[key, score, outer, inner] = sols[idx];
    if (key[2] < best_stretch) {
      best_stretch = key[2];
      best_idx = idx;
    }
  }

  const auto &[key, score, outer, inner] = sols[best_idx];

  status->Printf("Best stretch %.17g at %lld\n"
                 "Config: %s\n"
                 "Clearance %.17g.\n",
                 best_stretch, best_idx,
                 ConfigString(key[0], key[1], key[2]).c_str(),
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
    status->Printf("Wrote %s", filename.c_str());
  }

}

int main(int argc, char **argv) {
  ANSI::Init();

  status = new StatusBar(2);

  // STL();

  GetSol();

  Plot();

  // DoFootball();

  return 0;
}
