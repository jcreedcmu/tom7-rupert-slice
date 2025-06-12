
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <format>
#include <functional>
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
#include "nd-solutions.h"
#include "opt/opt.h"
#include "periodically.h"
#include "polyhedra.h"
#include "randutil.h"
#include "solutions.h"
#include "status-bar.h"
#include "threadutil.h"
#include "timer.h"
#include "util.h"
#include "yocto_matht.h"

DECLARE_COUNTERS(solve_attempts, improve_attempts, hard, noperts, prisms,
                 already);

// If true, look at antimanholecovers / antichurros instead.
static constexpr bool ANTIPRISM = true;

static StatusBar *status = nullptr;

using vec2 = yocto::vec<double, 2>;
using vec3 = yocto::vec<double, 3>;
using vec4 = yocto::vec<double, 4>;
using mat4 = yocto::mat<double, 4>;
using quat4 = yocto::quat<double, 4>;
using frame3 = yocto::frame<double, 3>;

// Return the number of iterations taken, or nullopt if we exceeded
// the limit. If solved and the arguments are non-null, sets the outer
// frame and inner frame to some solution.
// static constexpr int NOPERT_ITERS = 200000;
// We get too many noperts at n=34 with the above
static constexpr int NOPERT_ITERS = 1000000;
static constexpr int MIN_VERBOSE_ITERS = 5000;

static constexpr int NUM_THREADS = 16;

// One attempt. Wants exclusive access to rc.
static std::optional<std::pair<frame3, frame3>>
Minimize(ArcFour *rc, const Polyhedron &poly) {
  // four params for outer rotation, four params for inner
  // rotation, two for 2d translation of inner.
  static constexpr int D = 10;

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
      frame.o.x = dx;
      frame.o.y = dy;
      return frame;
    };

  std::function<double(const std::array<double, D> &)> Loss =
    [&poly, &OuterFrame, &InnerFrame](
        const std::array<double, D> &args) {
      solve_attempts++;
      return LossFunctionContainsOrigin(
          poly, OuterFrame(args), InnerFrame(args));
    };

  constexpr double Q = 0.15;

  const std::array<double, D> lb =
    {-Q, -Q, -Q, -Q,
     -Q, -Q, -Q, -Q, -1.0, -1.0};
  const std::array<double, D> ub =
    {+Q, +Q, +Q, +Q,
     +Q, +Q, +Q, +Q, +1.0, +1.0};

  const int seed = RandTo(rc, 0x7FFFFFFE);
  const auto &[args, error] =
    Opt::Minimize<D>(Loss, lb, ub, 1000, 2, 1, seed);

  if (error == 0.0) {
    return {std::make_pair(OuterFrame(args), InnerFrame(args))};
  } else {
    return std::nullopt;
  }
}

struct SolveResult {
  int64_t iters = 0;
  double clearance = 0.0;
  frame3 outer, inner;
};

// We take the hard mutex and solve in parallel once a single
// one has taken a lot of iterations.
static std::mutex hard_mutex;
static constexpr int HARD_ITERATIONS = 2000;

// Single-threaded.
static std::optional<SolveResult> DoSolve(
    int thread_idx,
    int max_iterations,
    // Just used for debug print.
    int num_points, double depth,
    ArcFour *rc, const Polyhedron &poly) {
  CHECK(!poly.faces->v.empty());

  for (int iter = 0; iter < max_iterations; iter++) {
    const std::optional<std::pair<frame3, frame3>> sol =
      Minimize(rc, poly);

    if (sol.has_value()) {
      const auto &[outer_frame, inner_frame] = sol.value();
      std::optional<double> oc = GetClearance(poly, outer_frame, inner_frame);
      if (!oc.has_value()) {
        status->Print(ARED("Yuck") ": Solution with bogus clearance");
        continue;
      }

      return {SolveResult{
          .iters = iter,
          .clearance = oc.value(),
          .outer = outer_frame,
          .inner = inner_frame,
        }};
    }
  }

  return std::nullopt;
}

static std::optional<SolveResult>
ParallelSolve(
    int num_threads,
    int start_iterations,
    int max_iterations,
    ArcFour *rc_all, const Polyhedron &poly) {
  CHECK(!poly.faces->v.empty());

  if (start_iterations >= max_iterations) return std::nullopt;
  Timer run_timer;

  std::mutex m;

  // When < 0, we have failed.
  int64_t iters_available = max_iterations - start_iterations;
  int64_t iters_done = start_iterations;
  // First solution to succeed sets this.
  std::optional<SolveResult> result;

  const int64_t BATCH_SIZE = 128;

  Periodically status_per(10.0);

  ParallelFan(
      num_threads,
      [&](int thread_idx) {
        m.lock();
        ArcFour rc(std::format("{}.{}", thread_idx, Rand64(rc_all)));
        // Maybe just assume this when parallel solving?
        bool reporting = false;
        m.unlock();

        for (;;) {
          {
            MutexLock ml(&m);
            if (result.has_value()) return;
            if (iters_available <= 0) return;
            iters_available -= BATCH_SIZE;
            reporting = reporting || iters_done > MIN_VERBOSE_ITERS;
          }

          // Do a batch.

          for (int i = 0; i < BATCH_SIZE; i++) {
            auto ro = Minimize(&rc, poly);

            {
              MutexLock ml(&m);
              // account for partial batch
              iters_done++;

              if (result.has_value()) return;
              if (ro.has_value()) {
                const auto &[outer_frame, inner_frame] = ro.value();
                std::optional<double> oc =
                  GetClearance(poly, outer_frame, inner_frame);
                if (!oc.has_value()) {
                  status->Print(ARED("Yuck")
                                ": Solution with bogus clearance");
                  continue;
                }

                result.emplace(SolveResult{
                    .iters = iters_done,
                    .clearance = oc.value(),
                    .outer = outer_frame,
                    .inner = inner_frame,
                  });
                return;
              }
            }
          }

          if (reporting) {
            status_per.RunIf([&]() {
                status->Print("Still "
                              AFGCOLOR(220, 220, 190, "not solved")
                              " after " AWHITE("{}") " iters... ({})\n",
                              iters_done, ANSI::Time(run_timer.Seconds()));
              });
          }
        }
      });

  return result;
}

// Improves a solution to increase clearance.
static std::tuple<frame3, frame3, double>
DoImprove(int thread_idx,
          // Just used for debug print.
          int num_points, double depth,
          ArcFour *rc, const Polyhedron &poly,
          const frame3 &outer_frame,
          const frame3 &inner_frame,
          double clearance,
          // e.g. 100 passes
          int max_improve_opts) {

  frame3 best_outer = outer_frame;
  frame3 best_inner = inner_frame;
  double best_clearance = clearance;

  for (int iter = 0; iter < max_improve_opts; iter++) {
    // four params for outer rotation, four params for inner
    // rotation, two for 2d translation of inner.
    static constexpr int D = 10;

    // We never translate along z.
    CHECK(best_outer.o.z == 0.0 &&
          best_inner.o.z == 0.0) << "After " << iter << "iters:\n"
                                    "Outer:\n"
                                 << FrameString(best_outer)
                                 << "\nand inner:\n"
                                 << FrameString(best_inner);

    const auto &[start_outer_rot, otrans] =
      UnpackFrame(best_outer);
    const auto &[start_inner_rot, itrans] =
      UnpackFrame(best_inner);

    CHECK(!AllZero(start_outer_rot) &&
          !AllZero(start_inner_rot)) << "Bad starting solution: "
                                     << poly.name;

    CHECK(AllZero(otrans)) << "Expected no outer translation.";
    CHECK(itrans.z == 0.0) << itrans.z;

    quat4 best_outer_rot = start_outer_rot;
    quat4 best_inner_rot = start_inner_rot;
    vec2 best_inner_trans{itrans.x, itrans.y};

    auto OuterFrame = [&best_outer_rot](
        const std::array<double, D> &args) {
        const auto &[o0, o1, o2, o3,
                     i0_, i1_, i2_, i3_, dx_, dy_] = args;
        quat4 tweaked_rot = normalize(quat4{
            .x = best_outer_rot.x + o0,
            .y = best_outer_rot.y + o1,
            .z = best_outer_rot.z + o2,
            .w = best_outer_rot.w + o3,
          });
        return yocto::rotation_frame(tweaked_rot);
      };

    auto InnerFrame = [&best_inner_rot, &best_inner_trans](
        const std::array<double, D> &args) {
        const auto &[o0_, o1_, o2_, o3_,
                     i0, i1, i2, i3, dx, dy] = args;
        quat4 tweaked_rot = normalize(quat4{
            .x = best_inner_rot.x + i0,
            .y = best_inner_rot.y + i1,
            .z = best_inner_rot.z + i2,
            .w = best_inner_rot.w + i3,
          });
        frame3 rotate = yocto::rotation_frame(tweaked_rot);
        frame3 translate = yocto::translation_frame(
            vec3{
              .x = best_inner_trans.x + dx,
              .y = best_inner_trans.y + dy,
              .z = 0.0
            });
        return translate * rotate;
      };

    std::function<double(const std::array<double, D> &)> Loss =
      [&poly, &OuterFrame, &InnerFrame](
          const std::array<double, D> &args) {
        improve_attempts++;
        return FullLossContainsOrigin(
            poly, OuterFrame(args), InnerFrame(args));
      };

    constexpr double Q = 0.15;

    const std::array<double, D> lb =
      {-Q, -Q, -Q, -Q,
       -Q, -Q, -Q, -Q, -1.0, -1.0};
    const std::array<double, D> ub =
      {+Q, +Q, +Q, +Q,
       +Q, +Q, +Q, +Q, +1.0, +1.0};

    const int seed = RandTo(rc, 0x7FFFFFFE);
    const auto &[args, error] =
      Opt::Minimize<D>(Loss, lb, ub, 1000, 2, 1, seed);

    if (error < best_clearance) {

      best_outer = OuterFrame(args);
      best_inner = InnerFrame(args);
      best_clearance = error;

    } else {
      break;
    }
  }

  return std::make_tuple(best_outer, best_inner, best_clearance);
}

// num_points is the number on each side.
std::optional<std::tuple<frame3, frame3, double>> ComputeMinimumClearance(
    int thread_idx,
    ArcFour *rc,
    int64_t num_points, double depth,
    int num_improve_opts) {
  Polyhedron poly =
    ANTIPRISM ? NAntiPrism(num_points, depth) : NPrism(num_points, depth);

  std::optional<SolveResult> solve_result =
    DoSolve(thread_idx, HARD_ITERATIONS,
            num_points, depth,
            rc, poly);

  if (!solve_result.has_value()) {
    hard++;
    MutexLock ml(&hard_mutex);
    status->Print(AYELLOW("HARD") " poly, {}-prism, depth={:.11g}\n",
                  num_points, depth);
    solve_result =
      ParallelSolve(NUM_THREADS,
                    HARD_ITERATIONS, NOPERT_ITERS,
                    rc, poly);
  }

  if (!solve_result.has_value()) {
    noperts++;
    status->Print(AGREEN("Nopert!!") " {}-prism, d={:.11g}.\n",
                  num_points, depth);
    if (noperts.Read() > 10) {
      status->Print("Too many noperts! Increase the threshold?");
    } else {
      SolutionDB db;
      db.AddNopert(poly, SolutionDB::NOPERT_METHOD_CHURRO);
    }

    delete poly.faces;
    return std::nullopt;

  } else {
    if (solve_result.value().iters > HARD_ITERATIONS) {
      status->Print("Solved after " AGREEN("{}") " iters. Improving...",
                    solve_result.value().iters);
    }

    const auto &[best_outer, best_inner, best_error] =
      DoImprove(thread_idx, num_points, depth, rc, poly,
                solve_result->outer, solve_result->inner,
                solve_result->clearance,
                num_improve_opts);

    delete poly.faces;

    if (best_error < 0.0) {
      return std::make_tuple(best_outer, best_inner, -best_error);
    } else {
      return std::nullopt;
    }
  }
}


static void DoChurro(int64_t num_points) {
  std::string filename =
    std::format("{}churro{}.png", ANTIPRISM ? "anti" : "", num_points);
  if (Util::ExistsFile(filename))
    return;

  solve_attempts.Reset();
  improve_attempts.Reset();
  already.Reset();
  prisms.Reset();

  NDSolutions<1> sols(std::format("{}churro{}.nds",
                                  ANTIPRISM ? "anti" : "",
                                  num_points));
  if (sols.Size() > 0) {
    status->Print("Continuing from " AWHITE("{}") " sols.",
                   sols.Size());
  }

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
  Periodically save_per(60.0, false);
  const int64_t total_prisms = 75000.0 / (num_points / 20.0);
  const int num_improve_opts =
    num_points > 50 ? (num_points > 100 ? 10 : 25) : 100;

  auto MaybeStatus = [&]() {
      if (status_per.ShouldRun()) {
        MutexLock ml(&m);
        double total_time = run_timer.Seconds();
        const int64_t p = prisms.Read();
        double pps = p / total_time;

        ANSI::ProgressBarOptions options;
        options.include_frac = false;
        options.include_percent = true;

        std::string timing = std::format("{:.4f} prisms/s", pps);

        std::string msg =
          std::format(
              AYELLOW("{}") AWHITE("⋮") "  |  "
              APURPLE("{}") AWHITE("s") " "
              ABLUE("{}") AWHITE("i") " "
              ARED("{}") AWHITE("⛔") " "
              AGREEN("{}") AWHITE("∼") " ",
              num_points,
              FormatNum(solve_attempts.Read()),
              FormatNum(improve_attempts.Read()),
              noperts.Read(),
              already.Read());


        std::string bar =
          ANSI::ProgressBar(p, total_prisms - already.Read(),
                            msg, total_time, options);

        status->Status(
            "{}\n"
            "{}\n",
            timing,
            bar);
      }

      save_per.RunIf([&]() {
          sols.Save();
          status->Print("Wrote " AWHITE("{}") " sols.\n",
                        sols.Size());
        });
    };

  int64_t next_work_idx = 0;

  static constexpr double MIN_DEPTH = 1.0e-6;
  const double MAX_DEPTH =
    ANTIPRISM ? (num_points > 44 ? 1.0 : 8.0) :
    (num_points >= 100 ? 2.2 : 8.0);
  const double DEPTH_SPAN = MAX_DEPTH - MIN_DEPTH;
  const double CLOSE = (DEPTH_SPAN / total_prisms) * 0.25;

  auto IdxToDepth = [DEPTH_SPAN, total_prisms](int64_t work_idx) {
      return MIN_DEPTH + ((work_idx * DEPTH_SPAN) / (double)total_prisms);
    };

  ParallelFan(
      NUM_THREADS,
      [&](int thread_idx) {
        ArcFour rc(std::format("adv.{}.{}.{}", thread_idx,
                               num_points, time(nullptr)));
        for (;;) {
          int64_t work_idx = 0;
          {
            MutexLock ml(&m);
            if (next_work_idx == total_prisms)
              return;
            work_idx = next_work_idx;
            next_work_idx++;
          }

          const double depth = IdxToDepth(work_idx);
          double dist = sols.Distance({depth});
          if (dist < CLOSE) {
            already++;
            continue;
          }

          const auto &oresult =
            ComputeMinimumClearance(thread_idx, &rc, num_points, depth,
                                    num_improve_opts);
          prisms++;

          MaybeStatus();

          if (oresult.has_value()) {
            const auto &[outer, inner, clearance] = oresult.value();
            sols.Add({depth}, clearance, outer, inner);
          } else {
            // Mark missing solutions.
            sols.Add({depth}, -1.0, frame3{}, frame3{});
          }
        }
      });

  status->Print("[" AWHITE("{}") "] Done in {}.\n", num_points,
                ANSI::Time(run_timer.Seconds()));

  sols.Save();

  sols.Plot1D(0, 3840, 2160, filename);
  status->Print("Wrote " AGREEN("{}") "\n", filename);
}

int main(int argc, char **argv) {
  ANSI::Init();

  status = new StatusBar(2);

  for (int n = 46; n < 100; n++) {
    DoChurro(n);
  }

  // DoChurro(51);
  for (int n = 100; n < 200; n += 10) {
    DoChurro(n);
    DoChurro(n + 1);
  }

  return 0;
}
