#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <numbers>
#include <optional>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "ansi.h"
#include "arcfour.h"
#include "atomic-util.h"
#include "auto-histo.h"
#include "base/stringprintf.h"
#include "hull.h"
#include "interval-cover-util.h"
#include "interval-cover.h"
#include "opt/opt.h"
#include "periodically.h"
#include "point-set.h"
#include "polyhedra.h"
#include "randutil.h"
#include "rendering.h"
#include "solutions.h"
#include "status-bar.h"
#include "threadutil.h"
#include "timer.h"
#include "util.h"
#include "yocto_matht.h"

// Try to find counterexamples.

#define ABLOOD(s) AFGCOLOR(148, 0, 0, s)

DECLARE_COUNTERS(polyhedra, attempts, degenerate, skipped, u1_, u2_, u3_, u4_);

static constexpr int VERBOSE = 0;
static constexpr bool SAVE_EVERY_IMAGE = false;
static constexpr int HISTO_LINES = 10;

static StatusBar *status = nullptr;

using vec2 = yocto::vec<double, 2>;
using vec3 = yocto::vec<double, 3>;
using vec4 = yocto::vec<double, 4>;
using mat4 = yocto::mat<double, 4>;
using quat4 = yocto::quat<double, 4>;
using frame3 = yocto::frame<double, 3>;

using Solution = SolutionDB::Solution;
using Attempt = SolutionDB::Attempt;

static double InnerDistanceLoss(
    const Polyhedron &poly,
    const frame3 &outer_frame, const frame3 &inner_frame) {
  Mesh2D souter = Shadow(Rotate(poly, outer_frame));
  Mesh2D sinner = Shadow(Rotate(poly, inner_frame));

  // Although computing the convex hull is expensive, the tests
  // below are O(n*m), so it is helpful to significantly reduce
  // one of the factors.
  const std::vector<int> outer_hull = GrahamScan(souter.vertices);
  HullCircle circle(souter.vertices, outer_hull);

  // Does every vertex in inner fall inside the outer shadow?
  double error = 0.0;
  int errors = 0;
  for (const vec2 &iv : sinner.vertices) {
    if (circle.DefinitelyInside(iv) || InHull(souter, outer_hull, iv)) {
      // Further from the hull is better, when on the inside.
      error -= DistanceToHull(souter.vertices, outer_hull, iv);
    } else {
      error += 1000000.0;
      errors++;
    }
  }

  if (error <= 0.0 && errors > 0) [[unlikely]] {
    // If they are not in the mesh, don't return an actual zero.
    return std::numeric_limits<double>::min() * errors;
  } else {
    return error;
  }
}

#define ANSI_ORANGE ANSI_FG(247, 155, 57)

static constexpr int MAX_ITERS = 1000;
static std::optional<std::pair<frame3, frame3>> TryImprove(
    int thread_idx,
    ArcFour *rc,
    const Polyhedron &poly,
    const frame3 &original_outer_frame,
    const frame3 &original_inner_frame) {
  CHECK(!poly.faces->v.empty());

  const auto &[original_outer_rot, otrans] =
    UnpackFrame(original_outer_frame);
  const auto &[original_inner_rot, itrans] =
    UnpackFrame(original_inner_frame);

  const double start_loss = InnerDistanceLoss(poly,
                                              original_outer_frame,
                                              original_inner_frame);
  status->Printf("Starting loss: %s%.17g" ANSI_RESET "\n",
                 start_loss < 0.999999 ? ANSI_GREEN :
                 start_loss < 0.0 ? ANSI_YELLOW :
                 start_loss == 0.0 ? ANSI_ORANGE : ANSI_RED,
                 start_loss);

  // The outer translation should be 0, but we can just
  // make that be the case. The z component should also
  // be zero, but can be ignored.
  const vec2 original_trans = [&]() {
      vec3 o = itrans - otrans;
      return vec2(o.x, o.y);
    }();

  for (int iter = 0; iter < MAX_ITERS; iter++) {
    // four params for outer rotation, four params for inner
    // rotation, two for 2d translation of inner.
    static constexpr int D = 10;

    // Get the frames from the appropriate positions in the
    // argument.

    auto OuterFrame = [&original_outer_rot](
        const std::array<double, D> &args) {
        const auto &[o0, o1, o2, o3,
                     i0_, i1_, i2_, i3_, dx_, dy_] = args;
        quat4 tweaked_rot = normalize(quat4{
            .x = original_outer_rot.x + o0,
            .y = original_outer_rot.y + o1,
            .z = original_outer_rot.z + o2,
            .w = original_outer_rot.w + o3,
          });
        return yocto::rotation_frame(tweaked_rot);
      };

    auto InnerFrame = [&original_inner_rot](
        const std::array<double, D> &args) {
        const auto &[o0_, o1_, o2_, o3_,
                     i0, i1, i2, i3, dx, dy] = args;
        quat4 tweaked_rot = normalize(quat4{
            .x = original_inner_rot.x + i0,
            .y = original_inner_rot.y + i1,
            .z = original_inner_rot.z + i2,
            .w = original_inner_rot.w + i3,
          });
        frame3 rotate = yocto::rotation_frame(tweaked_rot);
        frame3 translate = yocto::translation_frame(
            vec3{.x = dx, .y = dy, .z = 0.0});
        return rotate * translate;
      };

    std::function<double(const std::array<double, D> &)> Loss =
      [&poly, &OuterFrame, &InnerFrame](
          const std::array<double, D> &args) {
        attempts++;

        frame3 outer_frame = OuterFrame(args);
        frame3 inner_frame = InnerFrame(args);
        return InnerDistanceLoss(poly, outer_frame, inner_frame);
      };

    constexpr double Q = 0.15;

    const std::array<double, D> lb =
      {-Q, -Q, -Q, -Q,
       -Q, -Q, -Q, -Q, -1.0, -0.25};
    const std::array<double, D> ub =
      {+Q, +Q, +Q, +Q,
       +Q, +Q, +Q, +Q, +1.0, +0.25};

    const int seed = RandTo(rc, 0x7FFFFFFE);
    const auto &[args, error] =
      Opt::Minimize<D>(Loss, lb, ub, 2000, 2, 100, seed);

    if (error < start_loss) {
      status->Printf(AGREEN("Success!") " "
                     ABLUE("%.17g") " → " ACYAN("%.17g") "\n",
                     start_loss, error);

      frame3 outer_frame = OuterFrame(args);
      frame3 inner_frame = InnerFrame(args);

      return {std::make_pair(outer_frame, inner_frame)};
    }
  }

  return std::nullopt;
}

static void Impert() {
  polyhedra.Reset();
  attempts.Reset();
  degenerate.Reset();

  static constexpr int NUM_THREADS = 8;

  Timer timer;
  AutoHisto histo(10000);
  Periodically status_per(5.0);
  std::mutex m;
  bool should_die = false;
  double total_gen_sec = 0.0;
  double total_solve_sec = 0.0;
  bool success = false;

  SolutionDB db;
  std::vector<Solution> all_solutions;
  std::unordered_map<int, int> improvement_attempts;
  Periodically refresh_solutions_per(60.0);

  auto GetWork = [&m, &db, &all_solutions, &refresh_solutions_per]() ->
    std::optional<Solution> {
    MutexLock ml(&m);
    if (refresh_solutions_per.ShouldRun()) {
      all_solutions = db.GetAllSolutions();
      std::vector<Attempt> attempts = db.GetAllAttempts();
      for (const Attempt &att : attempts) {

      }
    }



  };

  ParallelFan(
      NUM_THREADS,
      [&](int thread_idx) {
        ArcFour rc(StringPrintf("noperts.%d.%lld\n",
                                thread_idx, time(nullptr)));

        for (;;) {
          {
            MutexLock ml(&m);
            if (should_die) return;
          }

          std::vector<Solution> sols = AllSolutions();

          // For every solution, see if we also have an improved
          // version.

          if constexpr (SAVE_EVERY_IMAGE) {
            printf("Rendering %d points %d faces...\n",
                   (int)poly.vertices.size(),
                   (int)poly.faces->v.size());
            Rendering r(poly, 1920, 1080);
            // r.RenderMesh(Shadow(poly));
            r.RenderPerspectiveWireframe(poly, 0x99AA99AA);
            static int count = 0;
            r.Save(StringPrintf("poly.%d.png", count++));
          }

          Timer solve_timer;
          std::optional<int> iters = TrySolve(thread_idx, &rc, poly);
          const double solve_sec = solve_timer.Seconds();

          {
            MutexLock ml(&m);
            if (should_die) return;

            total_gen_sec += gen_sec;
            total_solve_sec += solve_sec;

            if (iters.has_value()) {
              histo.Observe(iters.value());
            } else {

              SolutionDB db;
              db.AddNopert(poly, candidates->Method());

              status->Printf(
                  "\n\n" ABGCOLOR(200, 0, 200,
                                  ANSI_DARK_GREY "***** "
                                  ANSI_YELLOW "NOPERT"
                                  ANSI_DARK_GREY " with "
                                  ANSI_WHITE "%d"
                                  ANSI_DARK_GREY " vertices *****")
                  "\n", (int)poly.vertices.size());

              for (const vec3 &v : poly.vertices) {
                status->Printf("  %s\n",
                               VecString(v).c_str());
              }

              should_die = true;
              success = true;
              return;
            }
          }

          if (status_per.ShouldRun()) {
            MutexLock ml(&m);
            double total_time = timer.Seconds();
            const int64_t polys = polyhedra.Read();
            double tps = polys / total_time;

            std::string info = candidates->Info();
            const auto &[numer, denom] = candidates->Frac();

            ANSI::ProgressBarOptions options;
            options.include_frac = false;
            options.include_percent = true;

            std::string timing =
              StringPrintf(
                  "Timing: [" AWHITE("%.1f") "/s] %s " APURPLE("gen")
                  " %s " ABLUE("sol"),
                  tps,
                  ANSI::Time(total_gen_sec).c_str(),
                  ANSI::Time(total_solve_sec).c_str());

            std::string msg =
              StringPrintf(
                  ACYAN("%s") " %s%s" AGREY("|") " "
                  ARED("%s") ABLOOD("×") " "
                  ABLUE("%s") AWHITE("∎"),
                  name.c_str(),
                  info.c_str(), info.empty() ? "" : " ",
                  FormatNum(degenerate.Read()).c_str(),
                  FormatNum(polys).c_str());

            std::string bar =
              ANSI::ProgressBar(numer, denom,
                                msg, total_time, options);

            status->Statusf(
                "%s\n"
                "%s\n"
                "%s\n",
                histo.SimpleANSI(HISTO_LINES).c_str(),
                timing.c_str(),
                bar.c_str());
          }
        }
      });

  return success;
}

// Prep
static void Run(uint64_t parameter) {

  // static constexpr int METHOD = SolutionDB::NOPERT_METHOD_RANDOM;
  // static constexpr int METHOD = SolutionDB::NOPERT_METHOD_SYMMETRIC;
  static constexpr int64_t MAX_SECONDS = 60 * 60;

  static constexpr int METHOD = SolutionDB::NOPERT_METHOD_REDUCE_SC;

  switch (METHOD) {
  case SolutionDB::NOPERT_METHOD_RANDOM:
  case SolutionDB::NOPERT_METHOD_CYCLIC:
  case SolutionDB::NOPERT_METHOD_SYMMETRIC: {

    CHECK(parameter >= 4) << "Must have at least four vertices.";
    for (;;) {

      std::unique_ptr<CandidateMaker> candidates(
          new RandomCandidateMaker(
              parameter, METHOD, MAX_SECONDS));

      Nopert(candidates.get());
      parameter++;
      std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    break;
  }

  case SolutionDB::NOPERT_METHOD_REDUCE_SC: {
    constexpr int num_vertices = 24;
    // We do not need to try all bitmasks due to symmetry. Alas, the
    // vertices are just in some arbitrary order, so it's not easy to
    // figure out which masks are equivalent. However, for vertex-
    // transitive shapes, we can say without loss of generality that
    // one of the vertices (the msb) is always deleted. So we only
    // need n - 1 bits.
    constexpr int num_bits = num_vertices - 1;
    constexpr uint64_t reduce_end = 1 << num_bits;
    const char *reference_name = "snubcube";

    // XXX load from file!
    IntervalSet done(false);

    const std::string filename =
      ReduceCandidateMaker::Filename(METHOD, reference_name);
    std::string old = Util::ReadFile(filename);
    if (!old.empty()) {
      done = IntervalCoverUtil::ParseBool(old);
      status->Printf("Loaded " AYELLOW("%s") "\n", filename.c_str());
    }

    // "done" because there's nothing to do.
    done.SplitRight(reduce_end, true);
    // Done on Startropics, 30 Dec 2024
    done.SetSpan(0, 3989000, true);

    uint64_t reduce_start = parameter;
    while (reduce_start < reduce_end) {
      IntervalSet::Span sp = done.GetPoint(reduce_start);
      if (sp.data == false)
        break;
      reduce_start = sp.end;
    }

    std::unique_ptr<CandidateMaker> candidates(
        new ReduceCandidateMaker(
            METHOD, &done,
            reduce_start, reduce_end,
            reference_name));
    Nopert(candidates.get());

    break;
  }
  default:
    LOG(FATAL) << "Bad nopert method";
  }
}

int main(int argc, char **argv) {
  ANSI::Init();
  printf("\n");

  status = new StatusBar(HISTO_LINES + 3);

  int parameter = 6;

  if (argc == 2) {
    parameter = strtol(argv[1], nullptr, 10);
  }

  Run(parameter);

  printf("OK\n");
  return 0;
}
