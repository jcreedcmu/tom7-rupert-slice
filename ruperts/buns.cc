
// Solve two different shapes.
// Aside from the quadratic blow-up, I think this is
// non-canonical because how do you decide what scale to use?
// I originally used this to check a chiral pair (here
// the scale is obvious) like the snub cube and buns cube.
// But then I realized that adding reflections does not
// increase the solution space at all: A flip around the
// projection plane results in exactly the same orthographic
// projection.

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <format>
#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include "ansi.h"
#include "arcfour.h"
#include "atomic-util.h"
#include "auto-histo.h"
#include "base/logging.h"
#include "base/stringprintf.h"
#include "image.h"
#include "opt/opt.h"
#include "periodically.h"
#include "status-bar.h"
#include "threadutil.h"
#include "timer.h"
#include "util.h"

#include "polyhedra.h"
#include "rendering.h"
#include "solutions.h"
#include "yocto_matht.h"

DECLARE_COUNTERS(iters, attempts, u1_, u2_, u3_, u4_, u5_, u6_);

static constexpr bool WRITE_BEST_IMAGES = true;

using vec2 = yocto::vec<double, 2>;
using vec3 = yocto::vec<double, 3>;
using vec4 = yocto::vec<double, 4>;
using mat4 = yocto::mat<double, 4>;
using quat4 = yocto::quat<double, 4>;
using frame3 = yocto::frame<double, 3>;

static void SaveSolution(const Polyhedron &outer_poly,
                         const Polyhedron &inner_poly,
                         const frame3 &outer_frame,
                         const frame3 &inner_frame,
                         int method) {

  {
    Polyhedron outer = Rotate(outer_poly, outer_frame);
    Polyhedron inner = Rotate(inner_poly, inner_frame);
    Mesh2D souter = Shadow(outer);
    Mesh2D sinner = Shadow(inner);
    std::vector<int> outer_hull = QuickHull(souter.vertices);
    std::vector<int> inner_hull = QuickHull(sinner.vertices);

    Rendering rendering(outer_poly, 3840, 2160);
    rendering.RenderHull(souter, outer_hull, 0xAA0000FF);
    rendering.RenderHull(sinner, inner_hull, 0x00FF00AA);
    rendering.Save(std::format("hulls-{}-{}.png",
                               outer_poly.name, inner_poly.name));
  }

  // Would need to generalize these.
  #if 0
  std::optional<double> new_ratio =
    GetRatio(poly, outer_frame, inner_frame);
  std::optional<double> new_clearance =
    GetClearance(poly, outer_frame, inner_frame);

  if (!new_ratio.has_value() || !new_clearance.has_value()) {
    printf(ARED("SOLUTION IS INVALID!?") "\n");
    return;
  }

  const double ratio = new_ratio.value();
  const double clearance = new_clearance.value();
  #endif

  // Database only allows self solutions.
  /*
  SolutionDB db;
  db.AddSolution(poly.name, outer_frame, inner_frame,
                 method, SOURCE, ratio, clearance);

  printf("Added solution (" AYELLOW("%s") ") to database with "
         "ratio " APURPLE("%.17g") ", clearance " ABLUE("%.17g") "\n",
         poly.name.c_str(), ratio, clearance);
  */
}

static constexpr int NUM_THREADS = 12;
static constexpr int HISTO_LINES = 32;
static constexpr int STATUS_LINES = HISTO_LINES + 3;

struct HeteroSolver {
  const int METHOD = 0;
  const Polyhedron outer_poly, inner_poly;
  StatusBar *status = nullptr;
  const std::optional<double> time_limit;

  std::mutex m;
  bool should_die = false;
  Timer run_timer;
  Periodically status_per;
  Periodically image_per;
  double best_error = 1.0e42;
  AutoHisto error_histo;

  double prep_time = 0.0, opt_time = 0.0;

  HeteroSolver(int method,
         const Polyhedron &outer_poly, const Polyhedron &inner_poly,
         StatusBar *status,
         std::optional<double> time_limit = std::nullopt) :
    outer_poly(outer_poly), inner_poly(inner_poly),
    status(status), time_limit(time_limit),
    status_per(1.0), image_per(1.0), error_histo(100000) {

  }

  std::string LowerMethod() {
    return std::format("method_", METHOD);
  }

  void WriteImage(const std::string &filename,
                  const frame3 &outer_frame,
                  const frame3 &inner_frame) {
    Rendering rendering(outer_poly, 3840, 2160);

    Mesh2D souter = Shadow(Rotate(outer_poly, outer_frame));
    Mesh2D sinner = Shadow(Rotate(inner_poly, inner_frame));

    rendering.RenderMesh(souter);
    rendering.DarkenBG();

    rendering.RenderMesh(sinner);
    std::vector<int> hull = QuickHull(sinner.vertices);
    rendering.RenderHull(sinner, hull, 0x000000AA);
    rendering.RenderBadPoints(sinner, souter);
    rendering.img.Save(filename);

    status->Print("Wrote " AGREEN("{}") "\n", filename.c_str());
  }

  void Solved(const frame3 &outer_frame, const frame3 &inner_frame) {
    MutexLock ml(&m);
    // For easy ones, many threads will solve it at once, and then
    // write over each other's solutions.
    if (should_die && iters.Read() < 1000)
      return;
    should_die = true;

    status->Print("Solved! {} iters, {} attempts, in {}\n",
                  iters.Read(),
                  attempts.Read(), ANSI::Time(run_timer.Seconds()));

    WriteImage(std::format("solved-{}-{}.png",
                           outer_poly.name, inner_poly.name),
               outer_frame, inner_frame);

    std::string contents =
      std::format("outer:\n{}\n"
                  "inner:\n{}\n",
                  FrameString(outer_frame),
                  FrameString(inner_frame));

    AppendFormat(&contents,
                 "\n{}\n",
                 error_histo.SimpleAsciiString(50));

    std::string sfile = std::format("solution-{}-{}.txt",
                                    outer_poly.name,
                                    inner_poly.name);

    Util::WriteFile(sfile, contents);
    status->Print("Wrote " AGREEN("{}") "\n", sfile.c_str());

    SaveSolution(outer_poly, inner_poly, outer_frame, inner_frame, METHOD);
  }

  void Run() {
    attempts.Reset();
    iters.Reset();

    ParallelFan(
      NUM_THREADS,
      [&](int thread_idx) {
        ArcFour rc(std::format("solve.{}.{}", thread_idx, time(nullptr)));

        for (;;) {
          {
            MutexLock ml(&m);
            if (should_die) return;
            if (time_limit.has_value() &&
                run_timer.Seconds() > time_limit.value()) {
              should_die = true;
              /*
              SolutionDB db;
              db.AddAttempt(polyhedron.name, METHOD, SOURCE,
                            best_error, iters.Read(),
                            attempts.Read());
              */
              iters.Reset();
              attempts.Reset();
              status->Print(
                  "[" AWHITE("{}") "] Time limit exceeded after {}\n",
                  SolutionDB::MethodName(METHOD),
                  ANSI::Time(run_timer.Seconds()));
              return;
            }
          }

          const auto &[error, outer_frame, inner_frame] = RunOne(&rc);

          if (error == 0) {
            Solved(outer_frame, inner_frame);
            return;
          }

          {
            MutexLock ml(&m);
            error_histo.Observe(log(error));
            if (error < best_error) {
              best_error = error;
              if (WRITE_BEST_IMAGES &&
                  iters.Read() > 4096 &&
                  image_per.ShouldRun()) {
                // PERF: Maybe only write this at the end when
                // there is a time limit?
                std::string file_base =
                  std::format("best-{}-{}.{}",
                              outer_poly.name, inner_poly.name, iters.Read());
                WriteImage(file_base + ".png", outer_frame, inner_frame);
              }
            }

            status_per.RunIf([&]() {
                double total_time = run_timer.Seconds();
                int64_t it = iters.Read();
                double ips = it / total_time;

                int64_t end_sec =
                  time_limit.has_value() ? (int64_t)time_limit.value() :
                  9999999;
                std::string bar =
                  ANSI::ProgressBar(
                      (int64_t)total_time, end_sec,
                      std::format(APURPLE("{}") " | " ACYAN("{}"),
                                  outer_poly.name, inner_poly.name),
                      total_time);

                // TODO: Can use progress bar when there's a timer.
                status->Status(
                    "{}\n"
                    "{}\n"
                    "{} iters, {} attempts; best: {:.11g}"
                    " [" ACYAN("{:.3f}") "/s]\n",
                    error_histo.SimpleANSI(HISTO_LINES),
                    bar,
                    FormatNum(it),
                    FormatNum(attempts.Read()),
                    best_error, ips);
              });
          }

          iters++;
        }
      });

  }

  // Run one iteration, and return the error (and outer, inner
  // frames). Error of 0.0 means a solution. Exclusive access to rc.
  virtual std::tuple<double, frame3, frame3> RunOne(ArcFour *rc) = 0;
};

// Try simultaneously optimizing both the shadow and hole. This is
// much slower because we can't frontload precomputation (e.g. of a
// convex hull). But it could be that the perpendicular axis needs to
// be just right in order for it to be solvable; Solve() spends most
// of its time trying different shapes of the hole and only random
// samples for the shadow.
struct HeteroSimulSolver : public HeteroSolver {
  using HeteroSolver::HeteroSolver;
  HeteroSimulSolver(
      const Polyhedron &outer_poly_in, const Polyhedron &inner_poly_in,
      StatusBar *status_in,
      std::optional<double> time_limit_in = std::nullopt) :
    HeteroSolver(0, outer_poly_in, inner_poly_in, status_in, time_limit_in) {

  }

  std::tuple<double, frame3, frame3> RunOne(ArcFour *rc) override {
    // four params for outer rotation, four params for inner
    // rotation, two for 2d translation of inner.
    static constexpr int D = 10;

    Timer prep_timer;
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
        frame3 rotate = yocto::rotation_frame(tweaked_rot);
        frame3 translate = yocto::translation_frame(
            vec3{.x = dx, .y = dy, .z = 0.0});
        return translate * rotate;
      };

    std::function<double(const std::array<double, D> &)> Loss =
      [this, &OuterFrame, &InnerFrame](
          const std::array<double, D> &args) {
        attempts++;
        return HeteroLossFunctionContainsOrigin(
            outer_poly, inner_poly, OuterFrame(args), InnerFrame(args));
      };

    constexpr double Q = 0.15;

    const std::array<double, D> lb =
      {-Q, -Q, -Q, -Q,
       -Q, -Q, -Q, -Q, -0.25, -0.25};
    const std::array<double, D> ub =
      {+Q, +Q, +Q, +Q,
       +Q, +Q, +Q, +Q, +0.25, +0.25};
    [[maybe_unused]] const double prep_sec = prep_timer.Seconds();

    Timer opt_timer;
    const auto &[args, error] =
      Opt::Minimize<D>(Loss, lb, ub, 1000, 2);
    [[maybe_unused]] const double opt_sec = opt_timer.Seconds();

    return std::make_tuple(error, OuterFrame(args), InnerFrame(args));
  }
};

static void SolveSimul(const Polyhedron &outer,
                       const Polyhedron &inner,
                       StatusBar *status,
                       std::optional<double> time_limit = std::nullopt) {
  HeteroSimulSolver s(outer, inner, status, time_limit);
  s.Run();
}

int main(int argc, char **argv) {
  ANSI::Init();
  printf("\n");

  StatusBar status(STATUS_LINES);

  Polyhedron otarget = SnubCube();
  // Polyhedron otarget = SnubDodecahedron();

  Polyhedron itarget = ReflectXY(otarget);

  SolveSimul(otarget, itarget, &status);

  printf("OK\n");
  return 0;
}
