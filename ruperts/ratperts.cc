
// Pure rational solver.
//
// Start from the identity? Or a close solution?
//
// Use block box optimizer to tweak the orientations and translation.
// We can derive a rotation matrix without any square roots: An
// exact representation of the points. This allows us to do
// point-line side tests for the outer hull.

#include "base/stringprintf.h"
#include "big-polyhedra.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <ctime>
#include <optional>
#include <cstdint>
#include <mutex>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "ansi.h"
#include "arcfour.h"
#include "atomic-util.h"
#include "bignum/big-overloads.h"
#include "bignum/big.h"
#include "auto-histo.h"
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

DECLARE_COUNTERS(iters, attempts, u1_, u2_, u3_, u4_, u5_, u6_);

static constexpr int NUM_OUTER_THREADS = 8;

struct BigSolver {

  const BigPoly polyhedron;
  StatusBar *status = nullptr;
  const std::optional<double> time_limit;

  BigQuat initial_outer_rot;
  BigQuat initial_inner_rot;
  BigVec2 initial_translation;

  std::mutex m;
  bool should_die = false;
  Timer run_timer;
  Periodically status_per;
  // BigRat best_error = BigRat::FromDecimal("100000000000000000000000000000");
  double best_error = 9999999999999.0;
  int best_errors = 99999999;

  double prep_time = 0.0, opt_time = 0.0;

  BigSolver(const BigPoly &polyhedron, StatusBar *status,
            std::optional<double> time_limit = std::nullopt) :
    polyhedron(polyhedron), status(status), time_limit(time_limit),
    status_per(1.0) {

    SolutionDB db;
    SolutionDB::Solution sol = db.GetSolution(455);

    const auto &[douter_rot, dotrans] =
      UnpackFrame(sol.outer_frame);
    const auto &[dinner_rot, ditrans] =
      UnpackFrame(sol.inner_frame);

    CHECK(dotrans.x == 0.0 &&
          dotrans.y == 0.0 &&
          dotrans.z == 0.0) << "This can be handled, but we expect "
      "exact zero translation for the outer frame.";

    // z component does not matter, because we project along z.
    initial_translation = BigVec2(BigRat::FromDouble(ditrans.x),
                                  BigRat::FromDouble(ditrans.y));

    initial_outer_rot = BigQuat(BigRat::FromDouble(douter_rot.x),
                                BigRat::FromDouble(douter_rot.y),
                                BigRat::FromDouble(douter_rot.z),
                                BigRat::FromDouble(douter_rot.w));

    initial_inner_rot = BigQuat(BigRat::FromDouble(dinner_rot.x),
                                BigRat::FromDouble(dinner_rot.y),
                                BigRat::FromDouble(dinner_rot.z),
                                BigRat::FromDouble(dinner_rot.w));
  }

  static std::string LowerMethod() {
    return "ratperts";
  }

  void WriteImage(const std::string &filename,
                  const BigQuat &outer_rot,
                  const BigQuat &inner_rot,
                  const BigVec2 &translation) {

    // Convert back and render.
    Polyhedron poly = SmallPoly(polyhedron);

    const frame3 outer_frame =
      yocto::rotation_frame(normalize(SmallQuat(outer_rot)));
    // Note: ignoring translation
    const frame3 inner_frame =
      yocto::rotation_frame(normalize(SmallQuat(inner_rot)));

    Polyhedron outer = Rotate(poly, outer_frame);
    Polyhedron inner = Rotate(poly, inner_frame);
    Mesh2D souter = Shadow(outer);
    Mesh2D sinner = Shadow(inner);

    Rendering rendering(poly, 3840, 2160);
    rendering.RenderTriangulation(souter, 0xAA0000FF);
    rendering.RenderTriangulation(sinner, 0x00FF00AA);
    rendering.Save(filename, poly.name);
    status->Printf("Wrote " AGREEN("%s") "\n", filename.c_str());
  }

  void Solved(const BigQuat &outer_rot,
              const BigQuat &inner_rot,
              const BigVec2 &translation) {
    MutexLock ml(&m);
    // For easy ones, many threads will solve it at once, and then
    // write over each other's solutions.
    if (should_die && iters.Read() < 1000)
      return;
    should_die = true;

    status->Printf("Solved! %lld iters, %lld attempts, in %s\n", iters.Read(),
                   attempts.Read(), ANSI::Time(run_timer.Seconds()).c_str());

    WriteImage(StringPrintf("solved-%s-%s.png", LowerMethod().c_str(),
                            polyhedron.name),
               outer_rot, inner_rot, translation);

    std::string contents =
      StringPrintf("outer_rot:\n%s\n"
                   "inner_rot:\n%s\n"
                   "translation:\n%s\n",
                   PlainQuatString(outer_rot).c_str(),
                   PlainQuatString(inner_rot).c_str(),
                   PlainVecString(translation).c_str());

    std::string sfile = StringPrintf("solution-%s-%s.txt",
                                     LowerMethod().c_str(),
                                     polyhedron.name);

    Util::WriteFile(sfile, contents);
    status->Printf("Wrote " AGREEN("%s") "\n", sfile.c_str());

    // SaveSolution(polyhedron, outer_frame, inner_frame, METHOD);
  }

  void Run() {
    attempts.Reset();
    iters.Reset();

    ParallelFan(
      NUM_OUTER_THREADS,
      [&](int thread_idx) {
        ArcFour rc(StringPrintf("ratperts.%d.%lld", thread_idx,
                                time(nullptr)));

        for (;;) {
          {
            MutexLock ml(&m);
            if (should_die) return;
            if (time_limit.has_value() &&
                run_timer.Seconds() > time_limit.value()) {
              should_die = true;
              SolutionDB db;
              db.AddAttempt(polyhedron.name,
                            SolutionDB::METHOD_RATIONAL,
                            0,
                            best_error, iters.Read(),
                            attempts.Read());
              iters.Reset();
              attempts.Reset();
              status->Printf(
                  "[" AWHITE("rational") "] Time limit exceeded after %s\n",
                  ANSI::Time(run_timer.Seconds()).c_str());
              return;
            }
          }

          const auto &[error, outer_rot, inner_rot, translation] =
            RunOne(&rc, thread_idx);

          if (error == 0.0) {
            Solved(outer_rot, inner_rot, translation);
            return;
          }

          {
            MutexLock ml(&m);
            if (error < best_error) {
              best_error = error;
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
                      StringPrintf(APURPLE("%s") " | " ACYAN("%s"),
                                   polyhedron.name, LowerMethod().c_str()),
                      total_time);

                // TODO: Can use progress bar when there's a timer.
                status->EmitLine(NUM_OUTER_THREADS, bar.c_str());
                status->LineStatusf(
                    NUM_OUTER_THREADS + 1,
                    "%s iters, %s attempts; best: #%d, %.7f"
                    " [" ACYAN("%.3f") "/s]\n",
                    FormatNum(it).c_str(),
                    FormatNum(attempts.Read()).c_str(),
                    best_errors,
                    best_error, ips);
              });
          }

          iters++;
        }
      });

  }

  // Run one iteration, and return the error. Error of 0.0 means
  // a solution.
  // Exclusive access to rc.
  std::tuple<double, BigQuat, BigQuat, BigVec2> RunOne(ArcFour *rc,
                                                       int thread_idx) {

    // Add tiny random jitter.
    auto Jitter = [&rc](BigRat *r) {
        bool neg = (rc->Byte() & 1);
        *r += BigRat(neg ? BigInt(-1) : BigInt(1),
                     BigInt(Rand64(rc)) + BigInt(1000));
      };

    AutoHisto histo(10000);
    Timer timer;
    Periodically status_per(60.0);

    // Unlike the double-based solvers, we do not keep a unit
    // quaternion here; we want to avoid Sqrt so that everything is
    // exact.
    BigRat zero(0);
    BigQuat outer_rot = BigQuat(zero, zero, zero, BigRat(1));
    BigQuat inner_rot = BigQuat(zero, zero, zero, BigRat(1));
    BigVec2 translation = BigVec2(zero, zero);
    std::string what = "id";

    if (rc->Byte() & 1) {
      what = "ref";
      outer_rot = initial_outer_rot;
      inner_rot = initial_inner_rot;
      translation = initial_translation;
    }

    Jitter(&outer_rot.x);
    Jitter(&outer_rot.y);
    Jitter(&outer_rot.z);
    Jitter(&outer_rot.w);

    Jitter(&inner_rot.x);
    Jitter(&inner_rot.y);
    Jitter(&inner_rot.z);
    Jitter(&inner_rot.w);

    Jitter(&translation.x);
    Jitter(&translation.y);

    // For more precision, we could have a "scale" parameter
    // which we use to scale down every argument?
    //
    // Or we can set this randomly.
    const int SCALE = 64 + RandTo(rc, 1024);
    const BigRat scale_down(1, SCALE);

    int local_best_errors = 9999999;

    static constexpr int D = 10;
    auto MakeConfig = [&](const std::array<double, D> &args) {
        BigQuat orot = outer_rot;
        BigQuat irot = inner_rot;
        BigVec2 itrans = translation;

        const auto &[ox, oy, oz, ow,
                     ix, iy, iz, iw,
                     tx, ty] = args;
        orot.x += BigRat::FromDouble(ox) * scale_down;
        orot.y += BigRat::FromDouble(oy) * scale_down;
        orot.z += BigRat::FromDouble(oz) * scale_down;
        orot.w += BigRat::FromDouble(ow) * scale_down;

        irot.x += BigRat::FromDouble(ix) * scale_down;
        irot.y += BigRat::FromDouble(iy) * scale_down;
        irot.z += BigRat::FromDouble(iz) * scale_down;
        irot.w += BigRat::FromDouble(iw) * scale_down;

        itrans.x += BigRat::FromDouble(tx) * scale_down;
        itrans.y += BigRat::FromDouble(tx) * scale_down;

        return std::make_tuple(std::move(orot),
                               std::move(irot),
                               std::move(itrans));
      };

    int calls = 0;
    auto Loss = [&](const std::array<double, D> &args) {
        Timer timer;
        const auto &[orot, irot, itrans] = MakeConfig(args);
        BigFrame big_outer_frame = NonUnitRotationFrame(orot);
        BigFrame big_inner_frame = NonUnitRotationFrame(irot);
        BigPoly outer = Rotate(big_outer_frame, polyhedron);
        BigPoly inner = Rotate(big_inner_frame, polyhedron);

        BigMesh2D souter = Shadow(outer);
        BigMesh2D sinner = Translate(itrans, Shadow(inner));

        std::vector<int> hull = BigQuickHull(souter.vertices);
        std::vector<int> inner_hull = BigQuickHull(sinner.vertices);

        BigRat loss(0);
        int errors = 0;
        for (int i : inner_hull) {
          const BigVec2 &v = sinner.vertices[i];
          if (InHull(souter.vertices, hull, v)) {
            // Include inner gradient, but scaled down
            loss -= SquaredDistanceToHull(souter.vertices, hull, v) /
              BigRat(512);
          } else {
            loss += SquaredDistanceToHull(souter.vertices, hull, v);
            errors++;
          }
        }

        calls++;
        attempts++;
        histo.Observe(errors);
        if (status_per.ShouldRun()) {
          status->LineStatusf(
              thread_idx,
              AFGCOLOR(200, 200, 140, "%s") " " AWHITE("%s")
              " %d" ACYAN("×")
              ", err #%d (#%d" ABLUE("↓") "), in %s\n",
              histo.UnlabeledHoriz(32).c_str(),
              what.c_str(),
              calls,
              errors, local_best_errors,
              ANSI::Time(timer.Seconds()).c_str());
        }
        if (errors == 0) {
          Solved(orot, irot, itrans);
          // Can just abort here...
        }

        if (errors < local_best_errors) {
          local_best_errors = errors;
          MutexLock ml(&m);
          best_errors = std::min(best_errors, errors);
        }

        if (errors == 0) return 0.0;
        else return (double)errors + std::abs(loss.ToDouble());
      };

    const std::array<double, D> lb =
      {-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0};
    const std::array<double, D> ub =
      {+1.0, +1.0, +1.0, +1.0, +1.0, +1.0, +1.0, +1.0, +1.0, +1.0};

    Timer opt_timer;
    const auto &[args, error] =
      Opt::Minimize<D>(Loss, lb, ub, 1000, 2, 1, rc->Byte());
    [[maybe_unused]] const double opt_sec = opt_timer.Seconds();

    status->Printf(AYELLOW("%s") " | Done in %s. Best errs #%d\n",
                   histo.UnlabeledHoriz(32).c_str(),
                   ANSI::Time(opt_timer.Seconds()).c_str(),
                   local_best_errors);


    const auto &[orot, irot, itrans] = MakeConfig(args);
    return std::make_tuple(error, orot, irot, itrans);
  }
};

static void Ratpert() {
  BigPoly ridode(BigRidode(100));
  StatusBar status(NUM_OUTER_THREADS + 2);

  for (;;) {
    BigSolver solver(ridode, &status, {60.0 * 60.0});
    solver.Run();
  }
}


int main(int argc, char **argv) {
  ANSI::Init();

  Ratpert();

  return 0;
}
