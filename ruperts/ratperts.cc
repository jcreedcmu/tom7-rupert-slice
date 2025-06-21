
// Pure rational solver.
//
// Start from the identity? Or a close solution?
//
// Use block box optimizer to tweak the orientations and translation.
// We can derive a rotation matrix without any square roots: An
// exact representation of the points. This allows us to do
// exact point-line side tests for the outer hull.
//
// Note that this does not save solutions in the database (since they
// cannot necessarily be represented as doubles); it writes them to a
// text file.

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <format>
#include <mutex>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "ansi.h"
#include "arcfour.h"
#include "atomic-util.h"
#include "auto-histo.h"
#include "big-polyhedra.h"
#include "bignum/big-overloads.h"
#include "bignum/big.h"
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
    // TODO: Could also try the one with the best ratio, or really
    // any "solution."
    SolutionDB::Solution sol =
      db.GetBestSolutionFor(polyhedron.name, false);

    const auto &[douter_rot, dotrans] =
      UnpackFrame(sol.outer_frame);
    const auto &[dinner_rot, ditrans] =
      UnpackFrame(sol.inner_frame);

    CHECK(dotrans.x == 0.0 &&
          dotrans.y == 0.0 &&
          dotrans.z == 0.0) << "This can be handled, but we expect "
      "exact zero translation for the outer frame: "
                            << VecString(dotrans);

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
    frame3 inner_frame =
      yocto::rotation_frame(normalize(SmallQuat(inner_rot)));
    vec2 trans = SmallVec(translation);
    inner_frame.o.x = trans.x;
    inner_frame.o.y = trans.y;

    Polyhedron outer = Rotate(poly, outer_frame);
    Polyhedron inner = Rotate(poly, inner_frame);
    Mesh2D souter = Shadow(outer);
    Mesh2D sinner = Shadow(inner);

    Rendering rendering(poly, 3840, 2160);
    rendering.RenderTriangulation(souter, 0xAA0000FF);
    rendering.RenderTriangulation(sinner, 0x00FF00AA);
    rendering.Save(filename);
    status->Print("Wrote " AGREEN("{}") "\n", filename.c_str());
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

    status->Print("Solved! {} iters, {} attempts, in {}\n", iters.Read(),
                  attempts.Read(), ANSI::Time(run_timer.Seconds()));

    std::string suffix = std::format("{}-{}-{}",
                                     LowerMethod(),
                                     polyhedron.name,
                                     time(nullptr));

    WriteImage(std::format("solved-{}.png", suffix),
               outer_rot, inner_rot, translation);

    std::string contents =
      std::format("outer_rot:\n{}\n"
                  "inner_rot:\n{}\n"
                  "translation:\n{}\n",
                  PlainQuatString(outer_rot),
                  PlainQuatString(inner_rot),
                  PlainVecString(translation));

    std::string sfile = std::format("solution-{}.txt", suffix);
    Util::WriteFile(sfile, contents);
    status->Print("Wrote " AGREEN("{}") "\n", sfile);

    // SaveSolution(polyhedron, outer_frame, inner_frame, METHOD);
  }

  void Run() {
    attempts.Reset();
    iters.Reset();

    status->Print("Running on " APURPLE("{}") ".\n",
                  polyhedron.name);

    ParallelFan(
      NUM_OUTER_THREADS,
      [&](int thread_idx) {
        ArcFour rc(std::format("ratperts.{}.{}", thread_idx, time(nullptr)));

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
              status->Print(
                  "[" AWHITE("rational") " " APURPLE("{}")
                  "] Time limit exceeded after {}\n",
                  polyhedron.name,
                  ANSI::Time(run_timer.Seconds()));
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
                      std::format(APURPLE("{}") " | " ACYAN("{}"),
                                  polyhedron.name, LowerMethod()),
                      total_time);

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
    BigQuat outer_rot;
    BigQuat inner_rot;
    BigVec2 translation = BigVec2(zero, zero);
    std::string what;

    switch (RandTo(rc, 4)) {
    case 0: {
      what = "ref";
      outer_rot = initial_outer_rot;
      inner_rot = initial_inner_rot;
      translation = initial_translation;
      break;
    }
    case 1: {
      what = "id";
      BigQuat outer_rot = BigQuat(zero, zero, zero, BigRat(1));
      BigQuat inner_rot = BigQuat(zero, zero, zero, BigRat(1));
      break;
    }
    case 2:
    case 3: {
      // Same orientation, but not actually the identity.
      what = "eq";
      quat4 douter_rot = RandomQuaternion(rc);
      outer_rot.x = BigRat::FromDouble(douter_rot.x);
      outer_rot.y = BigRat::FromDouble(douter_rot.y);
      outer_rot.z = BigRat::FromDouble(douter_rot.z);
      outer_rot.w = BigRat::FromDouble(douter_rot.w);
      inner_rot = outer_rot;
      break;
    }
    default:
      LOG(FATAL) << "Impossible";
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

    // We optimize in the space of doubles, but we scale the
    // double down to add precision. Each parameter gets a
    // different scale (it might be better to use a non-linear
    // mapping?)
    std::vector<BigRat> scale_down;
    for (int i = 0; i < 10; i++) {
      const int SCALE = 64 + RandTo(rc, 1024 * 1024 * 1024);
      scale_down.emplace_back(1, SCALE);
    }

    const BigInt LOSS_SCALE = BigInt::Pow(BigInt(10), 20);

    int local_best_errors = 9999999;

    static constexpr int D = 10;
    auto MakeConfig = [&](const std::array<double, D> &args) {
        BigQuat orot = outer_rot;
        BigQuat irot = inner_rot;
        BigVec2 itrans = translation;

        const auto &[ox, oy, oz, ow,
                     ix, iy, iz, iw,
                     tx, ty] = args;
        orot.x += BigRat::FromDouble(ox) * scale_down[0];
        orot.y += BigRat::FromDouble(oy) * scale_down[1];
        orot.z += BigRat::FromDouble(oz) * scale_down[2];
        orot.w += BigRat::FromDouble(ow) * scale_down[3];

        irot.x += BigRat::FromDouble(ix) * scale_down[4];
        irot.y += BigRat::FromDouble(iy) * scale_down[5];
        irot.z += BigRat::FromDouble(iz) * scale_down[6];
        irot.w += BigRat::FromDouble(iw) * scale_down[7];

        itrans.x += BigRat::FromDouble(tx) * scale_down[8];
        itrans.y += BigRat::FromDouble(ty) * scale_down[9];

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
          double d = loss.ToDouble();
          status->LineStatusf(
              thread_idx,
              AFGCOLOR(200, 200, 140, "%s") " " AWHITE("%s")
              " %d" ACYAN("×")
              ", err #%d (#%d" ABLUE("↓") "), "
              "%.8g"
              // " in %s\n"
              ,
              histo.UnlabeledHoriz(32).c_str(),
              what.c_str(),
              calls,
              errors, local_best_errors,
              d
              // , ANSI::Time(timer.Seconds()).c_str()
                              );
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
        else return
               (double)errors +
               std::abs((LOSS_SCALE * loss).ToDouble());
      };

    const std::array<double, D> lb =
      {-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0};
    const std::array<double, D> ub =
      {+1.0, +1.0, +1.0, +1.0, +1.0, +1.0, +1.0, +1.0, +1.0, +1.0};

    Timer opt_timer;
    const auto &[args, error] =
      Opt::Minimize<D>(Loss, lb, ub, 1000, 2, 1, rc->Byte());
    [[maybe_unused]] const double opt_sec = opt_timer.Seconds();

    status->Print(AYELLOW("{}") " | Done in {}. Best errs #{}\n",
                  histo.UnlabeledHoriz(32),
                  ANSI::Time(opt_timer.Seconds()),
                  local_best_errors);

    const auto &[orot, irot, itrans] = MakeConfig(args);
    return std::make_tuple(error, orot, irot, itrans);
  }
};

static void Ratpert() {
  ArcFour rc(std::format("ratperts.{}", time(nullptr)));

  // This is indeed solved easily.
  // BigPoly target(BigCube(100));

  BigPoly ridode(BigRidode(100));
  BigPoly dhexe(BigDhexe(100));
  BigPoly phexe(BigPhexe(100));
  BigPoly scube(BigScube(100));
  BigPoly sdode(BigSdode(100));

  StatusBar status(NUM_OUTER_THREADS + 2);

  if (true) {
    BigPoly tetra(BigTetra(100));
    for (const BigVec3 &v : tetra.vertices) {
      printf("  %s\n", VecString(v).c_str());
    }
    BigSolver solver(tetra, &status, {60.0 * 60.0});
    solver.Run();
    return;
  }

  if (false) {
    // Solved
    BigPoly triac(BigTriac(100));
    printf("Made triac.\n");
    for (const BigVec3 &v : triac.vertices) {
      printf("  %s\n", VecString(v).c_str());
    }
    BigSolver solver(triac, &status, {60.0 * 60.0});
    solver.Run();
    return;
  }


  // Do them round-robin, but starting at a random one.
  for (int p = RandTo(&rc, 5); true; p = (p + 1) % 5) {

    switch (p) {
    case 0: {
      BigSolver solver(dhexe, &status, {60.0 * 60.0});
      solver.Run();
      break;
    }

    case 1: {
      BigSolver solver(phexe, &status, {60.0 * 60.0});
      solver.Run();
      break;
    }

    case 2: {
      BigSolver solver(scube, &status, {60.0 * 60.0});
      solver.Run();
      break;
    }

    case 3: {
      BigSolver solver(ridode, &status, {60.0 * 60.0});
      solver.Run();
      break;
    }

    case 4: {
      BigSolver solver(sdode, &status, {60.0 * 60.0});
      solver.Run();
      break;
    }

    default:;
    }
  }
}


int main(int argc, char **argv) {
  ANSI::Init();

  Ratpert();

  return 0;
}
