
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
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
#include "bounds.h"
#include "image.h"
#include "opt/opt.h"
#include "periodically.h"
#include "polyhedra.h"
#include "randutil.h"
#include "solutions.h"
#include "status-bar.h"
#include "threadutil.h"
#include "timer.h"
#include "yocto_matht.h"

DECLARE_COUNTERS(solve_attempts, improve_attempts, hard, noperts, prisms);

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
static constexpr int NOPERT_ITERS = 200000;
static constexpr int MIN_VERBOSE_ITERS = 5000;
static constexpr bool SAVE_HARD = false;
static std::optional<int> DoSolve(int thread_idx,
                                  // Just used for debug print.
                                  int num_points, double depth,
                                  ArcFour *rc, const Polyhedron &poly,
                                  frame3 *outer_frame_out,
                                  frame3 *inner_frame_out) {
  CHECK(!poly.faces->v.empty());

  for (int iter = 0; iter < NOPERT_ITERS; iter++) {

    if (iter > 0 && (iter % 5000) == 0) {
      status->Printf("[" APURPLE("%d") "] %d-prism (depth %.11g) "
                     AFGCOLOR(190, 220, 190, "not solved")
                     " yet; " AWHITE("%d") " iters...\n",
                     thread_idx, num_points, depth, iter);

      if (iter == 20000) {
        hard++;
        if (SAVE_HARD) {
          std::string filename = std::format("hard.{}.{}.stl",
                                             time(nullptr), thread_idx);
          SaveAsSTL(poly, filename);
          status->Printf("[" APURPLE("%d") "] Hard %d-point polyhedron saved "
                         "to " AGREEN("%s") " (%d-prism, depth %.11g)\n",
                         thread_idx, (int)poly.vertices.size(),
                         filename.c_str(),
                         num_points, depth);
        }
      }
    }

    // TODO: We should probably explicitly check churro vs manhole.
    // This reduces the number of parameters a lot; we're also
    // interested in the crossover.

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
        return LossFunction(poly, OuterFrame(args), InnerFrame(args));
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
      if (outer_frame_out != nullptr) {
        *outer_frame_out = OuterFrame(args);
      }
      if (inner_frame_out != nullptr) {
        *inner_frame_out = InnerFrame(args);
      }

      if (iter > MIN_VERBOSE_ITERS) {
        status->Printf("%d-prism (d %.11g) " AYELLOW("solved") " after "
                       AWHITE("%d") " iters.\n",
                       num_points, depth, iter);
      }
      return {iter};
    }
  }

  return std::nullopt;
}

// XXX maybe in polyhedra.h
static double FullLoss(
    const Polyhedron &poly,
    const frame3 &outer_frame, const frame3 &inner_frame) {

  Polyhedron outer = Rotate(poly, outer_frame);
  Polyhedron inner = Rotate(poly, inner_frame);
  Mesh2D souter = Shadow(outer);
  Mesh2D sinner = Shadow(inner);

  if (AllZero(souter.vertices) ||
      AllZero(sinner.vertices)) {

    return 1.0e6;
  }

  std::vector<int> outer_hull = QuickHull(souter.vertices);
  std::vector<int> inner_hull = QuickHull(sinner.vertices);

  if (outer_hull.size() < 3) {
    return 1.0e6;
  }

  HullInscribedCircle circle(souter.vertices, outer_hull);

  // Does every vertex in inner fall inside the outer shadow?
  double error = 0.0;
  int errors = 0;
  for (const vec2 &iv : sinner.vertices) {
    if (circle.DefinitelyInside(iv))
      continue;

    if (!InHull(souter, outer_hull, iv)) {
      // slow :(
      error += DistanceToHull(souter.vertices, outer_hull, iv);
      errors++;
    }
  }

  if (errors > 0) {
    if (error == 0.0) {
      [[unlikely]]
      return std::numeric_limits<double>::min() * errors;
    }
    return error;
  } else {
    double clearance = HullClearance(souter.vertices, outer_hull,
                                     sinner.vertices, inner_hull);
    return std::min(-clearance, 0.0);
  }
}

// Improves a solution to increase clearance.
static std::tuple<frame3, frame3, double>
DoImprove(int thread_idx,
          // Just used for debug print.
          int num_points, double depth,
          ArcFour *rc, const Polyhedron &poly,
          const frame3 &outer_frame,
          const frame3 &inner_frame,
          // e.g. 100 passes
          int max_improve_opts) {

  frame3 best_outer = outer_frame;
  frame3 best_inner = inner_frame;
  std::optional<double> oc = GetClearance(poly, outer_frame, inner_frame);
  CHECK(oc.has_value()) << "Bogus solution with no clearance";
  double best_clearance = oc.value();

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
        return FullLoss(poly, OuterFrame(args), InnerFrame(args));
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


static Polyhedron NPrism(int64_t num_points, double depth) {
  CHECK(num_points >= 3);
  Polyhedron poly;
  // We put the top face at d/2, and the obverse -d/2.

  std::vector<int> top_face, bot_face;
  for (int i = 0; i < num_points; i++) {
    const double angle = i * (2.0 * std::numbers::pi / num_points);

    double x = cos(angle);
    double y = sin(angle);

    int t = poly.vertices.size();
    poly.vertices.push_back(vec3{x, y, depth * 0.5});
    int b = poly.vertices.size();
    poly.vertices.push_back(vec3{x, y, depth * -0.5});

    top_face.push_back(t);
    bot_face.push_back(b);
  }

  // Add the rectangular side faces.
  std::vector<std::vector<int>> fs = {top_face, bot_face};
  for (int i = 0; i < num_points; i++) {
    int side1 = i * 2;
    int side2 = ((i + 1) % num_points) * 2;

    fs.push_back(std::vector<int>{side1, side1 + 1, side2 + 1, side2});
  }

  poly.faces = new Faces(num_points * 2, std::move(fs));
  poly.name = "prism";

  return poly;
}

// num_points is the number on each side.
std::optional<double> ComputeMinimumClearance(
    int thread_idx,
    ArcFour *rc,
    int64_t num_points, double depth,
    int num_improve_opts) {
  Polyhedron poly = NPrism(num_points, depth);

  frame3 outer, inner;
  std::optional<int> iters = DoSolve(thread_idx,
                                     num_points, depth,
                                     rc, poly, &outer, &inner);
  if (!iters.has_value()) {
    SolutionDB db;
    status->Printf(AGREEN("Nopert!!") " %d-prism, d=%.11g.\n");
    db.AddNopert(poly, SolutionDB::NOPERT_METHOD_CHURRO);
    noperts++;
    prisms++;
    if (noperts.Read() > 10) {
      status->Printf("Too many noperts! Increase the threshold?");
      exit(-1);
    }
    return std::nullopt;

  } else {
    const auto &[best_outer, best_inner, best_error] =
      DoImprove(thread_idx, num_points, depth, rc, poly,
                outer, inner, num_improve_opts);
    prisms++;

    if (best_error < 0.0) {
      return {-best_error};
    } else {
      return std::nullopt;
    }
  }
}

static void DoChurro(int64_t num_points) {

  solve_attempts.Reset();
  improve_attempts.Reset();
  prisms.Reset();

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
          StringPrintf(
              AYELLOW("%lld") AWHITE("⋮") "  |  "
              APURPLE("%s") AWHITE("s") " "
              ABLUE("%s") AWHITE("i") " "
              ARED("%lld") AWHITE("⛔") " ",
              num_points,
              FormatNum(solve_attempts.Read()).c_str(),
              FormatNum(improve_attempts.Read()).c_str(),
              noperts.Read());


        std::string bar =
          ANSI::ProgressBar(p, total_prisms,
                            msg, total_time, options);

        status->Statusf(
            "%s\n"
            "%s\n",
            timing.c_str(),
            bar.c_str());
      }
    };

  int64_t next_work_idx = 0;

  static constexpr double MIN_DEPTH = 1.0e-6;
  const double MAX_DEPTH =
    num_points >= 100 ? 2.2 : 8.0;
  const double DEPTH_SPAN = MAX_DEPTH - MIN_DEPTH;

  auto IdxToDepth = [DEPTH_SPAN, total_prisms](int64_t work_idx) {
      return MIN_DEPTH + ((work_idx * DEPTH_SPAN) / (double)total_prisms);
    };

  static constexpr double NO_SOLUTION = -1.0;
  std::vector<double> results(total_prisms, NO_SOLUTION);

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

          std::optional<double> clearance =
            ComputeMinimumClearance(thread_idx, &rc, num_points, depth,
                                    num_improve_opts);

          MaybeStatus();

          if (clearance.has_value()) {
            MutexLock ml(&m);
            results[work_idx] = clearance.value();
          }
        }
      });

  status->Printf("[" AWHITE("%d") "] Done in %s.\n", num_points,
                 ANSI::Time(run_timer.Seconds()).c_str());

  Bounds bounds;
  // Make sure x axis is included.
  bounds.Bound(IdxToDepth(0), 0.0);
  for (int i = 0; i < total_prisms; i++) {
    double d = IdxToDepth(i);
    if (results[i] >= 0.0) {
      bounds.Bound(d, results[i]);
    }
  }
  bounds.AddTwoMarginsFrac(0.02, 0.0);

  constexpr int WIDTH = 3840;
  constexpr int HEIGHT = 2160;
  constexpr float PX = 2.0f;
  constexpr float CIRCLE = 3.0f * PX;
  constexpr float DOT = 2.0f * PX;
  ImageRGBA image(WIDTH, HEIGHT);
  image.Clear32(0x000000FF);

  Bounds::Scaler scaler = bounds.Stretch(WIDTH, HEIGHT).FlipY();;
  {
    const auto y = scaler.ScaleY(0);
    image.BlendLine32(0, std::round(y), WIDTH - 1, std::round(y),
                      0xFF0000AA);
  }

  for (double x = 0.0; x < std::round(MAX_DEPTH); x += 0.25) {
    double xx = scaler.ScaleX(x);
    image.BlendLine32(std::round(xx), 0, std::round(xx), HEIGHT - 1,
                      0x00770099);
  }

  for (int x = (int)std::round(MIN_DEPTH); x < std::round(MAX_DEPTH); x++) {
    double xx = scaler.ScaleX(x);
    image.BlendLine32(std::round(xx), 0, std::round(xx), HEIGHT - 1,
                      0x33FF33AA);
  }

  for (int i = 0; i < total_prisms; i++) {
    double d = IdxToDepth(i);
    if (results[i] < 0.0) {
      const auto &[x, y] = scaler.Scale(d, 0.0);
      image.BlendThickCircleAA32(
          std::round(x), std::round(y) + 2 * PX, CIRCLE, PX, 0xFF000099);
    } else {
      const auto &[x, y] = scaler.Scale(d, results[i]);
      image.BlendFilledCircleAA32(
          std::round(x), std::round(y), DOT, 0xFFFFFF99);
    }
    // image.BlendPixel32(std::round(x), std::round(y), 0xFFFFFF99);
  }

  image.Save(std::format("churro{}.png", num_points));
}

int main(int argc, char **argv) {
  ANSI::Init();

  status = new StatusBar(2);

  // DoChurro(51);
  for (int n = 100; n < 200; n += 10) {
    DoChurro(n);
    DoChurro(n + 1);
  }

  return 0;
}
