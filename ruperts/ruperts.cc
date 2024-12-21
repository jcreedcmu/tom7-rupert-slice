
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <functional>
#include <limits>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
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
#include "randutil.h"
#include "status-bar.h"
#include "threadutil.h"
#include "timer.h"
#include "util.h"

#include "polyhedra.h"
#include "rendering.h"
#include "solutions.h"
#include "yocto_matht.h"

DECLARE_COUNTERS(iters, attempts, u1_, u2_, u3_, u4_, u5_, u6_);

using vec2 = yocto::vec<double, 2>;
using vec3 = yocto::vec<double, 3>;
using vec4 = yocto::vec<double, 4>;
using mat4 = yocto::mat<double, 4>;
using quat4 = yocto::quat<double, 4>;
using frame3 = yocto::frame<double, 3>;

static void SaveSolution(const Polyhedron &poly,
                         const frame3 &outer_frame,
                         const frame3 &inner_frame,
                         int method) {
  SolutionDB db;

  // Compute error ratio.
  Polyhedron outer = Rotate(poly, outer_frame);
  Polyhedron inner = Rotate(poly, inner_frame);
  Mesh2D souter = Shadow(outer);
  Mesh2D sinner = Shadow(inner);

  std::vector<int> outer_hull = QuickHull(souter.vertices);
  std::vector<int> inner_hull = QuickHull(sinner.vertices);

  double outer_area = AreaOfHull(souter, outer_hull);
  double inner_area = AreaOfHull(sinner, inner_hull);

  double ratio = inner_area / outer_area;

  Rendering rendering(poly, 3840, 2160);
  rendering.RenderHull(souter, outer_hull, 0xAA0000FF);
  rendering.RenderHull(sinner, inner_hull, 0x00FF00AA);
  rendering.Save(StringPrintf("hulls-%s.png", poly.name));

  db.AddSolution(poly.name, outer_frame, inner_frame, method, ratio);
  printf("Added solution (" AYELLOW("%s") ") to database with "
         "ratio " APURPLE("%.17g") "\n",
         poly.name, ratio);
}

static constexpr int NUM_THREADS = 4;
static constexpr int HISTO_LINES = 32;

template<int METHOD>
struct Solver {

  const Polyhedron polyhedron;
  StatusBar *status = nullptr;
  const std::optional<double> time_limit;

  std::mutex m;
  bool should_die = false;
  Timer run_timer;
  Periodically status_per;
  Periodically image_per;
  double best_error = 1.0e42;
  AutoHisto error_histo;
  // ArcFour rc(&

  double prep_time = 0.0, opt_time = 0.0;

  Solver(const Polyhedron &polyhedron, StatusBar *status,
         std::optional<double> time_limit = std::nullopt) :
    polyhedron(polyhedron), status(status), time_limit(time_limit),
    status_per(1.0), image_per(1.0), error_histo(100000) {

  }

  static std::string LowerMethod() {
    std::string name = Util::lcase(SolutionDB::MethodName(METHOD));
    (void)Util::TryStripPrefix("method_", &name);
    return name;
  }

  void WriteImage(const std::string &filename,
                  const frame3 &outer_frame,
                  const frame3 &inner_frame) {
    Rendering rendering(polyhedron, 3840, 2160);

    Mesh2D souter = Shadow(Rotate(polyhedron, outer_frame));
    Mesh2D sinner = Shadow(Rotate(polyhedron, inner_frame));

    rendering.RenderMesh(souter);
    rendering.DarkenBG();

    rendering.RenderMesh(sinner);
    std::vector<int> hull = QuickHull(sinner.vertices);
    rendering.RenderHull(sinner, hull, 0x000000AA);
    rendering.RenderBadPoints(sinner, souter);
    rendering.img.Save(filename);

    status->Printf("Wrote " AGREEN("%s") "\n", filename.c_str());
  }

  void Solved(const frame3 &outer_frame, const frame3 &inner_frame) {
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
               outer_frame, inner_frame);

    std::string contents =
      StringPrintf("outer:\n%s\n"
                   "inner:\n%s\n",
                   FrameString(outer_frame).c_str(),
                   FrameString(inner_frame).c_str());

    StringAppendF(&contents,
                  "\n%s\n",
                  error_histo.SimpleAsciiString(50).c_str());

    std::string sfile = StringPrintf("solution-%s-%s.txt",
                                     LowerMethod().c_str(),
                                     polyhedron.name);

    Util::WriteFile(sfile, contents);
    status->Printf("Wrote " AGREEN("%s") "\n", sfile.c_str());

    SaveSolution(polyhedron, outer_frame, inner_frame, METHOD);
  }

  void Run() {
    attempts.Reset();
    iters.Reset();

    ParallelFan(
      NUM_THREADS,
      [&](int thread_idx) {
        ArcFour rc(StringPrintf("solve.%d.%lld", thread_idx,
                                time(nullptr)));

        for (;;) {
          {
            MutexLock ml(&m);
            if (should_die) return;
            if (time_limit.has_value() &&
                run_timer.Seconds() > time_limit.value()) {
              should_die = true;
              SolutionDB db;
              db.AddAttempt(polyhedron.name, METHOD,
                            best_error, iters.Read(),
                            attempts.Read());
              iters.Reset();
              attempts.Reset();
              status->Printf(
                  "[" AWHITE("%s") "] Time limit exceeded after %s\n",
                  SolutionDB::MethodName(METHOD),
                  ANSI::Time(run_timer.Seconds()).c_str());
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
              if (iters.Read() > 4096 &&
                  image_per.ShouldRun()) {
                // PERF: Maybe only write this at the end when
                // there is a time limit?
                std::string file_base =
                  StringPrintf("best-%s-%s.%lld",
                               LowerMethod().c_str(),
                               polyhedron.name, iters.Read());
                WriteImage(file_base + ".png", outer_frame, inner_frame);
              }
            }

            status_per.RunIf([&]() {
                double total_time = run_timer.Seconds();
                int64_t it = iters.Read();
                double ips = it / total_time;

                // TODO: Can use progress bar when there's a timer.
                status->Statusf(
                    "%s\n"
                    "[" AWHITE("%s") "]" " run for %s "
                    "[" ACYAN("%.3f") "/s]\n"
                    "%s iters, %s attempts; best: %.11g",
                    error_histo.SimpleANSI(HISTO_LINES).c_str(),
                    LowerMethod().c_str(),
                    ANSI::Time(total_time).c_str(),
                    ips,
                    FormatNum(it).c_str(),
                    FormatNum(attempts.Read()).c_str(),
                    best_error);
              });
          }

          iters++;
        }
      });

  }

  // Run one iteration, and return the error. Error of 0.0 means
  // a solution.
  // Exclusive access to rc.
  virtual std::tuple<double, frame3, frame3> RunOne(ArcFour *rc) = 0;
};


struct HullSolver : public Solver<SolutionDB::METHOD_HULL> {
  using Solver::Solver;

  std::tuple<double, frame3, frame3> RunOne(ArcFour *rc) override {
    Timer prep_timer;
    quat4 outer_rot = RandomQuaternion(rc);
    const frame3 outer_frame = yocto::rotation_frame(outer_rot);
    Polyhedron outer = Rotate(polyhedron, outer_frame);
    Mesh2D souter = Shadow(outer);

    const std::vector<int> shadow_hull = QuickHull(souter.vertices);

    // Starting orientation/position.
    const quat4 inner_rot = RandomQuaternion(rc);

    static constexpr int D = 6;
    auto InnerFrame = [&inner_rot](const std::array<double, D> &args) {
        const auto &[di, dj, dk, dl, dx, dy] = args;
        quat4 tweaked_rot = normalize(quat4{
            .x = inner_rot.x + di,
            .y = inner_rot.y + dj,
            .z = inner_rot.z + dk,
            .w = inner_rot.w + dl,
          });
        frame3 rotate = yocto::rotation_frame(tweaked_rot);
        frame3 translate = yocto::translation_frame(
            vec3{.x = dx, .y = dy, .z = 0.0});
        return rotate * translate;
      };

    std::function<double(const std::array<double, D> &)> Loss =
      [this, &souter, &shadow_hull, &InnerFrame](
          const std::array<double, D> &args) {
        attempts++;
        frame3 frame = InnerFrame(args);
        Polyhedron inner = Rotate(polyhedron, frame);
        Mesh2D sinner = Shadow(inner);

        // Does every vertex in inner fall inside the outer shadow?
        double error = 0.0;
        int errors = 0;
        for (const vec2 &iv : sinner.vertices) {
          if (!InHull(souter, shadow_hull, iv)) {
            error += DistanceToHull(souter.vertices, shadow_hull, iv);
            errors++;
          }
        }

        if (error == 0.0 && errors > 0) [[unlikely]] {
          // If they are not in the mesh, don't return an actual zero.
          return std::numeric_limits<double>::min() * errors;
        } else {
          return error;
        }
      };

    const std::array<double, D> lb =
      {-0.15, -0.15, -0.15, -0.15, -0.25, -0.25};
    const std::array<double, D> ub =
      {+0.15, +0.15, +0.15, +0.15, +0.25, +0.25};
    [[maybe_unused]] const double prep_sec = prep_timer.Seconds();

    Timer opt_timer;
    const auto &[args, error] =
      Opt::Minimize<D>(Loss, lb, ub, 1000, 2);
    [[maybe_unused]] const double opt_sec = opt_timer.Seconds();

    return std::make_tuple(error, outer_frame, InnerFrame(args));
  }
};

static void SolveHull(const Polyhedron &polyhedron, StatusBar *status,
                      std::optional<double> time_limit = std::nullopt) {
  HullSolver s(polyhedron, status, time_limit);
  s.Run();
}

// Try simultaneously optimizing both the shadow and hole. This is
// much slower because we can't frontload precomputation (e.g. of a
// convex hull). But it could be that the perpendicular axis needs to
// be just right in order for it to be solvable; Solve() spends most
// of its time trying different shapes of the hole and only random
// samples for the shadow.
struct SimulSolver : public Solver<SolutionDB::METHOD_SIMUL> {
  using Solver::Solver;

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
        return rotate * translate;
      };

    std::function<double(const std::array<double, D> &)> Loss =
      [this, &OuterFrame, &InnerFrame](
          const std::array<double, D> &args) {
        attempts++;
        frame3 outer_frame = OuterFrame(args);
        frame3 inner_frame = InnerFrame(args);
        Mesh2D souter = Shadow(Rotate(polyhedron, outer_frame));
        Mesh2D sinner = Shadow(Rotate(polyhedron, inner_frame));

        // Does every vertex in inner fall inside the outer shadow?
        double error = 0.0;
        int errors = 0;
        for (const vec2 &iv : sinner.vertices) {
          if (!InMesh(souter, iv)) {
            // slow :(
            error += DistanceToMesh(souter, iv);
            errors++;
          }
        }

        if (error == 0.0 && errors > 0) [[unlikely]] {
          // If they are not in the mesh, don't return an actual zero.
          return std::numeric_limits<double>::min() * errors;
        } else {
          return error;
        }
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

static void SolveSimul(const Polyhedron &polyhedron, StatusBar *status,
                       std::optional<double> time_limit = std::nullopt) {
  SimulSolver s(polyhedron, status, time_limit);
  s.Run();
}

// Third approach: Maximize the area of the outer polygon before
// optimizing the placement of the inner.
struct MaxSolver : public Solver<SolutionDB::METHOD_MAX> {
  using Solver::Solver;

  std::tuple<double, frame3, frame3> RunOne(ArcFour *rc) override {
    Timer prep_timer;
    quat4 outer_rot = RandomQuaternion(rc);

    auto OuterFrame = [&outer_rot](const std::array<double, 4> &args) {
        const auto &[di, dj, dk, dl] = args;
        quat4 tweaked_rot = normalize(quat4{
            .x = outer_rot.x + di,
            .y = outer_rot.y + dj,
            .z = outer_rot.z + dk,
            .w = outer_rot.w + dl,
          });
        frame3 rotate = yocto::rotation_frame(tweaked_rot);
        return rotate;
      };

    auto AreaLoss = [&](const std::array<double, 4> &args) {
        const frame3 outer_frame = OuterFrame(args);
        Polyhedron outer = Rotate(polyhedron, outer_frame);
        Mesh2D souter = Shadow(outer);

        // PERF: Now we want a faster convex hull algorithm...
        const std::vector<int> shadow_hull = QuickHull(souter.vertices);
        return -AreaOfHull(souter, shadow_hull);
      };

    const std::array<double, 4> area_lb =
      {-0.05, -0.05, -0.05, -0.05};
    const std::array<double, 4> area_ub =
      {+0.05, +0.05, +0.05, +0.05};

    const auto &[area_args, area_error] =
      Opt::Minimize<4>(AreaLoss, area_lb, area_ub, 1000, 1);

    const frame3 outer_frame = OuterFrame(area_args);
    Polyhedron outer = Rotate(polyhedron, outer_frame);
    Mesh2D souter = Shadow(outer);
    const std::vector<int> shadow_hull = QuickHull(souter.vertices);

    // Starting orientation/position for inner polyhedron.
    const quat4 inner_rot = RandomQuaternion(rc);

    static constexpr int D = 6;
    auto InnerFrame = [&inner_rot](const std::array<double, D> &args) {
        const auto &[di, dj, dk, dl, dx, dy] = args;
        quat4 tweaked_rot = normalize(quat4{
            .x = inner_rot.x + di,
            .y = inner_rot.y + dj,
            .z = inner_rot.z + dk,
            .w = inner_rot.w + dl,
          });
        frame3 rotate = yocto::rotation_frame(tweaked_rot);
        frame3 translate = yocto::translation_frame(
            vec3{.x = dx, .y = dy, .z = 0.0});
        return rotate * translate;
      };

    std::function<double(const std::array<double, D> &)> Loss =
      [this, &souter, &shadow_hull, &InnerFrame](
          const std::array<double, D> &args) {
        attempts++;
        frame3 frame = InnerFrame(args);
        Polyhedron inner = Rotate(polyhedron, frame);
        Mesh2D sinner = Shadow(inner);

        // Does every vertex in inner fall inside the outer shadow?
        double error = 0.0;
        int errors = 0;
        for (const vec2 &iv : sinner.vertices) {
          if (!InHull(souter, shadow_hull, iv)) {
            error += DistanceToHull(souter.vertices, shadow_hull, iv);
            errors++;
          }
        }

        if (error == 0.0 && errors > 0) [[unlikely]] {
          // If they are not in the mesh, don't return an actual zero.
          return std::numeric_limits<double>::min() * errors;
        } else {
          return error;
        }
      };

    const std::array<double, D> lb =
      {-0.15, -0.15, -0.15, -0.15, -0.25, -0.25};
    const std::array<double, D> ub =
      {+0.15, +0.15, +0.15, +0.15, +0.25, +0.25};
    [[maybe_unused]] const double prep_sec = prep_timer.Seconds();

    Timer opt_timer;
    const auto &[args, error] =
      Opt::Minimize<D>(Loss, lb, ub, 1000, 2);
    [[maybe_unused]] const double opt_sec = opt_timer.Seconds();

    return std::make_tuple(error, outer_frame, InnerFrame(args));
  }
};

static void SolveMax(const Polyhedron &polyhedron, StatusBar *status,
                       std::optional<double> time_limit = std::nullopt) {
  MaxSolver s(polyhedron, status, time_limit);
  s.Run();
}

[[maybe_unused]]
static quat4 AlignFaceNormalWithX(const std::vector<vec3> &vertices,
                                  const std::vector<int> &face) {
  if (face.size() < 3) return quat4{0.0, 0.0, 0.0, 1.0};
  const vec3 &v0 = vertices[face[0]];
  const vec3 &v1 = vertices[face[1]];
  const vec3 &v2 = vertices[face[2]];

  vec3 face_normal = yocto::normalize(yocto::cross(v1 - v0, v2 - v0));

  vec3 x_axis = vec3{1.0, 0.0, 0.0};
  vec3 rot_axis = yocto::cross(face_normal, x_axis);
  double rot_angle = yocto::angle(face_normal, x_axis);
  return QuatFromVec(yocto::rotation_quat(rot_axis, rot_angle));
}

// face1 and face2 must not be parallel. Rotate the polyhedron such
// that face1 and face2 are both parallel to the z axis. face1 is made
// perpendicular to the x axis, and then face2 perpendicular to the xy
// plane.
static quat4 MakeTwoFacesParallelToZ(const std::vector<vec3> &vertices,
                                     const std::vector<int> &face1,
                                     const std::vector<int> &face2) {
  if (face1.size() < 3 || face1.size() < 3)
    return quat4{0.0, 0.0, 0.0, 1.0};

  auto Normal = [&vertices](const std::vector<int> &face) {
      const vec3 &v0 = vertices[face[0]];
      const vec3 &v1 = vertices[face[1]];
      const vec3 &v2 = vertices[face[2]];

      return yocto::normalize(yocto::cross(v1 - v0, v2 - v0));
    };

  const vec3 face1_normal = Normal(face1);
  const vec3 face2_normal = Normal(face2);

  vec3 x_axis = vec3{1.0, 0.0, 0.0};
  vec3 rot_axis = yocto::cross(face1_normal, x_axis);
  double rot1_angle = yocto::angle(face1_normal, x_axis);

  quat4 rot1 = QuatFromVec(yocto::rotation_quat(rot_axis, rot1_angle));

  // Project face2's normal to the yz plane.
  vec3 proj_normal = vec3{0.0, face2_normal.y, face2_normal.z};
  double rot2_angle = yocto::angle(proj_normal, vec3{0.0, 1.0, 0.0});
  quat4 rot2 = QuatFromVec(yocto::rotation_quat({1.0, 0.0, 0.0}, rot2_angle));

  return normalize(rot1 * rot2);
}


// Third approach: Joint optimization, but place the inner in some
// orientation where a face is parallel to the z axis. Then only
// consider rotations around the z axis (and translations) for the
// inner.
struct ParallelSolver : public Solver<SolutionDB::METHOD_PARALLEL> {
  using Solver::Solver;

  std::tuple<double, frame3, frame3> RunOne(ArcFour *rc) override {
    // four params for outer rotation, and two for its
    // translation. The inner polyhedron is fixed.
    static constexpr int D = 6;

    Timer prep_timer;
    const quat4 initial_outer_rot = RandomQuaternion(rc);

    // Get two face indices that are not parallel.
    const auto &[face1, face2] = TwoNonParallelFaces(rc, polyhedron);

    const quat4 initial_inner_rot = MakeTwoFacesParallelToZ(
        polyhedron.vertices,
        polyhedron.faces->v[face1],
        polyhedron.faces->v[face2]);

    const frame3 initial_inner_frame =
      yocto::rotation_frame(initial_inner_rot);

    // The inner polyhedron is fixed, so we can compute its
    // convex hull once up front.
    const Mesh2D sinner = Shadow(Rotate(polyhedron, initial_inner_frame));
    const std::vector<vec2> inner_hull_pts = [&]() {
        const std::vector<int> inner_hull = QuickHull(sinner.vertices);
        std::vector<vec2> v;
        v.reserve(inner_hull.size());
        for (int p : inner_hull) {
          v.push_back(sinner.vertices[p]);
        }
        return v;
      }();

    // Get the frames from the appropriate positions in the
    // argument.

    auto OuterFrame = [&initial_outer_rot](
        const std::array<double, D> &args) {
        const auto &[o0, o1, o2, o3, dx, dy] = args;
        quat4 tweaked_rot = normalize(quat4{
            .x = initial_outer_rot.x + o0,
            .y = initial_outer_rot.y + o1,
            .z = initial_outer_rot.z + o2,
            .w = initial_outer_rot.w + o3,
          });
        frame3 translate = yocto::translation_frame(
            vec3{.x = dx, .y = dy, .z = 0.0});
        return yocto::rotation_frame(tweaked_rot) * translate;
      };

    // The inner polyhedron is fixed.
    auto InnerFrame = [&initial_inner_frame](
        const std::array<double, D> &args) -> const frame3 & {
        return initial_inner_frame;
      };

    std::function<double(const std::array<double, D> &)> Loss =
      [this, &OuterFrame, &inner_hull_pts](
          const std::array<double, D> &args) {
        attempts++;
        frame3 outer_frame = OuterFrame(args);
        Mesh2D souter = Shadow(Rotate(polyhedron, outer_frame));

        // Does every vertex in inner fall inside the outer shadow?
        double error = 0.0;
        int errors = 0;
        for (const vec2 &iv : inner_hull_pts) {
          if (!InMesh(souter, iv)) {
            // slow :(
            error += DistanceToMesh(souter, iv);
            errors++;
          }
        }

        if (error == 0.0 && errors > 0) [[unlikely]] {
          // If they are not in the mesh, don't return an actual zero.
          return std::numeric_limits<double>::min() * errors;
        } else {
          return error;
        }
      };

    constexpr double Q = 0.25;

    const std::array<double, D> lb = {-Q, -Q, -Q, -Q, -0.5, -0.5};
    const std::array<double, D> ub = {+Q, +Q, +Q, +Q, +0.5, +0.5};
    [[maybe_unused]] const double prep_sec = prep_timer.Seconds();

    Timer opt_timer;
    const auto &[args, error] =
      Opt::Minimize<D>(Loss, lb, ub, 1000, 2);
    [[maybe_unused]] const double opt_sec = opt_timer.Seconds();

    return std::make_tuple(error, OuterFrame(args), InnerFrame(args));
  }
};

static void SolveParallel(const Polyhedron &polyhedron, StatusBar *status,
                          std::optional<double> time_limit = std::nullopt) {
  ParallelSolver s(polyhedron, status, time_limit);
  s.Run();
}


// Solve a constrained problem:
//   - Both solids have their centers on the projection axis
//   - The inner solid has two of its faces aligned to the projection axis.
//
// TODO: We should additionally rotate the inner shadow so that it has
// one of those two faces aligned with the x axis.
struct SpecialSolver : public Solver<SolutionDB::METHOD_SPECIAL> {
  using Solver::Solver;

  std::tuple<double, frame3, frame3> RunOne(ArcFour *rc) override {
    // four params for outer orientation. The inner solid is
    // in a fixed orientation. Both are centered at the origin.
    static constexpr int D = 4;

    Timer prep_timer;
    const quat4 initial_outer_rot = RandomQuaternion(rc);

    // Get two face indices that are not parallel.
    const auto &[face1, face2] = TwoNonParallelFaces(rc, polyhedron);

    const quat4 initial_inner_rot = MakeTwoFacesParallelToZ(
        polyhedron.vertices,
        polyhedron.faces->v[face1],
        polyhedron.faces->v[face2]);

    const frame3 initial_inner_frame =
      yocto::rotation_frame(initial_inner_rot);

    const Mesh2D sinner = Shadow(Rotate(polyhedron, initial_inner_frame));
    const std::vector<vec2> inner_hull_pts = [&]() {
        const std::vector<int> inner_hull = QuickHull(sinner.vertices);
        std::vector<vec2> v;
        v.reserve(inner_hull.size());
        for (int p : inner_hull) {
          v.push_back(sinner.vertices[p]);
        }
        return v;
      }();

    // Get the frames from the appropriate positions in the
    // argument.
    auto OuterFrame = [&initial_outer_rot](
        const std::array<double, D> &args) {
        const auto &[o0, o1, o2, o3] = args;
        quat4 tweaked_rot = normalize(quat4{
            .x = initial_outer_rot.x + o0,
            .y = initial_outer_rot.y + o1,
            .z = initial_outer_rot.z + o2,
            .w = initial_outer_rot.w + o3,
          });
        return yocto::rotation_frame(tweaked_rot);
      };

    auto InnerFrame = [&initial_inner_frame](
        const std::array<double, D> &args) -> const frame3 & {
        return initial_inner_frame;
      };

    std::function<double(const std::array<double, D> &)> Loss =
      [this, &OuterFrame, &inner_hull_pts](
          const std::array<double, D> &args) {
        attempts++;
        frame3 outer_frame = OuterFrame(args);
        Mesh2D souter = Shadow(Rotate(polyhedron, outer_frame));

        // Does every vertex in inner fall inside the outer shadow?
        double error = 0.0;
        int errors = 0;
        for (const vec2 &iv : inner_hull_pts) {
          if (!InMesh(souter, iv)) {
            // slow :(
            error += DistanceToMesh(souter, iv);
            errors++;
          }
        }

        if (error == 0.0 && errors > 0) [[unlikely]] {
          // If they are not in the mesh, don't return an actual zero.
          return std::numeric_limits<double>::min() * errors;
        } else {
          return error;
        }
      };

    constexpr double Q = 0.25;

    const std::array<double, D> lb = {-Q, -Q, -Q, -Q};
    const std::array<double, D> ub = {+Q, +Q, +Q, +Q};
    [[maybe_unused]] const double prep_sec = prep_timer.Seconds();

    Timer opt_timer;
    const auto &[args, error] =
      Opt::Minimize<D>(Loss, lb, ub, 1000, 2);
    [[maybe_unused]] const double opt_sec = opt_timer.Seconds();

    return std::make_tuple(error, OuterFrame(args), InnerFrame(args));
  }
};

static void SolveSpecial(const Polyhedron &polyhedron, StatusBar *status,
                         std::optional<double> time_limit = std::nullopt) {
  SpecialSolver s(polyhedron, status, time_limit);
  s.Run();
}


// Rotation-only solutions (no translation of either polyhedron).
struct OriginSolver : public Solver<SolutionDB::METHOD_ORIGIN> {
  using Solver::Solver;

  std::tuple<double, frame3, frame3> RunOne(ArcFour *rc) override {
    static constexpr int D = 8;

    Timer prep_timer;
    const quat4 initial_outer_rot = RandomQuaternion(rc);
    const quat4 initial_inner_rot = RandomQuaternion(rc);

    // Get the frames from the appropriate positions in the
    // argument.
    auto OuterFrame = [&initial_outer_rot](
        const std::array<double, D> &args) {
        const auto &[o0, o1, o2, o3, i0_, i1_, i2_, i3_] = args;
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
        const auto &[o0_, o1_, o2_, o3_, i0, i1, i2, i3] = args;
        quat4 tweaked_rot = normalize(quat4{
            .x = initial_inner_rot.x + i0,
            .y = initial_inner_rot.y + i1,
            .z = initial_inner_rot.z + i2,
            .w = initial_inner_rot.w + i3,
          });
        return yocto::rotation_frame(tweaked_rot);
      };

    std::function<double(const std::array<double, D> &)> Loss =
        [this, &OuterFrame, &InnerFrame](const std::array<double, D> &args) {
          attempts++;
          frame3 outer_frame = OuterFrame(args);
          Mesh2D souter = Shadow(Rotate(polyhedron, outer_frame));
          frame3 inner_frame = InnerFrame(args);
          Mesh2D sinner = Shadow(Rotate(polyhedron, inner_frame));

          // Does every vertex in inner fall inside the outer shadow?
          double error = 0.0;
          int errors = 0;
          for (const vec2 &iv : sinner.vertices) {
            if (!InMesh(souter, iv)) {
              // slow :(
              error += DistanceToMesh(souter, iv);
              errors++;
            }
          }

          if (error == 0.0 && errors > 0) [[unlikely]] {
            // If they are not in the mesh, don't return an actual zero.
            return std::numeric_limits<double>::min() * errors;
          } else {
            return error;
          }
        };

    constexpr double Q = 0.25;

    const std::array<double, D> lb = {-Q, -Q, -Q, -Q, -Q, -Q, -Q, -Q};
    const std::array<double, D> ub = {+Q, +Q, +Q, +Q, +Q, +Q, +Q, +Q};
    [[maybe_unused]] const double prep_sec = prep_timer.Seconds();

    Timer opt_timer;
    const auto &[args, error] = Opt::Minimize<D>(Loss, lb, ub, 1000, 2);
    [[maybe_unused]] const double opt_sec = opt_timer.Seconds();

    return std::make_tuple(error, OuterFrame(args), InnerFrame(args));
  }
};

static void SolveOrigin(const Polyhedron &polyhedron, StatusBar *status,
                        std::optional<double> time_limit = std::nullopt) {
  OriginSolver s(polyhedron, status, time_limit);
  s.Run();
}

static void SolveWith(const Polyhedron &poly, int method, StatusBar *status,
                      std::optional<double> time_limit) {
  status->Printf("Solve " AYELLOW("%s") " with " AWHITE("%s") "...\n",
                 poly.name,
                 SolutionDB::MethodName(method));

  switch (method) {
  case SolutionDB::METHOD_HULL:
    return SolveHull(poly, status, time_limit);
  case SolutionDB::METHOD_SIMUL:
    return SolveSimul(poly, status, time_limit);
  case SolutionDB::METHOD_MAX:
    return SolveMax(poly, status, time_limit);
  case SolutionDB::METHOD_PARALLEL:
    return SolveParallel(poly, status, time_limit);
  case SolutionDB::METHOD_SPECIAL:
    return SolveSpecial(poly, status, time_limit);
  case SolutionDB::METHOD_ORIGIN:
    return SolveOrigin(poly, status, time_limit);
  default:
    LOG(FATAL) << "Method not available";
  }
}

static void ReproduceEasySolutions(
    // The solution method to apply.
    int method,
    // If true, try hard cases (no known solution) as well
    bool hard,
    // Time limit, in seconds, per solve call
    double time_limit) {

  std::vector<SolutionDB::Solution> sols = []() {
      SolutionDB db;
      return db.GetAllSolutions();
    }();
  auto HasSolutionWithMethod = [&](const Polyhedron &poly) {
      for (const auto &sol : sols)
        if (sol.method == method && sol.polyhedron == poly.name)
          return true;
      return false;
    };

  StatusBar status(3 + HISTO_LINES);

  auto MaybeSolve = [&](Polyhedron poly) {
      if (HasSolutionWithMethod(poly)) {
        status.Printf(
            "Already solved " AYELLOW("%s") " with " AWHITE("%s") "\n",
            poly.name, SolutionDB::MethodName(method));
      } else {
        SolveWith(poly, method, &status, time_limit);
      }
    };

  // Platonic
  if (hard || method != SolutionDB::METHOD_SPECIAL) {
    MaybeSolve(Tetrahedron());
  }
  MaybeSolve(Cube());
  MaybeSolve(Dodecahedron());
  MaybeSolve(Icosahedron());
  MaybeSolve(Octahedron());

  // Archimedean
  if (hard || method != SolutionDB::METHOD_SPECIAL) {
    MaybeSolve(TruncatedTetrahedron());
  }
  // Hard?
  if (hard) {
    MaybeSolve(SnubCube());
  }
  MaybeSolve(Cuboctahedron());
  MaybeSolve(TruncatedCube());
  MaybeSolve(TruncatedOctahedron());
  MaybeSolve(Rhombicuboctahedron());
  MaybeSolve(Icosidodecahedron());
  MaybeSolve(TruncatedIcosahedron());
  MaybeSolve(TruncatedDodecahedron());
  MaybeSolve(TruncatedIcosidodecahedron());
  MaybeSolve(TruncatedCuboctahedron());
  // Hard?
  if (hard) {
    MaybeSolve(Rhombicosidodecahedron());
  }
  MaybeSolve(TruncatedIcosidodecahedron());
  if (hard) {
    MaybeSolve(SnubDodecahedron());
  }

  // Catalan
  // Hard?
  if (hard) {
    MaybeSolve(TriakisTetrahedron());
  }
  MaybeSolve(RhombicDodecahedron());
  MaybeSolve(TriakisOctahedron());
  MaybeSolve(TetrakisHexahedron());
  MaybeSolve(DeltoidalIcositetrahedron());
  MaybeSolve(DisdyakisDodecahedron());
  if (hard) {
    // New: Not sure it's actually hard, but it didn't
    // get solved immediately
    MaybeSolve(DeltoidalHexecontahedron());
  }
  if (hard || method != SolutionDB::METHOD_SPECIAL) {
    MaybeSolve(PentagonalIcositetrahedron());
  }
  MaybeSolve(RhombicTriacontahedron());
  MaybeSolve(TriakisIcosahedron());
  MaybeSolve(PentakisDodecahedron());
  MaybeSolve(DisdyakisTriacontahedron());
  MaybeSolve(DeltoidalIcositetrahedron());
  // Hard?
  if (hard) {
    MaybeSolve(PentagonalHexecontahedron());
  }
}

static void GrindRandom() {
  std::vector<Polyhedron> all = {
    Tetrahedron(),
    Cube(),
    Dodecahedron(),
    Icosahedron(),
    Octahedron(),

    // Archimedean
    TruncatedTetrahedron(),
    Cuboctahedron(),
    TruncatedCube(),
    TruncatedOctahedron(),
    Rhombicuboctahedron(),
    TruncatedCuboctahedron(),
    SnubCube(),
    Icosidodecahedron(),
    TruncatedDodecahedron(),
    TruncatedIcosahedron(),
    Rhombicosidodecahedron(),
    TruncatedIcosidodecahedron(),
    SnubDodecahedron(),

    // Catalan
    TriakisTetrahedron(),
    RhombicDodecahedron(),
    TriakisOctahedron(),
    TetrakisHexahedron(),
    DeltoidalIcositetrahedron(),
    DisdyakisDodecahedron(),
    DeltoidalHexecontahedron(),
    PentagonalIcositetrahedron(),
    RhombicTriacontahedron(),
    TriakisIcosahedron(),
    PentakisDodecahedron(),
    DisdyakisTriacontahedron(),
    PentagonalHexecontahedron(),
  };

  std::vector<SolutionDB::Solution> sols = []() {
      SolutionDB db;
      return db.GetAllSolutions();
    }();
  auto HasSolutionWithMethod = [&](const Polyhedron &poly, int method) {
      for (const auto &sol : sols)
        if (sol.method == method && sol.polyhedron == poly.name)
          return true;
      return false;
    };

  std::vector<std::pair<const Polyhedron *, int>> remaining;
  for (const Polyhedron &poly : all) {
    printf(AWHITE("%s") ":", poly.name);
    bool has_solution = false;
    for (int method : {
        SolutionDB::METHOD_HULL,
        SolutionDB::METHOD_SIMUL,
        SolutionDB::METHOD_MAX,
        SolutionDB::METHOD_PARALLEL,
        SolutionDB::METHOD_SPECIAL,
        SolutionDB::METHOD_ORIGIN}) {
      if (HasSolutionWithMethod(poly, method)) {
        has_solution = true;
        std::string name = Util::lcase(SolutionDB::MethodName(method));
        (void)Util::TryStripPrefix("method_", &name);
        printf(" " ACYAN("%s"), name.c_str());
      } else {
        remaining.emplace_back(&poly, method);
        /*
          printf("Unsolved: " AWHITE("%s") " with " ACYAN("%s") "\n",
          poly.name, SolutionDB::MethodName(method));
        */
      }
    }

    if (has_solution) {
      printf("\n");
    } else {
      printf(" " ARED("unsolved") "\n");
    }
  }

  printf("Total remaining: " APURPLE("%d") "\n", (int)remaining.size());

StatusBar status(3 + HISTO_LINES);
  ArcFour rc(StringPrintf("grind.%lld", time(nullptr)));
  for (;;) {
    int idx = RandTo(&rc, remaining.size());
    const auto &[poly, method] = remaining[idx];
    SolveWith(*poly, method, &status, 3600.0);
    using namespace std::chrono_literals;
    std::this_thread::sleep_for(1s);
  }
}

int main(int argc, char **argv) {
  ANSI::Init();
  printf("\n");

  // Grind every unsolved cell.
  if (true) {
    GrindRandom();
    return 0;
  }

  // Grind unsolved polyhedra for an hour at a time.
  if (true) {
    for (;;) {
      using namespace std::chrono_literals;
      ReproduceEasySolutions(SolutionDB::METHOD_PARALLEL, true, 3600.0);
      std::this_thread::sleep_for(1s);
      ReproduceEasySolutions(SolutionDB::METHOD_HULL, true, 3600.0);
      std::this_thread::sleep_for(1s);
      ReproduceEasySolutions(SolutionDB::METHOD_SIMUL, true, 3600.0);
      std::this_thread::sleep_for(1s);
      ReproduceEasySolutions(SolutionDB::METHOD_MAX, true, 3600.0);
      std::this_thread::sleep_for(1s);
      ReproduceEasySolutions(SolutionDB::METHOD_SPECIAL, true, 3600.0);
      std::this_thread::sleep_for(1s);
    }
  }

  if (false) {
    // ReproduceEasySolutions(SolutionDB::METHOD_SPECIAL, 3600.0);
    ReproduceEasySolutions(SolutionDB::METHOD_SIMUL, false, 60.0);
    printf("OK\n");
    return 0;
  }


  // Polyhedron target = SnubCube();
  // Polyhedron target = Rhombicosidodecahedron();
  // Polyhedron target = TruncatedCuboctahedron();
  // Polyhedron target = TruncatedDodecahedron();
  // Polyhedron target = TruncatedOctahedron();
  // Polyhedron target = TruncatedTetrahedron();
  // Polyhedron target = TruncatedIcosahedron();
  // Polyhedron target = TruncatedIcosidodecahedron();
  // Polyhedron target = SnubDodecahedron();
  // Polyhedron target = Dodecahedron();
  // Polyhedron target = RhombicTriacontahedron();
  // Polyhedron target = TriakisIcosahedron();
  // Polyhedron target = PentakisDodecahedron();
  // Polyhedron target = DisdyakisTriacontahedron();
  // Polyhedron target = DeltoidalIcositetrahedron();
  // Polyhedron target = PentakisDodecahedron();
  // Polyhedron target = PentagonalHexecontahedron();
  Polyhedron target = DeltoidalHexecontahedron();

  // Call one of the solution procedures:

  StatusBar status(3 + HISTO_LINES);

  SolveSimul(target, &status);

  printf("OK\n");
  return 0;
}
