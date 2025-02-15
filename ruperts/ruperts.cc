
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
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
#include <unordered_set>
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

static constexpr bool WRITE_BEST_IMAGES = false;

using vec2 = yocto::vec<double, 2>;
using vec3 = yocto::vec<double, 3>;
using vec4 = yocto::vec<double, 4>;
using mat4 = yocto::mat<double, 4>;
using quat4 = yocto::quat<double, 4>;
using frame3 = yocto::frame<double, 3>;

// These are all de novo, so the source is always zero.
static constexpr int SOURCE = 0;

static void SaveSolution(const Polyhedron &poly,
                         const frame3 &outer_frame,
                         const frame3 &inner_frame,
                         int method) {

  {
    Polyhedron outer = Rotate(poly, outer_frame);
    Polyhedron inner = Rotate(poly, inner_frame);
    Mesh2D souter = Shadow(outer);
    Mesh2D sinner = Shadow(inner);
    std::vector<int> outer_hull = QuickHull(souter.vertices);
    std::vector<int> inner_hull = QuickHull(sinner.vertices);

    Rendering rendering(poly, 3840, 2160);
    rendering.RenderHull(souter, outer_hull, 0xAA0000FF);
    rendering.RenderHull(sinner, inner_hull, 0x00FF00AA);
    rendering.Save(StringPrintf("hulls-%s.png", poly.name));
  }

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

  SolutionDB db;
  db.AddSolution(poly.name, outer_frame, inner_frame,
                 method, SOURCE, ratio, clearance);


  printf("Added solution (" AYELLOW("%s") ") to database with "
         "ratio " APURPLE("%.17g") ", clearance " ABLUE("%.17g") "\n",
         poly.name, ratio, clearance);
}

static constexpr int NUM_THREADS = 4;
static constexpr int HISTO_LINES = 32;
static constexpr int STATUS_LINES = HISTO_LINES + 3;

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
              db.AddAttempt(polyhedron.name, METHOD, SOURCE,
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
              if (WRITE_BEST_IMAGES &&
                  iters.Read() > 4096 &&
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
                status->Statusf(
                    "%s\n"
                    "%s\n"
                    "%s iters, %s attempts; best: %.11g"
                    " [" ACYAN("%.3f") "/s]\n",
                    error_histo.SimpleANSI(HISTO_LINES).c_str(),
                    bar.c_str(),
                    FormatNum(it).c_str(),
                    FormatNum(attempts.Read()).c_str(),
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


struct HullSolver : public Solver<SolutionDB::METHOD_HULL> {
  using Solver::Solver;

  std::tuple<double, frame3, frame3> RunOne(ArcFour *rc) override {
    Timer prep_timer;
    quat4 outer_rot = RandomQuaternion(rc);
    const frame3 outer_frame = yocto::rotation_frame(outer_rot);
    Polyhedron outer = Rotate(polyhedron, outer_frame);
    Mesh2D souter = Shadow(outer);

    const std::vector<int> shadow_hull = QuickHull(souter.vertices);
    // PERF: HullCircle would probably be helpful here.

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
        return translate * rotate;
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
        return translate * rotate;
      };

    std::function<double(const std::array<double, D> &)> Loss =
      [this, &OuterFrame, &InnerFrame](
          const std::array<double, D> &args) {
        attempts++;
        return LossFunction(polyhedron, OuterFrame(args), InnerFrame(args));
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
        return translate * rotate;
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

static vec3 FaceNormal(const std::vector<vec3> &vertices,
                       const std::vector<int> &face) {
  CHECK(face.size() >= 3);
  const vec3 &v0 = vertices[face[0]];
  const vec3 &v1 = vertices[face[1]];
  const vec3 &v2 = vertices[face[2]];
  return yocto::normalize(yocto::cross(v1 - v0, v2 - v0));
}

// face1 and face2 must not be parallel. Rotate the polyhedron such
// that face1 and face2 are both parallel to the z axis.
static quat4 MakeTwoFacesParallelToZ(const std::vector<vec3> &vertices,
                                     const std::vector<int> &face1,
                                     const std::vector<int> &face2) {
  if (face1.size() < 3 || face2.size() < 3)
    return quat4{0.0, 0.0, 0.0, 1.0};

  const vec3 face1_normal = FaceNormal(vertices, face1);
  const vec3 face2_normal = FaceNormal(vertices, face2);

  vec3 x_axis = vec3{1.0, 0.0, 0.0};
  vec3 rot_axis = yocto::cross(face1_normal, x_axis);
  double rot1_angle = yocto::angle(face1_normal, x_axis);

  quat4 rot1 = QuatFromVec(yocto::rotation_quat(rot_axis, rot1_angle));

  // Work with the face2 normal after rot1 is applied.
  const vec3 rot_face2_normal =
    yocto::transform_direction(yocto::rotation_frame(rot1), face2_normal);

  // Project face2's normal to the yz plane.
  vec3 proj_normal = vec3{0.0, rot_face2_normal.y, rot_face2_normal.z};
  double rot2_angle = yocto::angle(proj_normal, vec3{0.0, 1.0, 0.0});
  quat4 rot2 = QuatFromVec(yocto::rotation_quat({1.0, 0.0, 0.0}, rot2_angle));

  return normalize(rot2 * rot1);
}

// Make the two faces parallel to the z axis (as above) and then rotate
// around z such that face1 is is aligned with the y axis.
static quat4 AlignFaces(const std::vector<vec3> &vertices,
                        const std::vector<int> &face1,
                        const std::vector<int> &face2) {

  const quat4 parallel_inner_rot = MakeTwoFacesParallelToZ(
      vertices, face1, face2);

  // Now, compute the additional rotation around the z-axis.
  // First, get the transformed normal of face1.
  const vec3 face1_normal = FaceNormal(vertices, face1);

  // The face1 normal after the rotation. This is in the xy plane
  // by the construction above.
  const vec3 xy_normal = yocto::transform_direction(
      yocto::rotation_frame(parallel_inner_rot), face1_normal);

  CHECK(std::abs(xy_normal.z) < 0.01) << "The rotated face 1 normal "
    "should already be in the x/y plane: " << VecString(xy_normal);

  // Compute the angle to the y-axis. Since this is the normal, which
  // is perpendicular to the face, we then add 90 degrees.
  const double rot3_angle = std::atan2(xy_normal.x, xy_normal.y) +
    (std::numbers::pi * 0.5);
  const quat4 rot3 =
    QuatFromVec(yocto::rotation_quat({0.0, 0.0, 1.0}, rot3_angle));

  // Apply the additional rotation to the inner rotation.
  return normalize(rot3 * parallel_inner_rot);
}

// Third approach: Joint optimization, but place the inner in some
// orientation where two faces are parallel to the z axis and one is
// also aligned with they the y axis. The outer polyhedron is
// unconstrained.
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

    // Align 'em.
    const quat4 initial_inner_rot = AlignFaces(
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

        // Computing the outer hull is still much faster (at least for
        // snub cube) even though we have a reduced set of inner
        // points.
        const std::vector<int> outer_hull = GrahamScan(souter.vertices);
        HullInscribedCircle circle(souter.vertices, outer_hull);

        // Does every vertex in inner fall inside the outer shadow?
        double error = 0.0;
        int errors = 0;
        for (const vec2 &iv : inner_hull_pts) {
          if (circle.DefinitelyInside(iv))
            continue;

          if (!InHull(souter, outer_hull, iv)) {
            // slow :(
            error += DistanceToHull(souter.vertices, outer_hull, iv);
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

    const quat4 initial_inner_rot = AlignFaces(
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

        // Computing the outer hull is still much faster (at least for
        // snub cube) even though we have a reduced set of inner
        // points.
        const std::vector<int> outer_hull = GrahamScan(souter.vertices);
        HullInscribedCircle circle(souter.vertices, outer_hull);

        // Does every vertex in inner fall inside the outer shadow?
        double error = 0.0;
        int errors = 0;
        for (const vec2 &iv : inner_hull_pts) {
          if (circle.DefinitelyInside(iv))
            continue;

          if (!InHull(souter, outer_hull, iv)) {
            // slow :(
            error += DistanceToHull(souter.vertices, outer_hull, iv);
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
          return LossFunction(polyhedron, OuterFrame(args), InnerFrame(args));
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

// Simultaneous optimization, but starting from a transformation that is
// almost the identity.
struct AlmostIdSolver : public Solver<SolutionDB::METHOD_ALMOST_ID> {
  using Solver::Solver;

  std::tuple<double, frame3, frame3> RunOne(ArcFour *rc) override {
    // four params for outer rotation, four params for inner
    // rotation, two for 2d translation of inner.
    static constexpr int D = 10;

    Timer prep_timer;
    const quat4 initial_outer_rot = RandomQuaternion(rc);
    // ... use the same orientation for both.
    const quat4 initial_inner_rot = initial_outer_rot;

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
        return LossFunction(polyhedron, OuterFrame(args), InnerFrame(args));
      };

    constexpr double Q = 0.001;

    const std::array<double, D> lb =
      {-Q, -Q, -Q, -Q,
       -Q, -Q, -Q, -Q, -0.001, -0.001};
    const std::array<double, D> ub =
      {+Q, +Q, +Q, +Q,
       +Q, +Q, +Q, +Q, +0.001, +0.001};
    [[maybe_unused]] const double prep_sec = prep_timer.Seconds();

    Timer opt_timer;
    const auto &[args, error] =
      Opt::Minimize<D>(Loss, lb, ub, 1000, 2);
    [[maybe_unused]] const double opt_sec = opt_timer.Seconds();

    return std::make_tuple(error, OuterFrame(args), InnerFrame(args));
  }
};

static void SolveAlmostId(const Polyhedron &polyhedron, StatusBar *status,
                       std::optional<double> time_limit = std::nullopt) {
  AlmostIdSolver s(polyhedron, status, time_limit);
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
  case SolutionDB::METHOD_ALMOST_ID:
    return SolveAlmostId(poly, status, time_limit);
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

  StatusBar status(STATUS_LINES);

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

static void GrindNoperts() {
  using Nopert = SolutionDB::Nopert;
  using Solution = SolutionDB::Solution;
  ArcFour rc(std::format("grind.noperts.{}", time(nullptr)));
  SolutionDB db;
  std::unordered_set<int> banned;

  for (;;) {
    std::vector<Nopert> all_noperts = db.GetAllNoperts();
    std::vector<Nopert> noperts_unsolved;

    for (Nopert &nopert : all_noperts) {
      std::string name = SolutionDB::NopertName(nopert.id);
      std::vector<Solution> sols = db.GetSolutionsFor(name);
      if (sols.empty() && !banned.contains(nopert.id)) {
        noperts_unsolved.push_back(std::move(nopert));
      }
    }

    if (noperts_unsolved.empty()) {
      printf("All the noperts are solved (or banned).\n");
      return;
    }

    // Otherwise, pick one and grind it.
    std::sort(noperts_unsolved.begin(),
              noperts_unsolved.end(),
              [](const auto &a, const auto &b) {
                if (a.vertices.size() == b.vertices.size()) {
                  return a.id < b.id;
                } else {
                  return a.vertices.size() < b.vertices.size();
                }
              });

    const Nopert &nopert =
      (rc.Byte() & 1) ?
      // smallest unsolved
      noperts_unsolved[0] :
      // or randomly
      noperts_unsolved[RandTo(&rc, noperts_unsolved.size())];

    // Storage for name inside poly, which is just a char *.
    std::string name = SolutionDB::NopertName(nopert.id);

    std::optional<Polyhedron> opoly =
      PolyhedronFromVertices(nopert.vertices, name.c_str());

    if (!opoly.has_value()) {
      printf("Error constructing nopert #%d\n", nopert.id);
      banned.insert(nopert.id);
      std::this_thread::sleep_for(std::chrono::seconds(30));
      continue;
    }

    static constexpr int method = SolutionDB::METHOD_SIMUL;
    StatusBar status(STATUS_LINES);
    status.Printf("Try solving nopert #" APURPLE("%d") " with "
                  ABLUE("%d") " vertices...\n",
                  nopert.id, (int)nopert.vertices.size());
    SolveWith(opoly.value(), method, &status, 3600.0);

    delete opoly.value().faces;
  }
}

static void GrindRandom(const std::unordered_set<std::string> &poly_filter) {
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

  auto GetRemaining = [&all, &poly_filter]() {
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
            SolutionDB::METHOD_ORIGIN,
            SolutionDB::METHOD_ALMOST_ID}) {
          if (HasSolutionWithMethod(poly, method)) {
            has_solution = true;
            std::string name = Util::lcase(SolutionDB::MethodName(method));
            (void)Util::TryStripPrefix("method_", &name);
            printf(" " ACYAN("%s"), name.c_str());
          } else if (poly_filter.empty() || poly_filter.contains(poly.name)) {
            remaining.emplace_back(&poly, method);
          }
        }

        if (has_solution) {
          printf("\n");
        } else {
          printf(" " ARED("unsolved") "\n");
        }
      }

      printf("Total remaining: " APURPLE("%d") "\n", (int)remaining.size());
      return remaining;
    };

  StatusBar status(STATUS_LINES);
  ArcFour rc(StringPrintf("grind.%lld", time(nullptr)));
  Periodically database_per(10.0 * 60.0);
  std::vector<std::pair<const Polyhedron *, int>> remaining;
  for (;;) {
    if (database_per.ShouldRun() || remaining.empty()) {
      remaining = GetRemaining();
    }
    int idx = RandTo(&rc, remaining.size());
    const auto &[poly, method] = remaining[idx];
    SolveWith(*poly, method, &status, 3600.0);
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }
}

int main(int argc, char **argv) {
  ANSI::Init();
  printf("\n");

  StatusBar status(STATUS_LINES);

  if (argc > 1) {
    std::string name = argv[1];
    Polyhedron poly = PolyhedronByName(name);

    for (;;) {
      SolveWith(poly, SolutionDB::METHOD_SIMUL, &status, 3600.0);
      SolveWith(poly, SolutionDB::METHOD_HULL, &status, 3600.0);
      SolveWith(poly, SolutionDB::METHOD_MAX, &status, 3600.0);
    }
  }

  if (false) {
    GrindNoperts();
  }

  // Grind every unsolved cell.
  if (false) {
    std::unordered_set<std::string> poly_filter;
    for (int i = 1; i < argc; i++) {
      poly_filter.insert(argv[i]);
    }
    GrindRandom(poly_filter);
    return 0;
  }

  // Grind unsolved polyhedra for an hour at a time.
  if (false) {
    for (;;) {
      constexpr auto sec = std::chrono::seconds(1);
      ReproduceEasySolutions(SolutionDB::METHOD_PARALLEL, true, 3600.0);
      std::this_thread::sleep_for(sec);
      ReproduceEasySolutions(SolutionDB::METHOD_HULL, true, 3600.0);
      std::this_thread::sleep_for(sec);
      ReproduceEasySolutions(SolutionDB::METHOD_SIMUL, true, 3600.0);
      std::this_thread::sleep_for(sec);
      ReproduceEasySolutions(SolutionDB::METHOD_MAX, true, 3600.0);
      std::this_thread::sleep_for(sec);
      ReproduceEasySolutions(SolutionDB::METHOD_SPECIAL, true, 3600.0);
      std::this_thread::sleep_for(sec);
    }
  }

  if (true) {
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

  SolveSimul(target, &status);

  printf("OK\n");
  return 0;
}
