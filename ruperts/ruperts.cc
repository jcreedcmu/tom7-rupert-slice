
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <functional>
#include <limits>
#include <mutex>
#include <numbers>
#include <string>
#include <utility>
#include <vector>

#include "ansi.h"
#include "arcfour.h"
#include "atomic-util.h"
#include "auto-histo.h"
#include "base/logging.h"
#include "base/stringprintf.h"
#include "image.h"
#include "mov-recorder.h"
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

[[maybe_unused]]
static void AnimateMesh(const Polyhedron &poly) {
  ArcFour rc("animate");
  quat4 initial_rot = RandomQuaternion(&rc);

  constexpr int SIZE = 1080;
  constexpr int FRAMES = 10 * 60;
  MovRecorder rec(StringPrintf("animate-%s.mov", poly.name), SIZE, SIZE);

  StatusBar status(2);
  Periodically status_per(1.0);
  for (int i = 0; i < FRAMES; i++) {
    if (status_per.ShouldRun()) {
      status.Progressf(i, FRAMES, "rotate");
    }

    double t = i / (double)FRAMES;
    double angle = t * 2.0 * std::numbers::pi;

    // rotation quat actually returns vec4; isomorphic to quat4.
    quat4 frame_rot =
      QuatFromVec(yocto::rotation_quat<double>({0.0, 1.0, 0.0}, angle));

    quat4 final_rot = normalize(initial_rot * frame_rot);
    Polyhedron rpoly = Rotate(poly, yocto::rotation_frame(final_rot));

    Rendering rendering(poly, SIZE, SIZE);
    Mesh2D mesh = Shadow(rpoly);
    rendering.RenderMesh(mesh);
    rec.AddFrame(std::move(rendering.img));
  }
}

[[maybe_unused]]
static void AnimateHull() {
  ArcFour rc("animate");

  constexpr int WIDTH = 1920;
  constexpr int HEIGHT = 1080;
  constexpr int SIZE = HEIGHT;
  constexpr int FRAMES = 10 * 60;
  constexpr int POINTS = 100;
  MovRecorder rec("animate-hull.mov", WIDTH, HEIGHT);

  std::vector<vec2> points;
  std::vector<vec2> vels;

  RandomGaussian gauss(&rc);
  for (int i = 0; i < POINTS; i++) {
    double x =
      std::clamp(gauss.Next() * SIZE * 0.1 + SIZE * 0.5, 0.0, (double)SIZE);
    double y =
      std::clamp(gauss.Next() * SIZE * 0.1 + SIZE * 0.5, 0.0, (double)SIZE);
    points.emplace_back(vec2{x, y});
    vels.emplace_back(
        vec2{
          .x = RandDouble(&rc) * 8.0 - 4.0,
          .y = RandDouble(&rc) * 8.0 - 4.0,
        });
  }

  double sec1 = 0.0, sec2 = 0.0;

  StatusBar status(2);
  Periodically status_per(1.0);
  for (int i = 0; i < FRAMES; i++) {
    if (status_per.ShouldRun()) {
      status.Progressf(i, FRAMES, "hull");
    }

    ImageRGBA img(WIDTH, HEIGHT);
    img.Clear32(0x000000FF);

    for (int i = 0; i < (int)points.size(); i++) {
      img.BlendFilledCircle32(points[i].x, points[i].y, 6.0f,
                              (Rendering::Color(i) & 0xFFFFFFAA) |
                              0x33333300);
      img.BlendCircle32(points[i].x, points[i].y, 6.0f, 0xFFFFFF44);
    }

    Timer timer1;
    std::vector<int> hull1 = GrahamScan(points);
    sec1 += timer1.Seconds();

    Timer timer2;
    std::vector<int> hull2 = QuickHull(points);
    sec2 += timer2.Seconds();

    // printf("Got hull sized %d, %d\n", (int)hull1.size(), (int)hull2.size());

    auto DrawHull = [&](const std::vector<int> &hull, int32_t color) {
        for (int i = 0; i < hull.size(); i++) {
          const vec2 &a = points[hull[i]];
          const vec2 &b = points[hull[(i + 1) % hull.size()]];

          img.BlendThickLine32(a.x, a.y, b.x, b.y, 2.0f, color);
        }
      };

    DrawHull(hull1, 0xFFFFFF33);
    // DrawHull(hull2, 0x00FFFF33);

    // img.Save(StringPrintf("hull%d.png", i));

    rec.AddFrame(std::move(img));

    for (int i = 0; i < (int)points.size(); i++) {
      points[i] += vels[i];
      if (points[i].x < 0.0) {
        points[i].x = 0.0;
        vels[i].x = -vels[i].x;
      }
      if (points[i].y < 0.0) {
        points[i].y = 0.0;
        vels[i].y = -vels[i].y;
      }

      if (points[i].x > SIZE) {
        points[i].x = SIZE;
        vels[i].x = -vels[i].x;
      }

      if (points[i].y > SIZE) {
        points[i].y = SIZE;
        vels[i].y = -vels[i].y;
      }
    }

    // gravity :)
    for (vec2 &d : vels) {
      d += vec2{0.0, 0.05};
    }

  }

  printf("Hull1: %s. Hull2: %s\n",
         ANSI::Time(sec1).c_str(), ANSI::Time(sec2).c_str());

}

[[maybe_unused]]
static void Visualize(const Polyhedron &poly) {
  // ArcFour rc(StringPrintf("seed.%lld", time(nullptr)));
  ArcFour rc("fixed-seed");

  CHECK(PlanarityError(poly) < 1.0e-10);
  printf("Planarity OK.\n");

  {
    Rendering rendering(poly, 1920, 1080);
    for (int i = 0; i < 5; i++) {
      frame3 frame = yocto::rotation_frame(RandomQuaternion(&rc));
      Polyhedron rpoly = Rotate(poly, frame);

      CHECK(PlanarityError(rpoly) < 1.0e10);
      rendering.RenderPerspectiveWireframe(rpoly, Rendering::Color(i));
    }

    rendering.Save(StringPrintf("wireframe-%s.png", poly.name));
  }

  {
    Rendering rendering(poly, 1920, 1080);
    // quat4 q = RandomQuaternion(&rc);
    // frame3 frame = yocto::rotation_frame(q);
    // Polyhedron rpoly = Rotate(poly, frame);

    Mesh2D mesh = Shadow(poly);
    rendering.RenderMesh(mesh);

    printf("Get convex hull (%d vertices):\n",
           (int)mesh.vertices.size());
    std::vector<int> hull = QuickHull(mesh.vertices);
    printf("Hull size %d\n", (int)hull.size());
    // rendering.RenderHull(mesh, hull);

    rendering.Save(StringPrintf("shadow-%s.png", poly.name));
  }
}

static constexpr int HISTO_LINES = 32;

[[maybe_unused]]
static void Solve(const Polyhedron &polyhedron, StatusBar *status) {
  // ArcFour rc(StringPrintf("solve.%lld", time(nullptr)));

  status->Printf("Solve [method 1] " AWHITE("%s") ":\n",
                 polyhedron.name);

  std::mutex m;
  bool should_die = false;
  Timer run_timer;
  Periodically status_per(1.0);
  Periodically image_per(10.0);
  double best_error = 1.0e42;
  AutoHisto error_histo(100000);
  constexpr int NUM_THREADS = 4;

  double prep_time = 0.0, opt_time = 0.0;

  ParallelFan(
      NUM_THREADS,
      [&](int thread_idx) {
        ArcFour rc(StringPrintf("solve.%d.%lld", thread_idx,
                                time(nullptr)));
        for (;;) {
          {
            MutexLock ml(&m);
            if (should_die) return;
          }

          Timer prep_timer;
          quat4 outer_rot = RandomQuaternion(&rc);
          const frame3 outer_frame = yocto::rotation_frame(outer_rot);
          Polyhedron outer = Rotate(polyhedron, outer_frame);
          Mesh2D souter = Shadow(outer);

          const std::vector<int> shadow_hull = QuickHull(souter.vertices);

          // Starting orientation/position.
          const quat4 inner_rot = RandomQuaternion(&rc);

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

          // TODO: To Rendering?
          auto WriteImage = [&](const std::string &filename,
                                const std::array<double, D> &args) {
              // Show:
              Rendering rendering(polyhedron, 3840, 2160);

              rendering.RenderMesh(souter);
              rendering.DarkenBG();

              auto inner_frame = InnerFrame(args);
              Polyhedron inner = Rotate(polyhedron, inner_frame);
              Mesh2D sinner = Shadow(inner);
              rendering.RenderMesh(sinner);
              rendering.RenderBadPoints(sinner, souter);

              rendering.img.Save(filename);

              status->Printf("Wrote " AGREEN("%s") "\n", filename.c_str());
            };

          auto Parameters = [&](const std::array<double, D> &args,
                                double error) {
              auto inner_frame = InnerFrame(args);
              std::string contents =
                StringPrintf("Error: %.17g\n", error);

              contents += "Outer frame:\n";
              contents += FrameString(outer_frame);
              contents += "\nInner frame:\n";
              contents += FrameString(inner_frame);
              StringAppendF(&contents,
                            "\nTook %lld iters, %lld attempts, %.3f seconds\n",
                            iters.Read(), attempts.Read(), run_timer.Seconds());
              return contents;
            };


          std::function<double(const std::array<double, D> &)> Loss =
            [&polyhedron, &souter, &shadow_hull, &InnerFrame](
                const std::array<double, D> &args) {
              attempts++;
              frame3 frame = InnerFrame(args);
              Polyhedron inner = Rotate(polyhedron, frame);
              Mesh2D sinner = Shadow(inner);

              // Does every vertex in inner fall inside the outer shadow?
              double error = 0.0;
              for (const vec2 &iv : sinner.vertices) {
                if (!InHull(souter, shadow_hull, iv)) {
                  error += DistanceToHull(souter.vertices, shadow_hull, iv);
                }
              }

              return error;
            };

          const std::array<double, D> lb =
            {-0.15, -0.15, -0.15, -0.15, -0.25, -0.25};
          const std::array<double, D> ub =
            {+0.15, +0.15, +0.15, +0.15, +0.25, +0.25};
          const double prep_sec = prep_timer.Seconds();

          Timer opt_timer;
          const auto &[args, error] =
            Opt::Minimize<D>(Loss, lb, ub, 1000, 2);
          const double opt_sec = opt_timer.Seconds();

          if (error == 0.0) {
            MutexLock ml(&m);
            // For easy ones, many threads will solve it at once, and then
            // write over each other's solutions.
            if (should_die && iters.Read() < 1000)
              return;
            should_die = true;

            status->Printf("Solved! %lld iters, %lld attempts, in %s\n",
                           iters.Read(), attempts.Read(),
                           ANSI::Time(run_timer.Seconds()).c_str());

            WriteImage(StringPrintf("solved-%s.png", polyhedron.name), args);

            std::string contents = Parameters(args, error);
            StringAppendF(&contents,
                          "\n%s\n",
                          error_histo.SimpleAsciiString(50).c_str());

            std::string sfile = StringPrintf("solution-%s.txt",
                                             polyhedron.name);
            Util::WriteFile(sfile, contents);
            status->Printf("Wrote " AGREEN("%s") "\n", sfile.c_str());

            SaveSolution(polyhedron,
                         outer_frame,
                         InnerFrame(args),
                         SolutionDB::METHOD_HULL);
            return;
          }

          {
            MutexLock ml(&m);
            prep_time += prep_sec;
            opt_time += opt_sec;
            error_histo.Observe(log(error));
            if (error < best_error) {
              best_error = error;
              if (image_per.ShouldRun()) {
                std::string file_base =
                  StringPrintf("best-%s.%lld", polyhedron.name, iters.Read());
                WriteImage(file_base + ".png", args);
                Util::WriteFile(file_base + ".txt", Parameters(args, error));
              }
            }

            status_per.RunIf([&]() {
                double total_time = prep_time + opt_time;

                int64_t it = iters.Read();
                double ips = it / total_time;

                status->Statusf(
                    "%s\n"
                    "%s " ABLUE("prep") " %s " APURPLE("opt")
                    " (" ABLUE("%.3f%%") " / " APURPLE("%.3f%%") ") "
                    "[" AWHITE("%.3f") "/s]\n"
                    "%s iters, %s attempts; best: %.11g",
                    error_histo.SimpleANSI(HISTO_LINES).c_str(),
                    ANSI::Time(prep_time).c_str(),
                    ANSI::Time(opt_time).c_str(),
                    (100.0 * prep_time) / total_time,
                    (100.0 * opt_time) / total_time,
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

// Try simultaneously optimizing both the shadow and hole. This is
// much slower because we can't frontload precomputation (e.g. of a
// convex hull). But it could be that the perpendicular axis needs to
// be just right in order for it to be solvable; Solve() spends most
// of its time trying different shapes of the hole and only random
// samples for the shadow.
[[maybe_unused]]
static void Solve2(const Polyhedron &polyhedron, StatusBar *status) {
  status->Printf("Solve [method 2] " AWHITE("%s") ":\n",
                 polyhedron.name);

  static constexpr int HISTO_LINES = 32;

  std::mutex m;
  bool should_die = false;
  Timer run_timer;
  Periodically status_per(1.0);
  Periodically image_per(10.0);
  double best_error = 1.0e42;
  AutoHisto error_histo(100000);
  constexpr int NUM_THREADS = 4;

  double prep_time = 0.0, opt_time = 0.0;

  ParallelFan(
      NUM_THREADS,
      [&](int thread_idx) {
        ArcFour rc(StringPrintf("solve.%d.%lld", thread_idx,
                                time(nullptr)));
        for (;;) {
          {
            MutexLock ml(&m);
            if (should_die) return;
          }

          // four params for outer rotation, four params for inner
          // rotation, two for 2d translation of inner.
          static constexpr int D = 10;

          Timer prep_timer;
          const quat4 initial_outer_rot = RandomQuaternion(&rc);
          const quat4 initial_inner_rot = RandomQuaternion(&rc);

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

          auto WriteImage = [&](const std::string &filename,
                                const std::array<double, D> &args) {
              Rendering rendering(polyhedron, 3840, 2160);

              auto outer_frame = OuterFrame(args);
              auto inner_frame = InnerFrame(args);

              Mesh2D souter = Shadow(Rotate(polyhedron, outer_frame));
              Mesh2D sinner = Shadow(Rotate(polyhedron, inner_frame));

              rendering.RenderMesh(souter);
              rendering.DarkenBG();

              rendering.RenderMesh(sinner);
              rendering.RenderBadPoints(sinner, souter);

              rendering.img.Save(filename);

              status->Printf("Wrote " AGREEN("%s") "\n", filename.c_str());
            };

          auto Parameters = [&](const std::array<double, D> &args,
                                double error) {
              auto outer_frame = OuterFrame(args);
              auto inner_frame = InnerFrame(args);
              std::string contents =
                StringPrintf("Error: %.17g\n", error);

              contents += "Outer frame:\n";
              contents += FrameString(outer_frame);
              contents += "\nInner frame:\n";
              contents += FrameString(inner_frame);
              StringAppendF(&contents,
                            "\nTook %lld iters, %lld attempts, %.3f seconds\n",
                            iters.Read(), attempts.Read(), run_timer.Seconds());
              return contents;
            };

          std::function<double(const std::array<double, D> &)> Loss =
            [&polyhedron, &OuterFrame, &InnerFrame](
                const std::array<double, D> &args) {
              attempts++;
              frame3 outer_frame = OuterFrame(args);
              frame3 inner_frame = InnerFrame(args);
              Mesh2D souter = Shadow(Rotate(polyhedron, outer_frame));
              Mesh2D sinner = Shadow(Rotate(polyhedron, inner_frame));

              // Does every vertex in inner fall inside the outer shadow?
              double error = 0.0;
              for (const vec2 &iv : sinner.vertices) {
                if (!InMesh(souter, iv)) {
                  // slow :(
                  error += DistanceToMesh(souter, iv);
                }
              }

              return error;
            };

          constexpr double Q = 0.15;

          const std::array<double, D> lb =
            {-Q, -Q, -Q, -Q,
             -Q, -Q, -Q, -Q, -0.25, -0.25};
          const std::array<double, D> ub =
            {+Q, +Q, +Q, +Q,
             +Q, +Q, +Q, +Q, +0.25, +0.25};
          const double prep_sec = prep_timer.Seconds();

          Timer opt_timer;
          const auto &[args, error] =
            Opt::Minimize<D>(Loss, lb, ub, 1000, 2);
          const double opt_sec = opt_timer.Seconds();

          if (error == 0.0) {
            MutexLock ml(&m);
            // For easy ones, many threads will solve it at once, and then
            // write over each other's solutions.
            if (should_die && iters.Read() < 1000)
              return;
            should_die = true;

            status->Printf("Solved! %lld iters, %lld attempts, in %s\n",
                           iters.Read(),
                           attempts.Read(),
                           ANSI::Time(run_timer.Seconds()).c_str());

            WriteImage(StringPrintf("solved-%s.png", polyhedron.name), args);

            std::string contents = Parameters(args, error);
            StringAppendF(&contents,
                          "\n%s\n",
                          error_histo.SimpleAsciiString(50).c_str());

            std::string sfile = StringPrintf("solution-%s.txt",
                                             polyhedron.name);
            Util::WriteFile(sfile, contents);
            status->Printf("Wrote " AGREEN("%s") "\n", sfile.c_str());

            SaveSolution(polyhedron,
                         OuterFrame(args),
                         InnerFrame(args),
                         SolutionDB::METHOD_SIMUL);

            return;
          }

          {
            MutexLock ml(&m);
            prep_time += prep_sec;
            opt_time += opt_sec;
            error_histo.Observe(log(error));
            if (error < best_error) {
              best_error = error;
              if (image_per.ShouldRun()) {
                std::string file_base =
                  StringPrintf("best2-%s.%lld", polyhedron.name, iters.Read());
                WriteImage(file_base + ".png", args);
                Util::WriteFile(file_base + ".txt", Parameters(args, error));
              }
            }

            status_per.RunIf([&]() {
                double total_time = prep_time + opt_time;

                int64_t it = iters.Read();
                double ips = it / total_time;

                status->Statusf(
                    "%s\n"
                    "%s " ABLUE("prep") " %s " APURPLE("opt")
                    " (" ABLUE("%.3f%%") " / " APURPLE("%.3f%%") ") "
                    "[" AWHITE("%.3f") "/s]\n"
                    "%s iters, %s attempts; best: %.11g",
                    error_histo.SimpleANSI(HISTO_LINES).c_str(),
                    ANSI::Time(prep_time).c_str(),
                    ANSI::Time(opt_time).c_str(),
                    (100.0 * prep_time) / total_time,
                    (100.0 * opt_time) / total_time,
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


// Third approach: Maximize the area of the outer polygon before
// optimizing the placement of the inner.
[[maybe_unused]]
static void Solve3(const Polyhedron &polyhedron, StatusBar *status) {
  status->Printf("Solve [method 3] " AWHITE("%s") ":\n",
                 polyhedron.name);

  static constexpr int HISTO_LINES = 32;

  std::mutex m;
  bool should_die = false;
  Timer run_timer;
  Periodically status_per(1.0);
  Periodically image_per(10.0);
  double best_error = 1.0e42;
  AutoHisto error_histo(100000);
  constexpr int NUM_THREADS = 4;

  double prep_time = 0.0, opt_time = 0.0;

  ParallelFan(
      NUM_THREADS,
      [&](int thread_idx) {
        ArcFour rc(StringPrintf("solve.%d.%lld", thread_idx,
                                time(nullptr)));
        for (;;) {
          {
            MutexLock ml(&m);
            if (should_die) return;
          }

          Timer prep_timer;
          quat4 outer_rot = RandomQuaternion(&rc);

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
          const quat4 inner_rot = RandomQuaternion(&rc);

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

          auto WriteImage = [&](const std::string &filename,
                                const std::array<double, D> &args) {
              Rendering rendering(polyhedron, 3840, 2160);
              rendering.RenderMesh(souter);
              rendering.DarkenBG();

              auto inner_frame = InnerFrame(args);
              Polyhedron inner = Rotate(polyhedron, inner_frame);
              Mesh2D sinner = Shadow(inner);
              rendering.RenderMesh(sinner);
              rendering.RenderBadPoints(sinner, souter);

              rendering.img.Save(filename);
              status->Printf("Wrote " AGREEN("%s") "\n", filename.c_str());
            };

          auto Parameters = [&](const std::array<double, D> &args,
                                double error) {
              auto inner_frame = InnerFrame(args);
              std::string contents =
                StringPrintf("Error: %.17g\n", error);

              contents += "Outer frame:\n";
              contents += FrameString(outer_frame);
              contents += "\nInner frame:\n";
              contents += FrameString(inner_frame);
              StringAppendF(&contents,
                            "\nTook %lld iters, %lld attempts, %.3f seconds\n",
                            iters.Read(), attempts.Read(), run_timer.Seconds());
              return contents;
            };


          std::function<double(const std::array<double, D> &)> Loss =
            [&polyhedron, &souter, &shadow_hull, &InnerFrame](
                const std::array<double, D> &args) {
              attempts++;
              frame3 frame = InnerFrame(args);
              Polyhedron inner = Rotate(polyhedron, frame);
              Mesh2D sinner = Shadow(inner);

              // Does every vertex in inner fall inside the outer shadow?
              double error = 0.0;
              for (const vec2 &iv : sinner.vertices) {
                if (!InHull(souter, shadow_hull, iv)) {
                  error += DistanceToHull(souter.vertices, shadow_hull, iv);
                }
              }

              return error;
            };

          const std::array<double, D> lb =
            {-0.15, -0.15, -0.15, -0.15, -0.25, -0.25};
          const std::array<double, D> ub =
            {+0.15, +0.15, +0.15, +0.15, +0.25, +0.25};
          const double prep_sec = prep_timer.Seconds();

          Timer opt_timer;
          const auto &[args, error] =
            Opt::Minimize<D>(Loss, lb, ub, 1000, 2);
          const double opt_sec = opt_timer.Seconds();

          if (error == 0.0) {
            MutexLock ml(&m);
            // For easy ones, many threads will solve it at once, and then
            // write over each other's solutions.
            if (should_die && iters.Read() < 1000)
              return;
            should_die = true;

            status->Printf("Solved! %lld iters, %lld attempts, in %s\n",
                           iters.Read(),
                           attempts.Read(),
                           ANSI::Time(run_timer.Seconds()).c_str());

            WriteImage(StringPrintf("solved-%s.png", polyhedron.name), args);

            std::string contents = Parameters(args, error);
            StringAppendF(&contents,
                          "\n%s\n",
                          error_histo.SimpleAsciiString(50).c_str());

            std::string sfile = StringPrintf("solution-%s.txt",
                                             polyhedron.name);
            Util::WriteFile(sfile, contents);
            status->Printf("Wrote " AGREEN("%s") "\n", sfile.c_str());

            SaveSolution(polyhedron,
                         outer_frame,
                         InnerFrame(args),
                         SolutionDB::METHOD_MAX);

            return;
          }

          {
            MutexLock ml(&m);
            prep_time += prep_sec;
            opt_time += opt_sec;
            error_histo.Observe(log(error));
            if (error < best_error) {
              best_error = error;
              if (image_per.ShouldRun()) {
                std::string file_base =
                  StringPrintf("best3-%s.%lld", polyhedron.name, iters.Read());
                WriteImage(file_base + ".png", args);
                Util::WriteFile(file_base + ".txt", Parameters(args, error));
              }
            }

            status_per.RunIf([&]() {
                double total_time = prep_time + opt_time;

                int64_t it = iters.Read();
                double ips = it / total_time;

                status->Statusf(
                    "%s\n"
                    "%s " ABLUE("prep") " %s " APURPLE("opt")
                    " (" ABLUE("%.3f%%") " / " APURPLE("%.3f%%") ") "
                    "[" AWHITE("%.3f") "/s]\n"
                    "%s iters, %s attempts; best: %.11g",
                    error_histo.SimpleANSI(HISTO_LINES).c_str(),
                    ANSI::Time(prep_time).c_str(),
                    ANSI::Time(opt_time).c_str(),
                    (100.0 * prep_time) / total_time,
                    (100.0 * opt_time) / total_time,
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
[[maybe_unused]]
static void Solve4(const Polyhedron &polyhedron, StatusBar *status) {
  status->Printf("Solve [method 4] " AWHITE("%s") ":\n",
                 polyhedron.name);

  static constexpr int HISTO_LINES = 32;

  std::mutex m;
  bool should_die = false;
  Timer run_timer;
  Periodically status_per(1.0);
  Periodically image_per(10.0);
  double best_error = 1.0e42;
  AutoHisto error_histo(100000);
  constexpr int NUM_THREADS = 4;

  double prep_time = 0.0, opt_time = 0.0;

  ParallelFan(
      NUM_THREADS,
      [&](int thread_idx) {
        ArcFour rc(StringPrintf("solve.%d.%lld", thread_idx,
                                time(nullptr)));
        for (;;) {
          {
            MutexLock ml(&m);
            if (should_die) return;
          }

          // four params for outer rotation, one param for
          // inner rotation (around z axis), two for 2d translation of inner.
          static constexpr int D = 6;

          Timer prep_timer;
          const quat4 initial_outer_rot = RandomQuaternion(&rc);

          // Get two face indices that are not parallel.
          const auto &[face1, face2] = TwoNonParallelFaces(&rc, polyhedron);

          /*
          const quat4 initial_inner_rot = AlignFaceNormalWithX(
              polyhedron.vertices,
              polyhedron.faces->v[face1]);
          */
          const quat4 initial_inner_rot = MakeTwoFacesParallelToZ(
              polyhedron.vertices,
              polyhedron.faces->v[face1],
              polyhedron.faces->v[face2]);

          const frame3 initial_inner_frame =
            yocto::rotation_frame(initial_inner_rot);


          const Mesh2D sinner = Shadow(Rotate(polyhedron, initial_inner_frame));
          const std::vector<vec2> inner_hull_pts = [&]() {
              // status->Printf("[%d] Get convex hull.\n", thread_idx);
              const std::vector<int> inner_hull = QuickHull(sinner.vertices);
              // status->Printf("[%d] Done: Size %d.\n",
              //            thread_idx, inner_hull.size());
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

          // PERF: The inner polyhedron is not rotated, so we should
          // just compute the convex hull once. We could equivalently
          // just be applying the translation to the outer polyhedron.
          auto InnerFrame = [&initial_inner_frame](
              const std::array<double, D> &args) -> const frame3 & {
              return initial_inner_frame;
            };

          auto WriteImage = [&](const std::string &filename,
                                const std::array<double, D> &args) {
              Rendering rendering(polyhedron, 3840, 2160);

              auto outer_frame = OuterFrame(args);
              auto inner_frame = InnerFrame(args);

              Mesh2D souter = Shadow(Rotate(polyhedron, outer_frame));
              Mesh2D sinner = Shadow(Rotate(polyhedron, inner_frame));

              rendering.RenderMesh(souter);
              rendering.DarkenBG();

              rendering.RenderMesh(sinner);
              std::vector<int> hull = QuickHull(sinner.vertices);
              status->Printf("Hull points: %d\n", (int)hull.size());
              rendering.RenderHull(sinner, hull, 0x000000AA);
              rendering.RenderBadPoints(sinner, souter);
              rendering.img.Save(filename);

              if (hull.size() < 3) {
                printf("\n\n\n\n");
                for (const vec2 &v : sinner.vertices) {
                  printf("vec2{%.17g, %.17g},\n",
                         v.x, v.y);
                }
                LOG(FATAL) << "INVALID HULL!";
              }


              status->Printf("Wrote " AGREEN("%s") "\n", filename.c_str());
            };

          auto Parameters = [&](const std::array<double, D> &args,
                                double error) {
              auto outer_frame = OuterFrame(args);
              auto inner_frame = InnerFrame(args);
              std::string contents =
                StringPrintf("Error: %.17g\n", error);

              contents += "Outer frame:\n";
              contents += FrameString(outer_frame);
              contents += "\nInner frame:\n";
              contents += FrameString(inner_frame);
              StringAppendF(&contents,
                            "\nTook %lld iters, %lld attempts, %.3f seconds\n",
                            iters.Read(), attempts.Read(), run_timer.Seconds());
              return contents;
            };

          std::function<double(const std::array<double, D> &)> Loss =
            [&polyhedron, &OuterFrame, &inner_hull_pts](
                const std::array<double, D> &args) {
              attempts++;
              frame3 outer_frame = OuterFrame(args);
              Mesh2D souter = Shadow(Rotate(polyhedron, outer_frame));

              // Does every vertex in inner fall inside the outer shadow?
              double error = 0.0;
              for (const vec2 &iv : inner_hull_pts) {
                if (!InMesh(souter, iv)) {
                  // slow :(
                  error += DistanceToMesh(souter, iv);
                }
              }

              return error;
            };

          constexpr double Q = 0.25;

          const std::array<double, D> lb = {-Q, -Q, -Q, -Q, -0.5, -0.5};
          const std::array<double, D> ub = {+Q, +Q, +Q, +Q, +0.5, +0.5};
          const double prep_sec = prep_timer.Seconds();

          Timer opt_timer;
          const auto &[args, error] =
            Opt::Minimize<D>(Loss, lb, ub, 1000, 2);
          const double opt_sec = opt_timer.Seconds();

          if (error == 0.0) {
            MutexLock ml(&m);
            // For easy ones, many threads will solve it at once, and then
            // write over each other's solutions.
            if (should_die && iters.Read() < 1000)
              return;
            should_die = true;

            status->Printf("Solved! %lld iters, %lld attempts, in %s\n",
                          iters.Read(),
                          attempts.Read(),
                          ANSI::Time(run_timer.Seconds()).c_str());

            WriteImage(StringPrintf("solved-%s.png", polyhedron.name), args);

            std::string contents = Parameters(args, error);
            StringAppendF(&contents,
                          "\n%s\n",
                          error_histo.SimpleAsciiString(50).c_str());

            std::string sfile = StringPrintf("solution-%s.txt",
                                             polyhedron.name);
            Util::WriteFile(sfile, contents);
            status->Printf("Wrote " AGREEN("%s") "\n", sfile.c_str());

            SaveSolution(polyhedron,
                         OuterFrame(args),
                         InnerFrame(args),
                         SolutionDB::METHOD_PARALLEL);

            return;
          }

          {
            MutexLock ml(&m);
            prep_time += prep_sec;
            opt_time += opt_sec;
            error_histo.Observe(log(error));
            if (error < best_error) {
              best_error = error;
              if (image_per.ShouldRun()) {
                std::string file_base =
                  StringPrintf("best4-%s.%lld", polyhedron.name, iters.Read());
                WriteImage(file_base + ".png", args);
                Util::WriteFile(file_base + ".txt", Parameters(args, error));
              }
            }

            status_per.RunIf([&]() {
                double total_time = prep_time + opt_time;

                int64_t it = iters.Read();
                double ips = it / total_time;

                status->Statusf(
                    "%s\n"
                    "%s " ABLUE("prep") " %s " APURPLE("opt")
                    " (" ABLUE("%.3f%%") " / " APURPLE("%.3f%%") ") "
                    "[" AWHITE("%.3f") "/s]\n"
                    "%s iters, %s attempts; best: %.11g",
                    error_histo.SimpleANSI(HISTO_LINES).c_str(),
                    ANSI::Time(prep_time).c_str(),
                    ANSI::Time(opt_time).c_str(),
                    (100.0 * prep_time) / total_time,
                    (100.0 * opt_time) / total_time,
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

static void ReproduceEasySolutions(int method) {
  SolutionDB db;
  std::vector<SolutionDB::Solution> sols = db.GetAllSolutions();
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
        status.Printf("Solve " AYELLOW("%s") " with " AWHITE("%s") "...\n",
                      poly.name, SolutionDB::MethodName(method));

        switch (method) {
        case SolutionDB::METHOD_HULL:
          return Solve(poly, &status);
        case SolutionDB::METHOD_SIMUL:
          return Solve2(poly, &status);
        case SolutionDB::METHOD_MAX:
          return Solve3(poly, &status);
        case SolutionDB::METHOD_PARALLEL:
          return Solve4(poly, &status);
        default:
          LOG(FATAL) << "Method not available";
        }
      }
    };

  // Platonic
  MaybeSolve(Tetrahedron());
  MaybeSolve(Cube());
  MaybeSolve(Dodecahedron());
  MaybeSolve(Icosahedron());
  MaybeSolve(Octahedron());

  // Archimedean
  MaybeSolve(Cuboctahedron());
  MaybeSolve(TruncatedOctahedron());
  MaybeSolve(TruncatedCube());
  MaybeSolve(Rhombicuboctahedron());
  MaybeSolve(Icosidodecahedron());
  MaybeSolve(TruncatedIcosahedron());
  MaybeSolve(TruncatedDodecahedron());
  MaybeSolve(TruncatedTetrahedron());
  MaybeSolve(TruncatedIcosidodecahedron());

  // Catalan
  MaybeSolve(PentagonalIcositetrahedron());
  MaybeSolve(RhombicTriacontahedron());
}

int main(int argc, char **argv) {
  ANSI::Init();
  printf("\n");

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

  Polyhedron target = RhombicTriacontahedron();

  // These generate visualizations of the polyhedron;
  // they are unrelated to solving
  if (true) {
    // AnimateHull();
    Visualize(target);
    AnimateMesh(target);
  }

  if (false) {
    ReproduceEasySolutions(SolutionDB::METHOD_SIMUL);

  } else {
    // Call one of the solution procedures:

    StatusBar status(3 + HISTO_LINES);
    // Solve(target);
    Solve2(target, &status);
    // Solve3(target);
    // Solve4(target);
  }

  printf("OK\n");
  return 0;
}
