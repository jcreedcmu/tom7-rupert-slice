
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <format>
#include <limits>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "ansi.h"
#include "array-util.h"
#include "atomic-util.h"
#include "base/logging.h"
#include "arcfour.h"
#include "opt/opt.h"
#include "periodically.h"
#include "polyhedra.h"
#include "rendering.h"
#include "status-bar.h"
#include "threadutil.h"
#include "timer.h"
#include "yocto_matht.h"

DECLARE_COUNTERS(iters, attempts);

// A complex is a solid composed of multiple convex solids in
// a rupert configuration. Rupert configuration means that one
// solid can pass cleanly through another. rupert(outer, inner)
//

// Here we want to find some kind of loop. Suppose a sequence
// of complexes:
//  - complex_n passes through outer: rupert(outer, complex_n)
//  - for some parameter dz (along the z axis), we fuse outer
//    and complex_n, forming complex_(n+1).
//  - complex_(n+1) also has the property.
//
// We can certainly do this for a few steps; 1 step is just
// the original rupert problem, and it's pretty easy to get
// infinite sequences (see e.g. threeperts).
//
// Since each time we pass a complex through, we can think of the
// "drill bit" as an extrusion of infinite length, we would have a
// sort of loop if we can fit any complex inside a previous extrusion.
// Although when m > n, vol(complex_m) > vol(complex_n), (?? true ??)
// hulls are not necessarily monotonic since the orientations may
// be different.
//
// So, concretely, we have
//  complex_0 = cube
//  oframe0, iframe0, orientations that allow passing complex_0 through cube,
//                    (and that set the z pos of complex_0)
//  hole_0, the polygonal shape of the shape in the xy plane,
//          with z projection axis
//  complex_1 = transform(oframe0, complex_0) U transform(oframe1, cube)
//  oframe1, iframe1, orientations that allow passing complex_1 through cube,
//
//  and finally,
//  cframe, such that shadow(transform(complex_1)) fits inside hole_0.
// Is this a loop? This means I can pass the complex through a previous
// hole, but if I then unioned it, I would get a different shape than
// I had before, and it wouldn't necessarily be possible to continue.
//
// Is it really true that vol(complex_m) > vol(complex_n)? When I union
// two shapes it is of course monotonic, but what I'm actually doing
// here is carving (first removing material along z axis) and then
// unioning. The result has to have more volume than a cube, but it's
// not obvious that it grows this way. (Yes: It's actually possible for the
// volume to go down!)
//
// So is it interesting to look for a loop in the complexes?
// You


struct Meperts {
  static constexpr int NUM_THREADS = 1;
  static constexpr int STATUS_LINES = 2;

  Timer timer;
  Polyhedron poly;
  std::mutex m;
  bool should_die = false;
  StatusBar status = StatusBar(NUM_THREADS + STATUS_LINES);
  Periodically status_per = Periodically(5);
  int64_t time_limit = 60 * 60;

  Meperts(const Polyhedron &poly) : poly(poly) {
    iters.Reset();
  }

  void Solve() {
    std::vector<std::thread> threads;
    for (int i = 0; i < NUM_THREADS; i++) {
      threads.emplace_back(&Meperts::RunThread, this, i);
    }

    for (std::thread &t : threads) {
      t.join();
    }
  }

  void RunThread(int thread_idx) {
    ArcFour rc(std::format("thread.{}.{}", thread_idx, time(nullptr)));
    int64_t local_iters = 0;
    double best = std::numeric_limits<double>::infinity();
    Periodically line_status_per(4.1);
    for (;;) {
      {
        MutexLock ml(&m);
        if (should_die) return;
        if (timer.Seconds() > time_limit) {
          should_die = true;
          return;
        }
      }

      const quat4 outer_rot = RandomQuaternion(&rc);
      const quat4 middle_rot = RandomQuaternion(&rc);
      // const quat4 inner_rot = RandomQuaternion(&rc);
      const quat4 inner_rot{0.0, 0.0, 0.0, 1.0};

      // Simple joint optimization. Each shape is moved independently
      // (but it might be cleaner to couple inner and middle positions?)
      // Note that in this version, projection always happens along the
      // z axis.
      //
      // Parameters:
      //  Outer: 4 orientation
      //  Middle: 4 orientation + 2 xy pos
      //  Inner: /* 4 orientation */ + 2 xy pos

      static constexpr int D = 4 + (4 + 2) + (/* 4 + */ 2);
      static constexpr int OUTER_ARGS = 0;
      static constexpr int MIDDLE_ARGS = 4;
      static constexpr int INNER_ARGS = MIDDLE_ARGS + 6;

      auto OuterFrame = [&outer_rot](const std::array<double, D> &args) {
          const auto &[di, dj, dk, dl] = SubArray<OUTER_ARGS, 4>(args);
          quat4 tweaked_rot = normalize(quat4{
              .x = outer_rot.x + di,
              .y = outer_rot.y + dj,
              .z = outer_rot.z + dk,
              .w = outer_rot.w + dl,
            });
          return yocto::rotation_frame(tweaked_rot);
        };

      auto MiddleFrame = [&middle_rot](const std::array<double, D> &args) {
          const auto &[di, dj, dk, dl, dx, dy] = SubArray<MIDDLE_ARGS, 6>(args);
          quat4 tweaked_rot = normalize(quat4{
              .x = middle_rot.x + di,
              .y = middle_rot.y + dj,
              .z = middle_rot.z + dk,
              .w = middle_rot.w + dl,
            });
          frame3 rotate = yocto::rotation_frame(tweaked_rot);
          frame3 translate = yocto::translation_frame(
              vec3{.x = dx, .y = dy, .z = 0.0});
          // PERF: We can avoid multiplication here; the rotation frame has
          // identity offset and the translation frame has identity matrix.
          return translate * rotate;
        };

      auto InnerFrame = [&inner_rot](const std::array<double, D> &args) {
          const auto &[/* di, dj, dk, dl, */ dx, dy] = SubArray<INNER_ARGS, 2>(args);
          /*
          quat4 tweaked_rot = normalize(quat4{
              .x = inner_rot.x + di,
              .y = inner_rot.y + dj,
              .z = inner_rot.z + dk,
              .w = inner_rot.w + dl,
            });
          */
          quat4 tweaked_rot = inner_rot;
          frame3 rotate = yocto::rotation_frame(tweaked_rot);
          frame3 translate = yocto::translation_frame(
              vec3{.x = dx, .y = dy, .z = 0.0});
          // PERF: We can avoid multiplication here; the rotation frame has
          // identity offset and the translation frame has identity matrix.
          return translate * rotate;
        };

      auto LossParts = [&](const std::array<double, D> &args) ->
        std::pair<double, double> {
          frame3 outer_frame = OuterFrame(args);
          frame3 middle_frame = MiddleFrame(args);
          frame3 inner_frame = InnerFrame(args);

          // inner must fit in middle, as usual. We could just call
          // LossFunction, but we don't want to keep computing these things.
          Mesh2D souter = Shadow(Rotate(poly, outer_frame));
          Mesh2D smiddle = Shadow(Rotate(poly, middle_frame));
          Mesh2D sinner = Shadow(Rotate(poly, inner_frame));

          // Although computing the convex hull is expensive, the tests
          // below are O(n*m), so it is helpful to significantly reduce
          // one of the factors.
          const std::vector<int> outer_hull = GrahamScan(souter.vertices);
          if (outer_hull.size() < 3) {
            // If the outer hull is degenerate, then the inner hull
            // cannot be strictly within it. We don't have a good
            // way to measure the gradient here, though.
            return {1'000'000.0, 1'000'000.0};
          }

          // Same for inner/middle pair.
          const std::vector<int> middle_hull = GrahamScan(smiddle.vertices);
          if (middle_hull.size() < 3) {
            return {1'000'000.0, 1'000'000.0};
          }

          const std::vector<int> inner_hull = GrahamScan(sinner.vertices);
          // This would "fit", but we never expect this to be
          // degenerate. Could be something like a 0 quaternion.
          // Reject such situations as well.
          if (inner_hull.size() < 3) {
            return {1'000'000.0, 1'000'000.0};
          }

          HullInscribedCircle outer_circle(souter.vertices, outer_hull);
          HullInscribedCircle middle_circle(smiddle.vertices, middle_hull);

          // Use the same error function for inner/middle and middle/outer.
          auto ErrorPair = [](const HullInscribedCircle &outer_circle,
                              const std::vector<vec2> &outer_vertices,
                              const std::vector<vec2> &inner_vertices,
                              const std::vector<int> &outer_hull,
                              const std::vector<int> &inner_hull) {
              double error = 0.0;
              int errors = 0;
              for (int i : inner_hull) {
                const vec2 &iv = inner_vertices[i];
                if (false && outer_circle.DefinitelyInside(iv))
                  continue;

                if (!PointInPolygon(iv, outer_vertices, outer_hull)) {
                  error += DistanceToHull(outer_vertices, outer_hull, iv);
                  errors++;
                }
              }

              if (errors > 0) {
                if (error == 0.0) [[unlikely]] {
                  // If they are not in the mesh, don't return an actual zero.
                  return std::numeric_limits<double>::min() * errors;
                }

                CHECK(error > 0.0);
                return error;

              } else {

                // If successful, maximize clearance.
                double c = HullClearance(outer_vertices, outer_hull,
                                         inner_vertices, inner_hull);
                CHECK(c >= 0.0) << c;
                return -c;
              }
            };

          // inner must fit in middle
          double err1 = ErrorPair(middle_circle,
                                  smiddle.vertices,
                                  sinner.vertices,
                                  middle_hull,
                                  inner_hull);

          // middle must fit in outer
          double err2 = ErrorPair(outer_circle,
                                  souter.vertices,
                                  smiddle.vertices,
                                  outer_hull,
                                  middle_hull);

          return std::make_pair(err1, err2);
        };

      auto Loss = [&](const std::array<double, D> &args) {
          attempts++;

          const auto &[err1, err2] = LossParts(args);

          // Make sure we only return a negative error if the problem
          // is truly solved.

          // When both are solved, we to maximize the minimal clearance.
          if (err1 <= 0.0 && err2 <= 0.0) return std::max(err1, err2);
          // When unsolved, all errors are cumulative.
          else if (err1 >= 0.0 && err2 >= 0.0) return err1 + err2;
          // This could provide more of a gradient, but it is at least valid.
          else return std::max(err1, err2);
        };

      // Optimize.

      constexpr double Q = 0.25;

      // Could distinguish the positional parameters from the rotation
      // ones...
      std::array<double, D> lb;
      for (int i = 0; i < D; i++) lb[i] = -Q;

      std::array<double, D> ub;
      for (int i = 0; i < D; i++) ub[i] = +Q;

      Timer opt_timer;
      const auto &[args, error] = Opt::Minimize<D>(Loss, lb, ub, 1000, 2, 100);
      [[maybe_unused]] const double opt_sec = opt_timer.Seconds();
      iters++;

      best = std::min(best, error);

      line_status_per.RunIf([&]() {
          std::string line = std::format("{}. {} iters, " ABLUE("{:.6g}") " best",
                                         thread_idx,
                                         local_iters,
                                         best);
          status.EmitLine(thread_idx, line);
        });

      if (error <= 0.0) {
        MutexLock ml(&m);
        // Only succeed once.
        if (should_die) return;

        status.Printf(AGREEN("Success") "!\n");
        should_die = true;

        const auto &[err1, err2] = LossParts(args);

        printf("Err1: %.11g\n"
               "Err2: %.11g\n", err1, err2);

        frame3 outer_frame = OuterFrame(args);
        frame3 middle_frame = MiddleFrame(args);
        frame3 inner_frame = InnerFrame(args);

        printf("Outer:\n%s\n"
               "Middle:\n%s\n"
               "Inner:\n%s\n",
               FrameString(outer_frame).c_str(),
               FrameString(middle_frame).c_str(),
               FrameString(inner_frame).c_str());

        Rendering render(poly, 1920 * 2, 1080 * 2);


        Mesh2D souter = Shadow(Rotate(poly, outer_frame));
        Mesh2D smiddle = Shadow(Rotate(poly, middle_frame));
        Mesh2D sinner = Shadow(Rotate(poly, inner_frame));

        render.RenderMesh(souter);
        render.DarkenBG();
        render.RenderMesh(smiddle);
        render.DarkenBG();
        render.RenderMesh(sinner);

        render.Save(std::format("meperts-{}.png", poly.name));

        return;
      }

      {
        MutexLock ml(&m);
        MaybeStatus();
      }
    }
  }

  // Must hold lock.
  void MaybeStatus() {
    status_per.RunIf([&]{
        double total_time = timer.Seconds();
        int64_t it = iters.Read();
        double ips = it / total_time;

        int64_t end_sec = time_limit;

        std::string bar =
          ANSI::ProgressBar(
              (int64_t)total_time, end_sec,
              std::format(APURPLE("{}") " | " ACYAN("{}"),
                          poly.name, "MEPERTS"),
              total_time);

        status.EmitLine(NUM_THREADS + 0, bar.c_str());
        status.LineStatusf(
            NUM_THREADS + 0,
            "%s iters, %s attempts; "
            " [" ACYAN("%.3f") "/s]\n",
            FormatNum(it).c_str(),
            FormatNum(attempts.Read()).c_str(),
            ips);
      });
  }
};

int main(int argc, char **argv) {
  ANSI::Init();

  Polyhedron target = Cube();
  Meperts meperts(target);
  meperts.Solve();

  delete target.faces;

  return 0;
}
