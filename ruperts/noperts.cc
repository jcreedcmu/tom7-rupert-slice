#include <array>
#include <cstdio>
#include <ctime>
#include <functional>
#include <limits>
#include <optional>

#include "timer.h"
#include "ansi.h"
#include "auto-histo.h"
#include "base/stringprintf.h"
#include "opt/opt.h"
#include "arcfour.h"
#include "periodically.h"
#include "randutil.h"
#include "polyhedra.h"
#include "status-bar.h"
#include "yocto_matht.h"
#include "atomic-util.h"

// Try to find counterexamples.
// TODO: Expand beyond tetrahedra.

DECLARE_COUNTERS(tetrahedra, attempts, degenerate, u1_, u2_, u3_, u4_, u5_);


using vec2 = yocto::vec<double, 2>;
using vec3 = yocto::vec<double, 3>;
using vec4 = yocto::vec<double, 4>;
using mat4 = yocto::mat<double, 4>;
using quat4 = yocto::quat<double, 4>;
using frame3 = yocto::frame<double, 3>;

static vec3 RandomVec(ArcFour *rc) {
  return vec3(2.0 * RandDouble(rc) - 1.0,
              2.0 * RandDouble(rc) - 1.0,
              2.0 * RandDouble(rc) - 1.0);
}

// Return the number of iterations taken, or nullopt if we exceeded
// the limit.
static constexpr int MAX_ITERS = 10000;
static std::optional<int> TrySolve(ArcFour *rc, const Polyhedron &poly) {
  for (int iter = 0; iter < MAX_ITERS; iter++) {

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
        Mesh2D souter = Shadow(Rotate(poly, outer_frame));
        Mesh2D sinner = Shadow(Rotate(poly, inner_frame));

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
       -Q, -Q, -Q, -Q, -1.0, -1.0};
    const std::array<double, D> ub =
      {+Q, +Q, +Q, +Q,
       +Q, +Q, +Q, +Q, +1.0, +1.0};

    const auto &[args, error] =
      Opt::Minimize<D>(Loss, lb, ub, 1000, 2);

    if (error == 0.0) {
      return {iter};
    }
  }

  return std::nullopt;
}

static void Nopert() {
  static constexpr int HISTO_LINES = 10;
  ArcFour rc(StringPrintf("noperts.%lld\n", time(nullptr)));
  Timer timer;
  AutoHisto histo(10000);
  StatusBar status(HISTO_LINES + 2);
  Periodically status_per(5.0);
  Polyhedron tetra = Tetrahedron();
  for (;;) {
    tetrahedra++;
    // TODO: We could maybe frame this as an optimization problem
    // (an adversarial one). But to start, we just generate
    // random tetrahedra.
    vec3 v0 = RandomVec(&rc);
    vec3 v1 = RandomVec(&rc);
    vec3 v2 = RandomVec(&rc);
    vec3 v3 = RandomVec(&rc);
    if (TriangleIsDegenerate(v0, v1, v2) ||
        TriangleIsDegenerate(v0, v1, v3) ||
        TriangleIsDegenerate(v0, v2, v3) ||
        TriangleIsDegenerate(v1, v2, v3)) {
      degenerate++;
      continue;
    }

    // Put centroid at 0,0, since the optimization procedure assumes
    // this (and rotations are taken to be around the origin).
    vec3 centroid = (v0 + v1 + v2 + v3) * 0.25;
    v0 -= centroid;
    v1 -= centroid;
    v2 -= centroid;
    v3 -= centroid;

    // Now we try to solve it.
    tetra.vertices = {v0, v1, v2, v3};
    std::optional<int> iters = TrySolve(&rc, tetra);
    if (iters.has_value()) {
      histo.Observe(iters.value());
    } else {
      printf("\n\n\n\nFailed to solve:\n"
             "%s\n"
             "%s\n"
             "%s\n"
             "%s\n",
             VecString(v0).c_str(),
             VecString(v1).c_str(),
             VecString(v2).c_str(),
             VecString(v3).c_str());
      return;
    }

    if (status_per.ShouldRun()) {
      double total_time = timer.Seconds();
      double tps = tetrahedra.Read() / total_time;
      // TODO: Can use progress bar when there's a timer.
      status.Statusf(
          "%s\n"
          ABLUE("%s") " tetrahedra "
          "in %s "
          "[" ACYAN("%.3f") "/s]\n",
          histo.SimpleANSI(HISTO_LINES).c_str(),
          FormatNum(tetrahedra.Read()).c_str(),
          ANSI::Time(total_time).c_str(),
          tps);
    }
  }
}

int main(int argc, char **argv) {
  ANSI::Init();
  printf("\n");

  Nopert();

  printf("OK\n");
  return 0;
}
