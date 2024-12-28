#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <functional>
#include <limits>
#include <mutex>
#include <numbers>
#include <optional>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "ansi.h"
#include "arcfour.h"
#include "atomic-util.h"
#include "auto-histo.h"
#include "base/stringprintf.h"
#include "hull.h"
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

// Try to find counterexamples.
//
// TODO: Generate symmetric polyhedra.

DECLARE_COUNTERS(polyhedra, attempts, degenerate, u1_, u2_, u3_, u4_, u5_);


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
  CHECK(!poly.faces->v.empty());

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

[[maybe_unused]]
static Polyhedron RandomTetrahedron(ArcFour *rc) {
  Polyhedron tetra = Tetrahedron();
  for (;;) {
    // TODO: We could maybe frame this as an optimization problem
    // (an adversarial one). But to start, we just generate
    // random tetrahedra.
    vec3 v0 = RandomVec(rc);
    vec3 v1 = RandomVec(rc);
    vec3 v2 = RandomVec(rc);
    vec3 v3 = RandomVec(rc);
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

    return tetra;
  }
}

Polyhedron RandomPolyhedron(ArcFour *rc, int num_points) {
  for (;;) {
    std::vector<vec3> pts;
    pts.reserve(num_points);
    for (int i = 0; i < num_points; i++) {
      pts.push_back(RandomVec(rc));
    }

    // Just the set of points.
    std::unordered_set<int> hull_pts;
    for (const auto &[a, b, c] : QuickHull3D::Hull(pts)) {
      hull_pts.insert(a);
      hull_pts.insert(b);
      hull_pts.insert(c);
    }

    vec3 centroid(0.0, 0.0, 0.0);
    std::vector<vec3> pts2;
    pts2.reserve(hull_pts.size());
    for (int i : hull_pts) {
      centroid += pts[i];
      pts2.push_back(pts[i]);
    }
    centroid /= hull_pts.size();
    for (vec3 &pt : pts2) {
      pt -= centroid;
    }

    std::optional<Polyhedron> poly =
      ConvexPolyhedronFromVertices(std::move(pts2), "random");
    if (poly.has_value()) {
      return std::move(poly.value());
    } else {
      degenerate++;
    }
  }
}

// All of the vertices on the unit sphere.
static Polyhedron RandomCyclicPolyhedron(ArcFour *rc, int num_points) {
  for (;;) {
    std::vector<vec3> pts;
    pts.reserve(num_points);
    for (int i = 0; i < num_points; i++) {
      vec3 v = normalize(RandomVec(rc));
      pts.push_back(v);
    }

    std::optional<Polyhedron> poly =
      ConvexPolyhedronFromVertices(std::move(pts), "randomcyclic");
    if (poly.has_value()) {
      return std::move(poly.value());
    } else {
      degenerate++;
    }
  }
}

// The platonic solids are a nice way to get to their corresponding
// symmetry groups. We normalize them and convert each vertex to
// a rotation frame.
struct SymmetryGroups {
  std::vector<frame3> tetrahedron;
  std::vector<frame3> octahedron;
  std::vector<frame3> icosahedron;
  SymmetryGroups() {
    tetrahedron = MakeFrames(Tetrahedron());
    octahedron = MakeFrames(Octahedron());
    icosahedron = MakeFrames(Icosahedron());
  }

 private:
  // For normalized vectors a and b (interpreted as orientations on
  // the sphere), compute the rotation from a to b.
  quat4 RotationFromAToB(const vec3 &a, const vec3 &b) {
    vec3 norma = normalize(a);
    vec3 normb = normalize(b);
    double d = dot(norma, normb);
    vec3 axis = cross(norma, normb);
    if (length_squared(axis) < 1e-10) {
      if (d > 0) {
        return quat4{0, 0, 0, 1};
      } else {
        // Rotate around any perpendicular axis.
        vec3 perp_axis = orthogonal(norma);
        return QuatFromVec(yocto::rotation_quat(perp_axis, std::numbers::pi));
      }
    }

    double angle = std::acos(std::clamp(d, -1.0, 1.0));
    return QuatFromVec(yocto::rotation_quat(axis, angle));

    // TODO: We should be able to do this without the special cases?
#if 0
    double d = dot(a, b);
    vec3 axis = cross(a, b);

    double s = sqrt((1.0 + d) * 2.0);
    double inv_s = 1.0 / s;
    return normalize(quat4(axis.x * inv_s, axis.y * inv_s, axis.z * inv_s,
                           s * 0.5));
#endif
  }

  std::vector<frame3> MakeFrames(Polyhedron &&poly) {
    std::vector<frame3> rots;
    rots.reserve(poly.vertices.size());

    const vec3 z_axis(0, 0, 1);
    for (const vec3 &v : poly.vertices) {
      quat4 q = RotationFromAToB(z_axis, normalize(v));
      rots.push_back(yocto::rotation_frame(q));
    }

    delete poly.faces;
    return rots;
  }
};

// Note that we can get more (because we require a minimum of
// two random points so that they aren't just the platonic solids)
// or fewer (because we deduplicate at the end) than NUM_POINTS,
// but we try to get close.
static Polyhedron RandomSymmetricPolyhedron(ArcFour *rc, int num_points) {
  static SymmetryGroups *symmetry = new SymmetryGroups;

  auto MakePoints = [rc](int num) {
      printf("MakePoints %d\n", num);
      std::vector<vec3> pts;
      pts.reserve(num);
      for (int i = 0; i < num; i++) {
        vec3 v = normalize(RandomVec(rc));
        pts.push_back(v);
        printf("  point %s\n", VecString(v).c_str());
      }
      return pts;
  };

  for (;;) {

    // Depending on the symmetry chosen, we'll need fewer
    // random points.
    int target_points = num_points;

    const bool include_reflection = false && rc->Byte() & 1;
    if (include_reflection)
      target_points = std::max(target_points / 2, 1);

    const char *method = "error";
    std::vector<vec3> points;
    // Select symmetry groups if we have enough points and if a biased
    // coin flip comes in our favor. There's no point in doing it
    // unless we'll generate at least two seed points, since otherwise
    // we just get a rotated version of that polyhedron.
    auto UsePolyhedralGroup = [&](
        const char *what,
        const std::vector<frame3> &group, int chance) -> bool {
        if (target_points >= group.size() * 2 &&
            RandTo(rc, chance) == 0) {
          target_points = std::max(target_points / (int)group.size(), 2);
          // const std::vector<vec3> r = MakePoints(target_points);
          const std::vector<vec3> r = {normalize(vec3(1.0, 0.0, 0.0))};
          for (const vec3 &v : r) {
            for (const frame3 &rot : group) {
              printf("Rotate %s by %s\n",
                     VecString(v).c_str(),
                     FrameString(rot).c_str());
              points.push_back(yocto::transform_point(rot, v));
            }
          }
          method = what;
          return true;
        }
        return false;
      };

    if (UsePolyhedralGroup("icosahedron", symmetry->icosahedron, 3)) {
      // nothing
    } else if (UsePolyhedralGroup("octahedron", symmetry->octahedron, 3)) {
      // nothing
    } else if (UsePolyhedralGroup("tetrahedron", symmetry->tetrahedron,

                                  // XXX!
                                  1)) {
      // nothing
    } else {
      // Then we can always use the cyclic (or dihedral if reflections
      // are on) group.

      // Pick a nontrivial n, but not too big.
      int n = std::max((int)RandTo(rc, target_points), 2);
      target_points = std::max(target_points / n, 2);
      printf("Cyclic n=" AYELLOW("%d") "\n", n);

      std::vector<vec3> r = MakePoints(target_points);
      for (int i = 0; i < n; ++i) {
        double angle = (2.0 * std::numbers::pi * i) / n;
        // Rotate around the z-axis.
        frame3 rotation_frame = yocto::rotation_frame(vec3{0.0, 0.0, 1.0},
                                                      angle);
        for (const vec3 &pt : r) {
          points.push_back(yocto::transform_point(rotation_frame, pt));
        }
      }
      method = "cyclic";
    }

    if (include_reflection) {
      std::vector<vec3> refl_pts = points;
      for (const vec3 &p : points) {
        refl_pts.emplace_back(p.x, -p.y, p.z);
      }
      points = std::move(refl_pts);
    }

    // Deduplicate points if they are too close. This is particularly
    // important when reflections are included.
    {
      std::vector<vec3> dedup_pts = points;
      for (const vec3 &p : points) {
        for (const vec3 &q : dedup_pts) {
          if (distance_squared(p, q) < 0.0001) {
            goto next;
          }
        }
        dedup_pts.push_back(p);
      next:;
      }

      points = std::move(dedup_pts);
    }

    printf("Target points " AWHITE("%d") " method " ACYAN("%s")
           " refl %s\n",
           target_points, method, include_reflection ? AGREEN("y") :
           AORANGE("n"));

    printf("Result:\n");
    for (const vec3 & v: points) {
      printf("  %s\n", VecString(v).c_str());
    }

    std::optional<Polyhedron> poly =
      ConvexPolyhedronFromVertices(std::move(points), "randomsymmetric");
    if (poly.has_value()) {
      return std::move(poly.value());
    } else {
      printf(AORANGE("degenerate") "\n");
      degenerate++;
    }
  }
}


static void Nopert(double max_seconds, int num_points) {
  polyhedra.Reset();
  attempts.Reset();
  degenerate.Reset();

  static constexpr int NUM_THREADS = 1;
  static constexpr int HISTO_LINES = 10;
  // static constexpr int METHOD = SolutionDB::NOPERT_METHOD_RANDOM;
  static constexpr int METHOD = SolutionDB::NOPERT_METHOD_SYMMETRIC;

  const std::string lower_method = []() {
      std::string name = Util::lcase(SolutionDB::NopertMethodName(METHOD));
      (void)Util::TryStripPrefix("nopert_method_", &name);
      return name;
    }();

  Timer timer;
  AutoHisto histo(10000);
  StatusBar status(HISTO_LINES + 2);
  Periodically status_per(5.0);
  std::mutex m;
  bool should_die = false;

  ParallelFan(
      NUM_THREADS,
      [&](int thread_idx) {
        ArcFour rc(StringPrintf("noperts.%d.%lld\n",
                                thread_idx, time(nullptr)));

        for (;;) {
          {
            MutexLock ml(&m);
            if (should_die) return;

            if (timer.Seconds() > max_seconds) {
              should_die = true;
              status.Printf("Stopping after %s\n",
                            ANSI::Time(timer.Seconds()).c_str());
              SolutionDB db;
              db.AddNopertAttempt(num_points, polyhedra.Read(), histo,
                                  METHOD);
              return;
            }
          }

          polyhedra++;

          Polyhedron poly = [&rc, num_points]() {
              switch (METHOD) {
              case SolutionDB::NOPERT_METHOD_RANDOM:
                return RandomPolyhedron(&rc, num_points);
              case SolutionDB::NOPERT_METHOD_CYCLIC:
                return RandomCyclicPolyhedron(&rc, num_points);
              case SolutionDB::NOPERT_METHOD_SYMMETRIC:
                return RandomSymmetricPolyhedron(&rc, num_points);
              default:
                LOG(FATAL) << "Bad Nopert method";
                return Cube();
              }
            }();

          if constexpr (true) {
            printf("Rendering %d points %d faces...\n",
                   (int)poly.vertices.size(),
                   (int)poly.faces->v.size());
            Rendering r(poly, 1920, 1080);
            // r.RenderMesh(Shadow(poly));
            r.RenderPerspectiveWireframe(poly, 0x99AA99AA);
            static int count = 0;
            r.Save(StringPrintf("poly.%d.png", count++));
          }

          std::optional<int> iters = TrySolve(&rc, poly);

          {
            MutexLock ml(&m);
            if (should_die) return;

            if (iters.has_value()) {
              histo.Observe(iters.value());
            } else {
              SolutionDB db;
              db.AddNopert(poly, METHOD);

              status.Printf(
                  "\n\n" APURPLE("** NOPERT **") "\n");
              for (const vec3 &v : poly.vertices) {
                status.Printf("  %s\n",
                              VecString(v).c_str());
              }
              should_die = true;
              exit(0);
            }
          }

          if (status_per.ShouldRun()) {
            MutexLock ml(&m);
            double total_time = timer.Seconds();
            double tps = polyhedra.Read() / total_time;

            status.Statusf(
                "%s\n"
                ACYAN("%s") " " AGREY("|") " "
                AWHITE("%d") " pt " AGREY("|") " "
                ABLUE("%s") " polyhedra "
                "in %s "
                "[" AWHITE("%.3f") "/s]\n",
                histo.SimpleANSI(HISTO_LINES).c_str(),
                lower_method.c_str(),
                num_points,
                FormatNum(polyhedra.Read()).c_str(),
                ANSI::Time(total_time).c_str(),
                tps);
          }

          delete poly.faces;
        }
      });
}

int main(int argc, char **argv) {
  ANSI::Init();
  printf("\n");

  int NUM_POINTS = 6;

  if (argc == 2) {
    NUM_POINTS = strtol(argv[1], nullptr, 10);
    CHECK(NUM_POINTS >= 3);
  }

  for (;;) {
    Nopert(60 * 60, NUM_POINTS);
    NUM_POINTS++;
  }

  printf("OK\n");
  return 0;
}
