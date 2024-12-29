#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <functional>
#include <limits>
#include <memory>
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
#include "point-set.h"
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

#define ABLOOD(s) AFGCOLOR(148, 0, 0, s)

DECLARE_COUNTERS(polyhedra, attempts, degenerate, u1_, u2_, u3_, u4_, u5_);

static constexpr int VERBOSE = 0;
static constexpr bool SAVE_EVERY_IMAGE = false;
static constexpr int HISTO_LINES = 10;

static StatusBar *status = nullptr;

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
static std::optional<int> TrySolve(int thread_idx,
                                   ArcFour *rc, const Polyhedron &poly) {
  CHECK(!poly.faces->v.empty());

  for (int iter = 0; iter < MAX_ITERS; iter++) {

    if (iter > 0 && (iter % 100) == 0) {
      status->Printf("[" APURPLE("%d") "] %d-point polyhedron "
                     AFGCOLOR(190, 220, 190, "not solved")
                     " after " AWHITE("%d") " iters...\n",
                     thread_idx,
                     (int)poly.vertices.size(), iter);

      if (iter == 2000) {
        std::string filename = StringPrintf("hard.%lld.%d.stl",
                                            time(nullptr), thread_idx);
        SaveAsSTL(poly, filename);
      }
    }

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

    const int seed = RandTo(rc, 0x7FFFFFFE);
    const auto &[args, error] =
      Opt::Minimize<D>(Loss, lb, ub, 1000, 2, 10, seed);

    if (error == 0.0) {
      if (iter > 100) {
        printf("%d-point polyhedron " AYELLOW("solved") " after "
               AWHITE("%d") " iters.\n",
               (int)poly.vertices.size(), iter);
      }
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
  struct Group {
    std::vector<frame3> rots;
    int points = 0;
  };
  Group tetrahedron, octahedron, icosahedron;
  SymmetryGroups() {
    {
      Polyhedron t = Tetrahedron();
      VertexRotationsTriangular(t, &tetrahedron.rots);
      tetrahedron.points = 4;
      delete t.faces;
    }

    {
      Polyhedron o = Octahedron();
      VertexRotationsQuadrilateral(o, &octahedron.rots);
      EdgeRotations(o, &octahedron.rots);
      octahedron.points = 6;
      delete o.faces;
    }

    {
      Polyhedron i = Icosahedron();
      VertexRotationsPentagonal(i, &icosahedron.rots);
      EdgeRotations(i, &icosahedron.rots);
      icosahedron.points = 20;
      delete i.faces;
    }

    status->Printf(
        AYELLOW("tetrahedron") " has " ACYAN("%d") " rotations, %d pts\n",
        (int)tetrahedron.rots.size(), tetrahedron.points);
    status->Printf(
        AYELLOW("octahedron") " has " ACYAN("%d") " rotations, %d pts\n",
        (int)octahedron.rots.size(), octahedron.points);
    status->Printf(
        AYELLOW("icosahedron") " has " ACYAN("%d") " rotations, %d pts\n",
        (int)icosahedron.rots.size(), icosahedron.points);
  }

 private:

  // For tetrahedron.
  void VertexRotationsTriangular(const Polyhedron &poly,
                                 std::vector<frame3> *rots) {
    for (const vec3 &v : poly.vertices) {
      vec3 axis = normalize(v);
      rots->push_back(
          yocto::rotation_frame(
              yocto::rotation_quat(axis, 2.0 / 3.0 * std::numbers::pi)));
      rots->push_back(
          yocto::rotation_frame(
              yocto::rotation_quat(axis, -2.0 / 3.0 * std::numbers::pi)));
    }
  }

  // For octahedron.
  void VertexRotationsQuadrilateral(const Polyhedron &poly,
                                    std::vector<frame3> *rots) {
    for (const vec3 &v : poly.vertices) {
      vec3 axis = normalize(v);
      for (int quarter_turn = 1; quarter_turn < 4; quarter_turn++) {
        rots->push_back(
            yocto::rotation_frame(
                yocto::rotation_quat(
                    axis, quarter_turn * std::numbers::pi / 2.0)));
      }
    }
  }

  // For icosahedron.
  void VertexRotationsPentagonal(const Polyhedron &poly,
                                 std::vector<frame3> *rots) {

    for (const vec3 &v : poly.vertices) {
      vec3 axis = normalize(v);
      for (int fifth_turn = 1; fifth_turn < 5; fifth_turn++) {
        rots->push_back(
            yocto::rotation_frame(
                yocto::rotation_quat(
                    axis, fifth_turn / 5.0 * std::numbers::pi / 2.0)));
      }
    }
  }


  // Take an edge and rotate it 180 degrees, using the axis that runs
  // from the origin to its midpoint. This flips the edge around (so it
  // ends up the same).
  void EdgeRotations(const Polyhedron &poly,
                     std::vector<frame3> *rots) {
    for (int i = 0; i < poly.vertices.size(); ++i) {
      const vec3 &v0 = poly.vertices[i];
      for (int j : poly.faces->neighbors[i]) {
        CHECK(i != j);
        // Only consider the edge in one orientation.
        if (i < j) {
          const vec3 &v1 = poly.vertices[j];
          const vec3 mid = (v0 + v1) * 0.5;
          const vec3 axis = normalize(mid);

          // But also, we don't want to do this for both an edge and
          // its opposite edge, because that gives us an equivalent
          // rotation. So only do this when the axis is in one half
          // space. We can check the dot product with an arbitrary
          // reference axis. This only fails if the rotation axis is
          // perpendicular to the reference axis, so use one that we
          // know is not perpendicular to any of the rotation axes in
          // these regular polyhedra.
          constexpr vec3 half_space{1.23456789, 0.1133557799, 0.777555};

          if (dot(half_space, axis) > 0.0) {
            // Rotate 180 degrees about the axis.
            rots->push_back(yocto::rotation_frame(
                                yocto::rotation_quat(axis, std::numbers::pi)));
          }
        }
      }
    }
  }

};


// Note that we can get more (because we require a minimum of
// two random points so that they aren't just the platonic solids)
// or fewer (because we deduplicate at the end) than NUM_POINTS,
// but we try to get close.
static Polyhedron RandomSymmetricPolyhedron(ArcFour *rc, int num_points) {
  static SymmetryGroups *symmetry = new SymmetryGroups;

  auto MakePoints = [rc](int num) {
      if (VERBOSE > 1) {
        status->Printf("MakePoints %d\n", num);
      }
      std::vector<vec3> pts;
      pts.reserve(num);
      for (int i = 0; i < num; i++) {
        vec3 v = normalize(RandomVec(rc));
        pts.push_back(v);
        if (VERBOSE > 1) {
          status->Printf("  point %s\n", VecString(v).c_str());
        }
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
        const SymmetryGroups::Group &group, int chance) -> bool {
        if (target_points >= group.points &&
            RandTo(rc, chance) == 0) {
          PointSet pointset;

          if (VERBOSE > 0) {
            status->Printf(
                "* To quiescence, method " ACYAN("%s") ", target %d\n",
                what, target_points);
          }

          while (pointset.Size() < target_points) {
            if (VERBOSE > 1) {
              status->Printf(
                  "Enter loop with %d points, target %d\n",
                  (int)pointset.Size(), target_points);
            }
            std::vector<vec3> todo = {
              normalize(RandomVec(rc))
            };

            // Run until quiescence, even if we exceed the target
            // point size.
            while (!todo.empty()) {
              if (pointset.Size() > 6000) {
                status->Printf(ARED("Too big!!") "\n");

                DebugPointCloudAsSTL(pointset.ExtractVector(),
                                     "too-big.stl");

                LOG(FATAL) << "Something is wrong";
              }

              if (pointset.Size() == target_points)
                break;

              vec3 v = todo.back();
              todo.pop_back();

              if (!pointset.Contains(v)) {
                // identity is not included.
                pointset.Add(v);
              }

              for (const frame3 &rot : group.rots) {
                vec3 vr = yocto::transform_point(rot, v);
                if (pointset.Contains(vr)) {
                  // Skip.
                } else {
                  pointset.Add(vr);
                  todo.push_back(vr);
                }
              }
            }
          }

          points = pointset.ExtractVector();
          if (VERBOSE > 0) {
            status->Printf("Done with num points = " AYELLOW("%d") "\n",
                           (int)points.size());
          }

          method = what;
          return true;
        }
        return false;
      };

    if (false && UsePolyhedralGroup("icosahedron", symmetry->icosahedron, 3)) {
      // nothing
    } else if (UsePolyhedralGroup("octahedron", symmetry->octahedron, 3)) {
      // nothing
    } else if (UsePolyhedralGroup("tetrahedron", symmetry->tetrahedron, 3)) {
      // nothing
    } else {
      // Then we can always use the cyclic (or dihedral if reflections
      // are on) group.

      // Pick a nontrivial n, but not too big.
      int n = std::max((int)RandTo(rc, target_points), 2);
      target_points = std::max(target_points / n, 2);
      if (VERBOSE > 0) {
        status->Printf("Cyclic n=" AYELLOW("%d") "\n", n);
      }

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

    if (VERBOSE > 0) {
      status->Printf("Target points " AWHITE("%d") " method " ACYAN("%s")
                     " refl %s\n",
                     target_points, method, include_reflection ? AGREEN("y") :
                     AORANGE("n"));
    }

    if (VERBOSE > 1) {
      status->Printf("Result:\n");
      for (const vec3 & v: points) {
        status->Printf("  %s\n", VecString(v).c_str());
      }
    }

    std::optional<Polyhedron> poly =
      ConvexPolyhedronFromVertices(std::move(points), "randomsymmetric");
    if (poly.has_value()) {
      CHECK(!poly.value().vertices.empty());
      return std::move(poly.value());
    } else {
      if (VERBOSE > 0) {
        status->Printf(AORANGE("degenerate") "\n");
      }
      degenerate++;
    }
  }
}

// Must be thread safe.
struct CandidateMaker {
  virtual ~CandidateMaker() {}
  virtual std::string Name() const = 0;

  virtual std::pair<int64_t, int64_t> Frac() = 0;

  // or nullopt when done.
  virtual std::optional<Polyhedron> Next() = 0;
};

struct RandomCandidateMaker : public CandidateMaker {

  RandomCandidateMaker(int num_points, int method, int64_t max_seconds) :
    rc(StringPrintf("rand.%d.%lld", method, time(nullptr))),
    num_points(num_points),
    method(method), max_seconds(max_seconds) {
    std::string name = Util::lcase(SolutionDB::NopertMethodName(method));
    (void)Util::TryStripPrefix("nopert_method_", &name);
    lower_name = name;
  }

  std::string Name() const override {
    return lower_name;
  }

  std::pair<int64_t, int64_t> Frac() override {
    MutexLock ml(&m);
    return std::make_pair((int64_t)run_timer.Seconds(), max_seconds);
  }

  std::optional<Polyhedron> Next() override {
    MutexLock ml(&m);
    if (run_timer.Seconds() > max_seconds)
      return std::nullopt;
    // PERF: If we had multiple random streams, this could avoid
    // taking the lock.
    switch (method) {
    case SolutionDB::NOPERT_METHOD_RANDOM:
      return {RandomPolyhedron(&rc, num_points)};
    case SolutionDB::NOPERT_METHOD_CYCLIC:
      return {RandomCyclicPolyhedron(&rc, num_points)};
    case SolutionDB::NOPERT_METHOD_SYMMETRIC:
      return {RandomSymmetricPolyhedron(&rc, num_points)};
    default:
      LOG(FATAL) << "Bad Nopert method";
      return Cube();
    }
  }

 private:
  std::mutex m;
  ArcFour rc;
  const int num_points = 0;
  const int method = 0;
  const int64_t max_seconds = 0;
  std::string lower_name;
  Timer run_timer;
};

static bool Nopert(int num_points) {
  polyhedra.Reset();
  attempts.Reset();
  degenerate.Reset();

  static constexpr int NUM_THREADS = 4;
  // static constexpr int METHOD = SolutionDB::NOPERT_METHOD_RANDOM;
  static constexpr int METHOD = SolutionDB::NOPERT_METHOD_SYMMETRIC;
  static constexpr int64_t MAX_SECONDS = 60 * 60;

  std::unique_ptr<CandidateMaker> candidates(
      new RandomCandidateMaker(num_points, METHOD, MAX_SECONDS));

  std::string name = candidates->Name();

  Timer timer;
  AutoHisto histo(10000);
  Periodically status_per(5.0);
  std::mutex m;
  bool should_die = false;
  double total_gen_sec = 0.0;
  double total_solve_sec = 0.0;
  bool success = false;

  ParallelFan(
      NUM_THREADS,
      [&](int thread_idx) {
        ArcFour rc(StringPrintf("noperts.%d.%lld\n",
                                thread_idx, time(nullptr)));

        for (;;) {
          {
            MutexLock ml(&m);
            if (should_die) return;
          }

          Timer gen_timer;
          std::optional<Polyhedron> opoly =
            candidates->Next();
          const double gen_sec = gen_timer.Seconds();

          if (!opoly.has_value()) {
            MutexLock ml(&m);
            if (should_die) return;

            should_die = true;
            status->Printf("No more polyhedra after %s\n",
                           ANSI::Time(timer.Seconds()).c_str());
            SolutionDB db;
            db.AddNopertAttempt(num_points, polyhedra.Read(), histo,
                                METHOD);
            return;
          }

          Polyhedron poly = std::move(opoly.value());
          polyhedra++;

          if constexpr (SAVE_EVERY_IMAGE) {
            printf("Rendering %d points %d faces...\n",
                   (int)poly.vertices.size(),
                   (int)poly.faces->v.size());
            Rendering r(poly, 1920, 1080);
            // r.RenderMesh(Shadow(poly));
            r.RenderPerspectiveWireframe(poly, 0x99AA99AA);
            static int count = 0;
            r.Save(StringPrintf("poly.%d.png", count++));
          }

          Timer solve_timer;
          std::optional<int> iters = TrySolve(thread_idx, &rc, poly);
          const double solve_sec = solve_timer.Seconds();

          {
            MutexLock ml(&m);
            if (should_die) return;

            total_gen_sec += gen_sec;
            total_solve_sec += solve_sec;

            if (iters.has_value()) {
              histo.Observe(iters.value());
            } else {
              SolutionDB db;
              db.AddNopert(poly, METHOD);

              status->Printf(
                  "\n\n" ABGCOLOR(200, 0, 200,
                                  ANSI_DARK_GREY "***** "
                                  ANSI_YELLOW "NOPERT"
                                  ANSI_DARK_GREY " with "
                                  ANSI_WHITE "%d"
                                  ANSI_DARK_GREY " vertices *****")
                  "\n", (int)poly.vertices.size());

              for (const vec3 &v : poly.vertices) {
                status->Printf("  %s\n",
                               VecString(v).c_str());
              }

              should_die = true;
              success = true;
              return;
            }
          }

          if (status_per.ShouldRun()) {
            MutexLock ml(&m);
            double total_time = timer.Seconds();
            const int64_t polys = polyhedra.Read();
            double tps = polys / total_time;

            const auto &[numer, denom] = candidates->Frac();

            std::string msg =
              StringPrintf(
                  AWHITE("%d") AWHITE("⋮") " "
                  ACYAN("%s") " " AGREY("|") " " AGREY("|") " "
                  ARED("%s") ABLOOD("×") " "
                  ABLUE("%s") AWHITE("∎") " "
                  "[" AWHITE("%.1f") "/s] %s gen %s sol",
                  num_points,
                  name.c_str(),
                  FormatNum(degenerate.Read()).c_str(),
                  FormatNum(polys).c_str(),
                  tps,
                  ANSI::Time(total_gen_sec).c_str(),
                  ANSI::Time(total_solve_sec).c_str());

            status->Printf("%s\n", msg.c_str());

            std::string bar =
              ANSI::ProgressBar(numer, denom,
                                msg, total_time);

            status->Statusf(
                "%s\n"
                "%s\n",
                histo.SimpleANSI(HISTO_LINES).c_str(),
                bar.c_str());
          }
          delete poly.faces;
        }
      });

  return success;
}

int main(int argc, char **argv) {
  ANSI::Init();
  printf("\n");

  status = new StatusBar(HISTO_LINES + 2);

  int NUM_POINTS = 6;

  if (argc == 2) {
    NUM_POINTS = strtol(argv[1], nullptr, 10);
    CHECK(NUM_POINTS >= 3);
  }

  for (;;) {
    // Could break if we find one, but might as well try to find
    // more examples.
    (void)Nopert(NUM_POINTS);
    NUM_POINTS++;
  }

  printf("OK\n");
  return 0;
}
