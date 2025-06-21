#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <format>
#include <functional>
#include <memory>
#include <mutex>
#include <numbers>
#include <optional>
#include <span>
#include <string>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

#include "ansi.h"
#include "arcfour.h"
#include "atomic-util.h"
#include "auto-histo.h"
#include "base/stringprintf.h"
#include "hull3d.h"
#include "interval-cover-util.h"
#include "interval-cover.h"
#include "opt/large-optimizer.h"
#include "opt/opt.h"
#include "periodically.h"
#include "point-map.h"
#include "polyhedra.h"
#include "randutil.h"
#include "rendering.h"
#include "solutions.h"
#include "status-bar.h"
#include "symmetry-groups.h"
#include "threadutil.h"
#include "timer.h"
#include "util.h"
#include "yocto_matht.h"
#include "smallest-sphere.h"

// Try to find counterexamples.

// TODO: Random planar faces that aren't just triangles (regular polygons, etc.)

#define ABLOOD(s) AFGCOLOR(148, 0, 0, s)

DECLARE_COUNTERS(polyhedra, attempts, degenerate, skipped,
                 hard, successes, infeasible, random_retries);

static constexpr int VERBOSE = 0;
static constexpr bool SAVE_EVERY_IMAGE = false;
static constexpr int HISTO_LINES = 10;

using IntervalSet = IntervalCover<bool>;

static StatusBar *status = nullptr;

using vec2 = yocto::vec<double, 2>;
using vec3 = yocto::vec<double, 3>;
using vec4 = yocto::vec<double, 4>;
using mat4 = yocto::mat<double, 4>;
using quat4 = yocto::quat<double, 4>;
using frame3 = yocto::frame<double, 3>;

// Each of x,y,z in [-1, 1].
static vec3 RandomVec(ArcFour *rc) {
  return vec3(2.0 * RandDouble(rc) - 1.0,
              2.0 * RandDouble(rc) - 1.0,
              2.0 * RandDouble(rc) - 1.0);
}

// Return the number of iterations taken, or nullopt if we exceeded
// the limit. If solved and the arguments are non-null, sets the outer
// frame and inner frame to some solution.
static constexpr int NOPERT_ITERS = 200000;
static constexpr int MIN_REPORT_ITERS = 5000;
static constexpr bool SAVE_HARD = false;

// One iteration. Wants exclusive access to rc.
static std::optional<std::pair<frame3, frame3>>
Minimize(ArcFour *rc, const Polyhedron &poly) {
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
      return translate * rotate;
    };

  std::function<double(const std::array<double, D> &)> Loss =
    [&poly, &OuterFrame, &InnerFrame](
        const std::array<double, D> &args) {
      attempts++;
      // Note that random polyhedra often do not contain the
      // origin, so it is not sound to use LossFunctionContainsOrigin.
      // We could consider requiring this (using smallest sphere)?
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
    return {std::make_pair(OuterFrame(args), InnerFrame(args))};
  } else {
    return std::nullopt;
  }
}


static std::optional<int> DoSolve(int thread_idx,
                                  ArcFour *rc, const Polyhedron &poly,
                                  frame3 *outer_frame_out,
                                  frame3 *inner_frame_out) {
  CHECK(!poly.faces->v.empty());

  for (int iter = 0; iter < NOPERT_ITERS; iter++) {

    if (iter > 0 && (iter % 5000) == 0) {
      status->Print("[" APURPLE("{}") "] {}-point polyhedron "
                     AFGCOLOR(190, 220, 190, "not solved")
                     " after " AWHITE("{}") " iters...\n",
                     thread_idx,
                     (int)poly.vertices.size(), iter);

      if (iter == 20000) {
        hard++;
        if (SAVE_HARD) {
          std::string filename = std::format("hard.{}.{}.stl",
                                             time(nullptr), thread_idx);
          SaveAsSTL(poly, filename);
          status->Print("[" APURPLE("{}") "] Hard {}-point polyhedron saved "
                        "to " AGREEN("{}") "\n",
                        thread_idx, (int)poly.vertices.size(),
                        filename);
        }
      }
    }

    if (auto ro = Minimize(rc, poly)) {

      const auto &[outer, inner] = ro.value();
      if (outer_frame_out != nullptr) {
        *outer_frame_out = outer;
      }
      if (inner_frame_out != nullptr) {
        *inner_frame_out = inner;
      }

      return std::make_optional(iter);
    }
  }

  return std::nullopt;
}

static std::optional<int> ParallelSolve(
    int num_threads,
    ArcFour *rc_all, const Polyhedron &poly,
    frame3 *outer_frame_out,
    frame3 *inner_frame_out) {
  CHECK(!poly.faces->v.empty());

  std::mutex m;

  // When < 0, we have failed.
  int64_t iters_left = NOPERT_ITERS;
  int64_t iters_done = 0;
  // First thread to find a solution sets this to true and
  // modifies outer_frame_out, inner_frame_out.
  bool success = false;

  const int64_t BATCH_SIZE = 1024;

  Periodically status_per(10.0);

  ParallelFan(
      num_threads,
      [&](int thread_idx) {
        m.lock();
        ArcFour rc(std::format("{}.{}", thread_idx, Rand64(rc_all)));
        m.unlock();

        for (;;) {
          {
            MutexLock ml(&m);
            if (success) return;
            if (iters_left < 0) return;
            iters_left -= BATCH_SIZE;
          }

          // Do a batch.

          for (int i = 0; i < BATCH_SIZE; i++) {
            if (auto ro = Minimize(&rc, poly)) {
              MutexLock ml(&m);
              // account for partial batch
              iters_done += i;
              if (success) return;

              const auto &[outer, inner] = ro.value();
              if (outer_frame_out != nullptr) {
                *outer_frame_out = outer;
              }
              if (inner_frame_out != nullptr) {
                *inner_frame_out = inner;
              }

              success = true;
              return;
            }
          }

          {
            MutexLock ml(&m);
            iters_done += BATCH_SIZE;

            if (iters_done > MIN_REPORT_ITERS) {
              status_per.RunIf([&]() {
                  status->Print("{}-point polyhedron "
                                AFGCOLOR(190, 220, 190, "not solved")
                                " after " AWHITE("{}") " iters...\n",
                                (int)poly.vertices.size(), iters_done);
                });
            }
          }
        }
      });

  if (success) {
    return {iters_done};
  }

  return std::nullopt;
}


[[maybe_unused]]
static Polyhedron RandomTetrahedron(ArcFour *rc) {
  Polyhedron tetra = Tetrahedron();
  for (;;) {
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

std::vector<vec3> RandomHullOfSize(ArcFour *rc, int num_points) {
  std::vector<vec3> pts;
  pts.reserve(num_points);
  for (;;) {
    while (pts.size() < num_points * 3) {
      pts.push_back(RandomVec(rc));
    }

    // The set of points on the hull.
    std::unordered_set<int> hull_pts;
    for (int a : Hull3D::HullPoints(pts)) {
      hull_pts.insert(a);
    }

    std::vector<vec3> orig = pts;

    {
      std::vector<vec3> new_pts;
      new_pts.reserve(hull_pts.size());
      for (int a : hull_pts) {
        new_pts.push_back(pts[a]);
      }
      pts = std::move(new_pts);
      Shuffle(rc, &pts);
    }

    if (pts.size() > num_points) {
      pts.resize(num_points);
    }

    if (pts.size() == num_points) {

      constexpr bool SELF_CHECK = false;
      if (SELF_CHECK) {
        std::vector<int> hv = Hull3D::HullPoints(pts);
        std::unordered_set<int> hull_pts(hv.begin(), hv.end());

        if (hull_pts.size() != pts.size()) {
          DebugPointCloudAsSTL(orig, "hull_not_preserved-orig.stl");
          DebugPointCloudAsSTL(pts, "hull_not_preserved-hull.stl");

          CHECK(hull_pts.size() == pts.size()) <<
            hull_pts.size() << " of " << pts.size();
        }
        for (int i = 0; i < pts.size(); i++) {
          CHECK(hull_pts.contains(i));
        }
      }


      return pts;
    }

    random_retries++;
  }
}

Polyhedron RandomPolyhedron(ArcFour *rc, int num_points) {
  for (;;) {
    std::vector<vec3> pts = RandomHullOfSize(rc, num_points);
    CHECK(pts.size() == num_points);

    // Center at 0,0.
    const auto &[center, r] = SmallestSphere::Smallest(rc, pts);
    for (vec3 &v : pts) {
      v -= center;
    }

    std::optional<Polyhedron> poly =
      PolyhedronFromConvexVertices(pts, "random");
    if (poly.has_value()) {
      return std::move(poly.value());
    } else {
      DebugPointCloudAsSTL(pts, "random-degenerate.stl");
      degenerate++;
      LOG(FATAL) << "hi";
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
      PolyhedronFromConvexVertices(std::move(pts), "randomcyclic");
    if (poly.has_value()) {
      return std::move(poly.value());
    } else {
      degenerate++;
    }
  }
}

// Note that we can get more (because we require a minimum of
// two random points so that they aren't just the platonic solids)
// or fewer (because we deduplicate at the end) than NUM_POINTS,
// but we try to get close.
static Polyhedron RandomSymmetricPolyhedron(ArcFour *rc, int num_points) {
  static const SymmetryGroups *symmetry = new SymmetryGroups;

  static constexpr SymmetryGroup GROUPS_ENABLED = SYM_TETRAHEDRAL;

  auto MakePoints = [rc](int num) {
      if (VERBOSE > 1) {
        status->Print("MakePoints {}\n", num);
      }
      std::vector<vec3> pts;
      pts.reserve(num);
      for (int i = 0; i < num; i++) {
        vec3 v = normalize(RandomVec(rc));
        pts.push_back(v);
        if (VERBOSE > 1) {
          status->Print("  point {}\n", VecString(v));
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
          PointSet3 pointset;

          if (VERBOSE > 0) {
            status->Print(
                "* To quiescence, method " ACYAN("{}") ", target {}\n",
                what, target_points);
          }

          while (pointset.Size() < target_points) {
            if (VERBOSE > 1) {
              status->Print(
                  "Enter loop with {} points, target {}\n",
                  (int)pointset.Size(), target_points);
            }
            std::vector<vec3> todo = {
              normalize(RandomVec(rc))
            };

            // Run until quiescence, even if we exceed the target
            // point size.
            while (!todo.empty()) {
              if (pointset.Size() > 6000) {
                status->Print(ARED("Too big!!") "\n");

                DebugPointCloudAsSTL(pointset.Points(),
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

          points = pointset.Points();
          if (VERBOSE > 0) {
            status->Print("Done with num points = " AYELLOW("{}") "\n",
                          (int)points.size());
          }

          method = what;
          return true;
        }
        return false;
      };

    if ((GROUPS_ENABLED & SYM_ICOSAHEDRAL) &&
        UsePolyhedralGroup("icosahedron", symmetry->icosahedron, 3)) {
      // nothing
    } else if ((GROUPS_ENABLED & SYM_OCTAHEDRAL) &&
               UsePolyhedralGroup("octahedron", symmetry->octahedron, 3)) {
      // nothing
    } else if ((GROUPS_ENABLED & SYM_TETRAHEDRAL) &&
               UsePolyhedralGroup("tetrahedron", symmetry->tetrahedron, 3)) {
      // nothing
    } else {
      // Then we can always use the cyclic (or dihedral if reflections
      // are on) group.

      // TODO: Consider sometimes generating m>3 points on planes, to
      // create non-triangular facets.

      // Pick a nontrivial n, but not too big.
      int n = std::max((int)RandTo(rc, target_points), 2);
      target_points = std::max(target_points / n, 2);
      if (VERBOSE > 0) {
        status->Print("Cyclic n=" AYELLOW("{}") "\n", n);
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
      status->Print("Target points " AWHITE("{}") " method " ACYAN("{}")
                    " refl {}\n",
                    target_points, method, include_reflection ? AGREEN("y") :
                    AORANGE("n"));
    }

    if (VERBOSE > 1) {
      status->Print("Result:\n");
      for (const vec3 & v: points) {
        status->Print("  {}\n", VecString(v).c_str());
      }
    }

    std::optional<Polyhedron> poly =
      PolyhedronFromConvexVertices(std::move(points), "randomsymmetric");
    if (poly.has_value()) {
      CHECK(!poly.value().vertices.empty());
      return std::move(poly.value());
    } else {
      if (VERBOSE > 0) {
        status->Print(AORANGE("degenerate") "\n");
      }
      degenerate++;
    }
  }
}

static Polyhedron RandomRhombicPolyhedron(ArcFour *rc, int num_points) {
  static const SymmetryGroups *symmetry = new SymmetryGroups;

  auto Rotate = [rc](const vec3 &v) {
      switch (RandTo(rc, 4)) {
      default:
      case 0:
        return v;
      case 1:
        return vec3(v.y, v.z, v.x);
      case 2:
        return vec3(v.z, v.x, v.y);
      }
    };

  auto Permute = [rc](const vec3 &v) {
      switch (RandTo(rc, 6)) {
      default:
      case 0:
        return v;
      case 1:
        return vec3(v.y, v.z, v.x);
      case 2:
        return vec3(v.z, v.x, v.y);
      case 3:
        return vec3(v.x, v.z, v.y);
      case 4:
        return vec3(v.y, v.x, v.z);
      case 5:
        return vec3(v.z, v.y, v.x);
      }
    };

  for (;;) {
    // Rather than taking an arbitrary point set and inducing symmetry
    // for it, this generates points in special positions (on the
    // axes of symmetry).

    // All points that have ever been added.
    PointSet3 point_set;
    std::vector<vec3> points;

    while (points.size() < num_points) {
      double dist = 0.5 + RandDouble(rc);

      std::vector<vec3> todo;
      auto MaybeAdd = [&](const vec3 &pt) {
          if (!point_set.Contains(pt)) {
            point_set.Add(pt);
            todo.push_back(pt);
            points.push_back(pt);
          }
        };

      switch (RandTo(rc, 4)) {
      default:
      case 0: {
        // 4-fold axis (z-axis)
        MaybeAdd(Rotate(vec3{0.0, 0.0, dist}));
        break;
      }
      case 1: {
        // 3-fold axis (body diagonal)
        MaybeAdd(Rotate(normalize(vec3{1.0, 1.0, 1.0}) * dist));
        break;
      }
      case 2: {
        // 2-fold axis (edge midpoint direction, e.g., (1, 1, 0))
        MaybeAdd(Rotate(normalize(vec3{1.0, 1.0, 0.0}) * dist));
        break;
      }

      case 3: {
        double d2 = RandDouble(rc);
        MaybeAdd(Permute(normalize(vec3{1.0, d2, 0.0}) * dist));
        break;
      }

      }

      // Add the full orbit of this point, and any points it generates.
      while (!todo.empty()) {
        vec3 p = todo.back();
        todo.pop_back();

        for (const frame3 &rot : symmetry->octahedron.rots) {
          vec3 rotated_point = yocto::transform_point(rot, p);
          MaybeAdd(rotated_point);
        }
      }

      points = Hull3D::ReduceToHull(points);
    }

    std::optional<Polyhedron> poly =
      PolyhedronFromVertices(points, "randomrhombic");
    if (poly.has_value()) {
      CHECK(!poly.value().vertices.empty());
      return std::move(poly.value());
    } else {
      degenerate++;
    }
  }
}

// Like a pointer to a polyhedron, but it should be explicitly
// returned.
struct LeasedPoly {
  virtual ~LeasedPoly() {}
  virtual const Polyhedron &Value() const = 0;
};

// Must be thread safe.
struct CandidateMaker {
  virtual ~CandidateMaker() {}
  virtual int Method() const = 0;
  virtual std::string Name() const = 0;
  virtual std::string Info() { return ""; }

  virtual std::pair<int64_t, int64_t> Frac() = 0;

  // Or nullptr when done.
  virtual std::unique_ptr<LeasedPoly> Next() = 0;

  virtual void AttemptFailed(int64_t num_poly, const AutoHisto &iter_histo) = 0;
};

struct RandomCandidateMaker : public CandidateMaker {

  RandomCandidateMaker(int num_points, int method, int64_t max_seconds) :
    rc(std::format("rand.{}.{}", method, time(nullptr))),
    num_points(num_points),
    method(method), max_seconds(max_seconds) {
    std::string name = Util::lcase(SolutionDB::NopertMethodName(method));
    (void)Util::TryStripPrefix("nopert_method_", &name);
    lower_name = name;
  }

  int Method() const override { return method; }

  std::string Name() const override {
    return lower_name;
  }

  std::string Info() override {
    return std::format(APURPLE("{}") AWHITE("⋮"), num_points);
  }

  std::pair<int64_t, int64_t> Frac() override {
    MutexLock ml(&m);
    return std::make_pair((int64_t)run_timer.Seconds(), max_seconds);
  }

  std::unique_ptr<LeasedPoly> Next() override {
    MutexLock ml(&m);
    auto LP = [](Polyhedron p) {
        return std::unique_ptr<LeasedPoly>(new RandomLeasedPoly(std::move(p)));
      };

    if (run_timer.Seconds() > max_seconds)
      return {nullptr};
    // PERF: If we had multiple random streams, this could avoid
    // taking the lock.
    switch (method) {
    case SolutionDB::NOPERT_METHOD_RANDOM:
      return LP(RandomPolyhedron(&rc, num_points));
    case SolutionDB::NOPERT_METHOD_CYCLIC:
      return LP(RandomCyclicPolyhedron(&rc, num_points));
    case SolutionDB::NOPERT_METHOD_SYMMETRIC:
      return LP(RandomSymmetricPolyhedron(&rc, num_points));
    case SolutionDB::NOPERT_METHOD_RHOMBIC:
      return LP(RandomRhombicPolyhedron(&rc, num_points));
    default:
      LOG(FATAL) << "Bad Nopert method";
      return LP(Cube());
    }
  }

  void AttemptFailed(int64_t num_poly, const AutoHisto &iter_histo) override {
    SolutionDB db;
    db.AddNopertAttempt(num_points, num_poly, iter_histo, method);
  }

 private:

  struct RandomLeasedPoly : public LeasedPoly {
    RandomLeasedPoly(Polyhedron p) : poly(std::move(p)) {}

    ~RandomLeasedPoly() override {
      delete poly.faces;
    }

    const Polyhedron &Value() const override { return poly; }

    Polyhedron poly;
  };

  std::mutex m;
  ArcFour rc;
  const int num_points = 0;
  const int method = 0;
  const int64_t max_seconds = 0;
  std::string lower_name;
  Timer run_timer;
};


// All subsets of a polyhedron. Does not attempt to account for
// symmetry at all, so for many of the regular polyhedra this
// can be very redundant.
struct ReduceCandidateMaker : public CandidateMaker {
  ReduceCandidateMaker(int method,
                       // These are bitmasks.
                       // Records the completed ranges.
                       IntervalSet *done,
                       // The region to do.
                       uint64_t start, uint64_t end,
                       const char *reference_name) :
    save_per(60.0 * 10.0),
    method(method), done(done), reference_name(reference_name),
    start(start), end(end) {
    name = std::format("reduce_{}", reference_name);
    reference = PolyhedronByName(reference_name);
    next = start;
  }

  int Method() const override { return method; }

  std::pair<int64_t, int64_t> Frac() override {
    MutexLock ml(&m);
    return std::make_pair(next - start, end - start);
  }

  std::string Info() override {
    MutexLock ml(&m);
    IntervalSet::Span sp = done->GetPoint(next);
    return std::format(
        APURPLE("{}") AGREY("-") APURPLE("{}"),
        sp.start, sp.end);
  }

  std::unique_ptr<LeasedPoly> Next() override {
    auto LP = [this](uint64_t idx, Polyhedron p) {
        return std::unique_ptr<LeasedPoly>(
            new ReduceLeasedPoly(this, idx, std::move(p)));
      };

    MutexLock ml(&m);

    if (save_per.ShouldRun()) {
      Save();
    }

    for (; next < end; next++) {
      const uint64_t idx = next;
      int pop = std::popcount<uint64_t>(idx);
      // Skip degenerate ones.
      if (pop < 4 || pop == reference.vertices.size()) {
        done->SetPoint(idx, true);
        continue;
      }

      // Also: Skip completed ranges.
      IntervalSet::Span sp = done->GetPoint(next);
      if (sp.data == true) {
        // Done until infinity.
        if (IntervalSet::IsAfterLast(sp.end))
          break;

        // We want to try sp.end (which is the start of the next
        // interval) next. But next gets incremented when we
        // continue, so place it one before.
        CHECK(next < sp.end);
        skipped += (sp.end - 1 - next);
        next = sp.end - 1;
        continue;
      }


      std::vector<vec3> vertices;
      vertices.reserve(pop);
      for (int i = 0; i < reference.vertices.size(); i++) {
        if (idx & (int64_t{1} << i)) {
          vertices.push_back(reference.vertices[i]);
        }
      }

      // PERF: We could reduce lock contention by doing this
      // without holding the lock, but poly generation is
      // only a tiny fraction of the overall time.
      std::optional<Polyhedron> poly =
        PolyhedronFromConvexVertices(std::move(vertices), "reduced");
      if (poly.has_value()) {
        next++;
        CHECK(!poly.value().vertices.empty());
        return LP(idx, std::move(poly.value()));
      } else {
        done->SetPoint(idx, true);
        degenerate++;
      }
    }

    // Save when we exhaust the range, but only once.
    if (!saved_final) {
      Save();
      saved_final = true;
    }

    return {nullptr};
  }

  std::string Name() const override {
    return name;
  }

  ~ReduceCandidateMaker() {
    delete reference.faces;
  }

  void AttemptFailed(int64_t num_poly, const AutoHisto &iter_histo) override {
    MutexLock ml(&m);
    // Should insert or something?
    Save();
  }

  // Must hold lock!
  void Save() {
    std::string filename = Filename(method, reference_name);
    // XXX To allow parallel work, we could merge with the file here.
    // There's still a race condition without some kind of filesystem
    // locking, though.
    Util::WriteFile(filename, IntervalCoverUtil::ToString(*done));
    status->Print("Wrote " AGREEN("{}") "\n", filename);
  }

  static std::string Filename(int method, const char *reference_name) {
    return std::format("noperts-reduce-{}-{}.txt", method, reference_name);
  }


 private:
  friend struct ReduceLeasedPoly;

  void MarkDone(uint64_t idx) {
    MutexLock ml(&m);
    done->SetPoint(idx, true);
  }

  struct ReduceLeasedPoly : public LeasedPoly {
    ReduceLeasedPoly(ReduceCandidateMaker *parent,
                     uint64_t idx,
                     Polyhedron p) : parent(parent),
                                     idx(idx),
                                     poly(std::move(p)) {}

    ~ReduceLeasedPoly() override {
      parent->MarkDone(idx);
      delete poly.faces;
    }

    const Polyhedron &Value() const override { return poly; }

    ReduceCandidateMaker *parent = nullptr;
    uint64_t idx = 0;
    Polyhedron poly;
  };

  Periodically save_per;

  std::mutex m;
  bool saved_final = false;
  const int method = 0;
  IntervalSet *done = nullptr;
  const char *reference_name = nullptr;
  Polyhedron reference;
  uint64_t next = 0;
  [[maybe_unused]] const uint64_t start = 0, end = 0;
  std::string name;
};


static bool Nopert(CandidateMaker *candidates) {
  polyhedra.Reset();
  attempts.Reset();
  degenerate.Reset();

  static constexpr int NUM_THREADS = 8;

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
        ArcFour rc(std::format("noperts.{}.{}\n",
                               thread_idx, time(nullptr)));
        for (;;) {
          {
            MutexLock ml(&m);
            if (should_die) return;
          }

          Timer gen_timer;
          std::unique_ptr<LeasedPoly> lease = candidates->Next();
          const double gen_sec = gen_timer.Seconds();

          if (lease.get() == nullptr) {
            MutexLock ml(&m);
            if (should_die) return;

            should_die = true;
            status->Print("No more polyhedra after {}\n",
                          ANSI::Time(timer.Seconds()));
            candidates->AttemptFailed(polyhedra.Read(), histo);
            return;
          }

          const Polyhedron &poly = lease->Value();
          polyhedra++;

          if constexpr (SAVE_EVERY_IMAGE) {
            printf("Rendering %d points %d faces...\n",
                   (int)poly.vertices.size(),
                   (int)poly.faces->v.size());
            Rendering r(poly, 1920, 1080);
            // r.RenderMesh(Shadow(poly));
            r.RenderPerspectiveWireframe(poly, 0x99AA99AA);
            static int count = 0;
            r.Save(std::format("poly.{}.png", count++));
          }

          Timer solve_timer;
          std::optional<int> iters = DoSolve(thread_idx, &rc, poly,
                                             nullptr, nullptr);
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
              db.AddNopert(poly, candidates->Method());
              successes++;

              status->Print(
                  "\n\n" ABGCOLOR(200, 0, 200,
                                  ANSI_DARK_GREY "***** "
                                  ANSI_YELLOW "NOPERT"
                                  ANSI_DARK_GREY " with "
                                  ANSI_WHITE "{}"
                                  ANSI_DARK_GREY " vertices *****")
                  "\n", (int)poly.vertices.size());

              for (const vec3 &v : poly.vertices) {
                status->Print("  {}\n",
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

            std::string info = candidates->Info();
            const auto &[numer, denom] = candidates->Frac();

            ANSI::ProgressBarOptions options;
            options.include_frac = false;
            options.include_percent = true;

            std::string timing =
              std::format(
                  "Timing: [" AWHITE("{:.1f}") "/s] {} " APURPLE("gen")
                  " {} " ABLUE("sol"),
                  tps,
                  ANSI::Time(total_gen_sec),
                  ANSI::Time(total_solve_sec));

            std::string msg =
              std::format(
                  ACYAN("{}") " {}{}" AGREY("|") " "
                  ARED("{}") ABLOOD("×") " "
                  ABLUE("{}") APURPLE("∲") " "
                  ABLUE("{}") AWHITE("∎"),
                  name,
                  info, info.empty() ? "" : " ",
                  FormatNum(degenerate.Read()),
                  FormatNum(random_retries.Read()),
                  FormatNum(polys));

            std::string bar =
              ANSI::ProgressBar(numer, denom,
                                msg, total_time, options);

            status->Status(
                "{}\n"
                "{}\n"
                "{}\n",
                histo.SimpleANSI(HISTO_LINES),
                timing,
                bar);
          }
        }
      });

  return success;
}

static void UnOpt(int64_t num_points) {
  constexpr double MAX_SECONDS = 60.0 * 60.0;

  polyhedra.Reset();
  attempts.Reset();
  degenerate.Reset();

  Timer run_timer;
  std::mutex m;
  bool should_die = false;
  int64_t run_calls = 0;
  AutoHisto iter_histo;
  iter_histo.AddFlag(0.0);
  iter_histo.AddFlag(500.0);
  static constexpr int NUM_THREADS = 8;
  // static constexpr int NUM_THREADS = 1;
  Periodically status_per(5.0);
  ParallelFan(
      NUM_THREADS,
      [&](int thread_idx) {
        ArcFour rc(std::format("unopt.{}.{}.{}", thread_idx,
                               num_points, time(nullptr)));

        // Here we have nested optimization. The outer optimizer is trying to
        // set vertices such that the inner optimizer takes many iterations to
        // find solutions.

        // Each point is *three* parameters so we use the large optimizer
        // for this. No need for a cache since everything is a double.
        using LargeOpt = LargeOptimizer<false>;

        auto ToPoints = [num_points](const std::vector<double> &args) {
            std::vector<vec3> points;
            // Without loss of generality, we always place one point
            // at 0, 0, 1. This reduces the number of parameters to
            // search, but still covers all polyhedra (up to
            // scaling/rotation).
            points.emplace_back(0.0, 0.0, 1.0);

            CHECK(args.size() == (num_points - 1) * 3);
            for (int i = 0; i < args.size(); i += 3) {
              double theta = args[i + 0];
              double phi = args[i + 1];
              double len = args[i + 2];

              // Convert to Cartesian.
              double sin_theta = std::sin(theta);
              points.emplace_back(
                  len * sin_theta * std::cos(phi),
                  len * sin_theta * std::sin(phi),
                  len * cos(theta));
            }
            return points;
          };

        auto OuterLoss = [num_points, &ToPoints,
                          &rc, &m, &iter_histo](
            const std::vector<double> &args) -> LargeOpt::return_type {
            std::vector<vec3> points = ToPoints(args);
            CHECK(points.size() == num_points);

            // It would also be nice to reject polyhedra that do not
            // contain the origin, especially if we could do that cheaply
            // before computing the hull. But it's not that easy?

            // Insist that it be convex.
            std::vector<int> hull = Hull3D::HullPoints(points);
            if (hull.size() != num_points) {
              std::unordered_set<int> hull_set(hull.begin(), hull.end());
              double badness = 100000.0;

              for (int i = 0; i < num_points; i++) {
                if (!hull_set.contains(i)) {
                  // This point is not in the hull, which means it's
                  // inside it. Don't allow this, and provide some
                  // gradient.
                  //
                  // Best would be to compute the distance from the
                  // point to the hull itself, but this is
                  // computationally expensive (need to compute the
                  // faces, do various distance checks). We know that
                  // making the length longer will always move it
                  // towards the hull (since we are inside), so we
                  // penalize it for being short, simply by negating
                  // the length.
                  double len = args[i * 3 + 2];
                  badness += -len;
                }
              }

              infeasible++;
              // Not feasible.
              /*
              status->Print("Infeasible ({} pts; {} hull) with badness {}",
                            (int)points.size(), (int)hull.size(),
                            badness);
              */
              return std::pair(badness, false);
            }

            // Otherwise, we have a convex polyhedron.
            // PERF: We don't actually need to compute the faces, which
            // is slow!
            std::optional<Polyhedron> opoly = PolyhedronFromConvexVertices(
                std::move(points), "unopt");

            if (!opoly.has_value()) {
              // This can happen if the hull has degenerate faces
              // (colinear points?).
              // status->Print("wasn't convex??");
              degenerate++;
              return LargeOpt::INFEASIBLE;
            }

            const Polyhedron &poly = opoly.value();
            polyhedra++;
            std::optional<int> oiters = DoSolve(0, &rc, poly,
                                                nullptr, nullptr);
            delete poly.faces;

            if (!oiters.has_value()) {
              // Success!
              return std::make_pair(0.0, true);
            } else {
              int iters = oiters.value();
              {
                MutexLock ml(&m);
                iter_histo.Observe(iters);
              }
              CHECK(iters <= NOPERT_ITERS);
              /*
              status->Print("Feasible! {} iters\n", iters);
              */
              return std::make_pair(NOPERT_ITERS - iters, true);
            }
          };

        // θ, φ, ρ triples
        std::vector<LargeOpt::arginfo> arginfos;
        for (int i = 0; i < num_points - 1; i++) {
          // polar angle (theta)
          if (i == 0) {
            // we only care about polyhedra that contain the origin,
            // so we know at least one vertex is on the bottom hemisphere
            // (the fixed vertex is in the upper hemisphere at (0, 0, 1)).
            // So we say that the first parameterized point has reduced
            // range.
            arginfos.push_back(LargeOpt::Double(std::numbers::pi * 0.5,
                                                std::numbers::pi));
          } else {
            arginfos.push_back(LargeOpt::Double(0.0, std::numbers::pi));
          }

          // azimuth angle (phi)
          arginfos.push_back(LargeOpt::Double(0.0, 2.0 * std::numbers::pi));
          // length
          arginfos.push_back(LargeOpt::Double(0.000001, std::numbers::sqrt2));
        }

        LargeOpt lopt(OuterLoss, (num_points - 1) * 3, Rand64(&rc));

        // Need a feasible input to start. Anything is feasible as long as
        // it's a convex polyhedron, so use a random cyclic one.
        // (Note that this can still generate degenerate polyhedra; maybe
        // we should use a more regular one?)
        std::vector<double> start;
        for (int i = 0; i < num_points - 1; i++) {
          // We can easily generate such polyhedra using spherical coordinates,
          // by just setting length = 1.
          if (i == 0) {
            start.push_back(RandDouble(&rc) * std::numbers::pi * 0.5 +
                            std::numbers::pi * 0.5);
          } else {
            start.push_back(RandDouble(&rc) * std::numbers::pi);
          }
          start.push_back(RandDouble(&rc) * std::numbers::pi * 2.0);
          start.push_back(1.0);
        }

        lopt.Sample(start);

        for (;;) {
          {
            MutexLock ml(&m);
            if (should_die) {
              status->Print("Thread {} exits.\n", thread_idx);
              return;
            }
            const double total_time = run_timer.Seconds();
            if (total_time > MAX_SECONDS) {
              should_die = true;
              status->Print("Time's up!");
              return;
            }

            status_per.RunIf([&]{
                std::string timing = "TODO: Timing";
                double numer = total_time;
                double denom = MAX_SECONDS;

                std::string msg =
                  std::format(
                      ACYAN("unopt") " " AWHITE("{}") " " AGREY("|") " "
                      AORANGE("{}") ABLOOD("∞") " "
                      ARED("{}") ABLOOD("×") " "
                      ABLUE("{}") AWHITE("∎") " "
                      AGREEN("{}") AWHITE("♚") " "
                      APURPLE("{}") AWHITE("∳"),
                      num_points,
                      FormatNum(infeasible.Read()),
                      FormatNum(degenerate.Read()),
                      FormatNum(polyhedra.Read()),
                      FormatNum(successes.Read()),
                      FormatNum(run_calls));

                ANSI::ProgressBarOptions options;
                options.include_frac = false;
                options.include_percent = true;

                std::string bar =
                  ANSI::ProgressBar(numer, denom, msg, total_time, options);

                status->Status(
                    "{}\n"
                    "{}\n"
                    "{}\n",
                    iter_histo.SimpleANSI(HISTO_LINES),
                    timing,
                    bar);
              });
          }

          lopt.Run(arginfos,
                   // number of calls
                   std::nullopt, std::nullopt,
                   // optimize for a minute at a time
                   {60.0},
                   // Stop if we succeed
                   {0});

          {
            MutexLock ml(&m);
            run_calls++;
          }

          const auto besto = lopt.GetBest();
          CHECK(besto.has_value()) << "We sampled a feasible arg to start.";
          const auto &[args, outer_loss] = besto.value();
          if (outer_loss == 0.0) {
            status->Print("Success!");
            // Success!
            MutexLock ml(&m);
            // If multiple threads finish at the same time, we save all
            // of them.
            should_die = true;

            std::optional<Polyhedron> opoly =
              PolyhedronFromConvexVertices(ToPoints(args), "unopt");
            CHECK(opoly.has_value()) << "Bug: These should be convex "
              "point sets by construction!";
            const Polyhedron &poly = opoly.value();

            successes++;

            status->Print(
                "\n\n" ABGCOLOR(200, 0, 200,
                                ANSI_DARK_GREY "***** "
                                ANSI_YELLOW "NOPERT"
                                ANSI_DARK_GREY " with "
                                ANSI_WHITE "{}"
                                ANSI_DARK_GREY " vertices *****")
                "\n", (int)poly.vertices.size());


            SolutionDB db;
            db.AddNopert(poly, SolutionDB::NOPERT_METHOD_UNOPT);
            delete poly.faces;
            return;
          }
        }
      });

  status->Print("Failed " ACYAN("unopt") " after {}\n",
                ANSI::Time(run_timer.Seconds()));
  SolutionDB db;
  db.AddNopertAttempt(num_points, attempts.Read(), iter_histo,
                      SolutionDB::NOPERT_METHOD_UNOPT);
}

static void DoAdversary(int64_t num_points) {
  const double TIME_LIMIT = 60.0 * 60.0;
  Timer run_timer;

  polyhedra.Reset();
  attempts.Reset();
  degenerate.Reset();

  // Start from a random polyhedron with n vertices.
  // Solve it. If we can't, we're done.
  // Otherwise, take the solution and make it invalid by moving
  // one inner point to the outer hull.
  // Normalize so that we don't grow without bound. Repeat.

  static constexpr int NUM_THREADS = 8;

  std::mutex m;
  std::vector<int> thread_turn(NUM_THREADS, 0);
  Periodically status_per(5.0);
  AutoHisto iter_histo;
  iter_histo.AddFlag(0.0);
  iter_histo.AddFlag(500.0);
  auto MaybeStatus = [&]() {
      if (status_per.ShouldRun()) {
        MutexLock ml(&m);
        double total_time = run_timer.Seconds();
        const int64_t polys = polyhedra.Read();
        double tps = polys / total_time;

        ANSI::ProgressBarOptions options;
        options.include_frac = false;
        options.include_percent = true;

        std::string timing = std::format("{:.4f} polys/s", tps);

        for (int i = 0; i < thread_turn.size(); i++) {
          AppendFormat(&timing, " {}", thread_turn[i]);
        }

        std::string msg =
          std::format(
              AYELLOW("{}") AWHITE("⋮") "  |  "
              APURPLE("{}") AWHITE("x") " "
              ABLUE("{}") AWHITE("∎") " "
              AORANGE("{}") AWHITE("⋆") " "
              AGREEN("{}") AWHITE("♚"),
              num_points,
              FormatNum(attempts.Read()),
              FormatNum(polys),
              hard.Read(),
              successes.Read());

        std::string bar =
          ANSI::ProgressBar(total_time, TIME_LIMIT,
                            msg, total_time, options);

        status->Status(
            "{}\n"
            "{}\n"
            "{}\n",
            iter_histo.SimpleANSI(HISTO_LINES),
            timing,
            bar);
      }
    };

  bool should_die = false;
  int MAX_TURNS = 1000;
  ParallelFan(
      NUM_THREADS,
      [&](int thread_idx) {
        ArcFour rc(std::format("adv.{}.{}.{}", thread_idx,
                               num_points, time(nullptr)));
        for (;;) {
          Polyhedron poly = RandomCyclicPolyhedron(&rc, num_points);

          for (int turn = 0; turn < MAX_TURNS; turn++) {
            frame3 outer_frame, inner_frame;
            polyhedra++;
            std::optional<int> iters =
              DoSolve(thread_idx,
                      &rc, poly, &outer_frame, &inner_frame);
            if (!iters.has_value()) {
              MutexLock ml(&m);
              SolutionDB db;
              db.AddNopert(poly, SolutionDB::NOPERT_METHOD_ADVERSARY);
              successes++;
              printf(AGREEN("Success!") "\n");
              should_die = true;
              return;
            } else {
              MutexLock ml(&m);
              if (should_die) return;
              if (run_timer.Seconds() > TIME_LIMIT) {
                should_die = true;
                return;
              }
              thread_turn[thread_idx] = turn;
              iter_histo.Observe(iters.value());
            }

            Polyhedron opoly = Rotate(poly, outer_frame);
            Mesh2D oshadow = Shadow(opoly);
            Polyhedron ipoly = Rotate(poly, inner_frame);
            Mesh2D ishadow = Shadow(ipoly);

            std::vector<int> ohull = QuickHull(oshadow.vertices);
            std::vector<int> ihull = QuickHull(ishadow.vertices);

            // Heuristically, it would be better to move a vertex that is
            // on the inner hull (thus certainly increasing the area of the
            // inner shadow) but not on the outer hull (not necessarily
            // increasing the area of the outer shadow).

            std::unordered_set<int> oset(ohull.begin(), ohull.end());
            std::vector<int> best;
            for (int i : ihull)
              if (!oset.contains(i))
                best.push_back(i);

            // TODO: Rather than picking randomly, we can:
            //  * use heuristics like closest
            //  * try several and see which one produces the hardest to solve?

            // The vertex we'll move.
            int idx = best.empty() ? ihull[RandTo(&rc, ihull.size())] :
              best[RandTo(&rc, best.size())];

            // Find the shortest distance from the vertex to the outer hull
            // in the 2D shadow.
            const auto &[o, d] = ClosestPointOnHull(oshadow.vertices, ohull,
                                                    ishadow.vertices[idx]);

            // The polyhedra have already been rotated, so we can just
            // extend it in the xy plane; this is the shortest vector
            // to the outer hull.

            // Note that this can make the polyhedron non-convex! That
            // isn't really a problem (we just use the convex hull) but
            // it may mean that we have effectively fewer vertices.
            //
            // We could fix it by never extending outside the existing
            // hull, or by only moving points on the surface of the
            // sphere, or something like that.

            constexpr double EPSILON = 0.00001;

            vec2 displacement(o.x - ipoly.vertices[idx].x,
                              o.y - ipoly.vertices[idx].y);

            vec2 extra_displacement = displacement * (1 + EPSILON);

            // XXX: Typically this does not really introduce an
            // unsolvable 2D shadow, since we can shift the inner
            // polyhedron away from this direction and still have a
            // solution (assuming epsilon clearance on the other
            // sides). We might consider doing this for three points
            // on the hull that are on "different sides."
            ipoly.vertices[idx].x += extra_displacement.x;
            ipoly.vertices[idx].y += extra_displacement.y;

            static constexpr int RENDER_FRAMES = 0;
            if (RENDER_FRAMES > 0 && thread_idx == 0) {
              MutexLock ml(&m);
              static int frames = 0;
              if (frames < RENDER_FRAMES) {
                Rendering rendering(ipoly, 1920, 1080);
                rendering.RenderMesh(oshadow);
                rendering.DarkenBG();

                // old
                rendering.RenderHull(ishadow, ihull, 0x6666FFAA);

                // new
                rendering.RenderHull(oshadow, ohull, 0xCCCCCCAA);

                Mesh2D nishadow = Shadow(ipoly);
                std::vector<int> nihull = QuickHull(nishadow.vertices);
                rendering.RenderHull(nishadow, nihull, 0x66FF66AA);

                std::string filename = std::format("adversary{}.png", frames);
                rendering.Save(filename, false);
                status->Print("Wrote " AGREEN("{}") "\n", filename);
                frames++;
              }
            }

            // Anyway, normalize. Recentering is important not just because
            // we've moved a vertex, but typically the inner polyhedron is
            // also translated!
            poly = NormalizeRadius(Recenter(ipoly));

            MaybeStatus();
          }
        }
      });

  status->Print("No nopert (adversarial, {} pts) after {}\n",
                num_points,
                ANSI::Time(run_timer.Seconds()));

  // Record the attempt, but fail.
  SolutionDB db;
  db.AddNopertAttempt(num_points, polyhedra.Read(), iter_histo,
                      SolutionDB::NOPERT_METHOD_ADVERSARY);
}

// Note this is sensitive to the order of the quad's vertices.
[[maybe_unused]]
static double NonPlanarity1(const vec3 &v0,
                            const vec3 &v1,
                            const vec3 &v2,
                            const vec3 &v3) {
  // The first three vertices define a plane.
  vec3 normal = yocto::normalize(yocto::cross(v1 - v0, v2 - v0));
  return std::abs(yocto::dot(v3 - v0, normal));
}

// Symmetric.
// Assumes non-degenerate quad.
static double NonPlanarity(const vec3 &v0,
                           const vec3 &v1,
                           const vec3 &v2,
                           const vec3 &v3) {
  // Two diagonals of the quad.
  vec3 diag1 = v2 - v0;
  vec3 diag2 = v3 - v1;

  vec3 connect = v1 - v0;

  vec3 normal = cross(diag1, diag2);

  double denom = length(normal);
  if (denom == 0.0) {
    return 0.0;
  }

  double numerator = std::abs(yocto::dot(connect, normal));
  return numerator / denom;
}

static void DoFlatten76() {
  SolutionDB db;
  Polyhedron source_poly = db.AnyPolyhedronByName("nopert_76");

  AutoHisto iter_histo;
  Timer run_time;
  int64_t notplanar = 0;

  // This polyhedron has 5-fold symmetry around the z axis,
  // and 4 points with different z coordinates. There should
  // be four with (x, y) close to (0, 1).

  const std::array<vec3, 4> canon_root1 = [&]() {
      std::vector<vec3> canon;
      for (const vec3 &v : source_poly.vertices) {
        const vec3 ref(0, 1, v.z);
        if (distance(v, ref) < 0.5) {
          canon.push_back(v);
        }
      }

      // Put vertices in ascending z order.
      std::sort(canon.begin(), canon.end(),
                [](const vec3 &a, const vec3 &b) {
                  return a.z < b.z;
                });

      std::array<vec3, 4> ret;
      memcpy(ret.data(), canon.data(), sizeof (vec3) * canon.size());
      return ret;
    }();
  CHECK(canon_root1.size() == 4);

  for (int i = 0; i < 4; i++) {
    printf("Canon Root %d: %s\n",
           i, VecString(canon_root1[i]).c_str());
  }

  // rotate 1/5
  const frame2 rot2d = rotation_frame2(std::numbers::pi * 2.0 / 5.0);

  auto Rootate = [&rot2d](const std::array<vec3, 4> &root) {
      std::array<vec3, 4> root2;
      for (int i = 0; i < 4; i++) {
        vec2 p = vec2(root[i].x, root[i].y);
        vec2 p2 = transform_point(rot2d, p);
        root2[i].x = p2.x;
        root2[i].y = p2.y;
        root2[i].z = root[i].z;
      }
      return root2;
    };

  const std::array<vec3, 4> canon_root2 = Rootate(canon_root1);

  // Now our goal is to place vertices nearby the original ones,
  // but such that they produce some rectangular faces instead
  // of triangular ones. Of course we still want it to be a
  // nopert!

  // Looking at the four vertices from top to bottom,

  //    a0---b0
  //     | / |
  //    a1---b1
  //     | / |
  //    a2---b2
  //     | \ |
  //    a3---b3

  // We always have an axis-aligned edge a-b because of the 5-fold
  // symmetry. a0-a1 (etc.) are not axis aligned. The question is
  // whether we can make e.g. the quad a0-b0-b1-a1 planar, or the
  // quad a2-b1-b0-a1. The quad might have been triangulated one way
  // or the other, but we don't care about that here because we will
  // reconstruct the 3D convex hull.

  // I should probably also consider quads that are 1+3 and 3+1
  // instead of 2+2. There is one of these in #76 that is nearly
  // planar.

  // There are also quads that cross the "root", but I don't try
  // flattening these. They are far from planar in the input and
  // flattening them symmetrically can easily lead to degeneracy.

  // We only care about solutions here where we have an actual rhombic
  // face; otherwise it is not an improvement over nopert_76. So we
  // select one of these quads and make its vertices planar, then test
  // the resulting shape.

  Periodically status_per(1.0);
  ArcFour rc(std::format("flatten-{}", time(nullptr)));
  double temp = 0.0;
  RandomGaussian gauss(&rc);
  int64_t solved = 0, successes = 0;
  for (;;) {
    std::array<vec3, 4> noise_root1;
    for (int i = 0; i < 4; i++) {
      noise_root1[i].x = canon_root1[i].x + gauss.Next() * temp;
      noise_root1[i].y = canon_root1[i].y + gauss.Next() * temp;
      noise_root1[i].z = canon_root1[i].z + gauss.Next() * temp;
    }

    // Which quad will we make planar? The only valid starting
    // points are vertices 0, 1, 2, since we will take it to be
    // the upper left corner of the quad.
    int v0 = RandTo(&rc, 3);
    // -1, 0, or 1. Whether we go up (-1), across (0), or down
    // when we move from the first row to the second. We then
    // always go down, then back to the first row with -delta.
    //
    // 0
    //
    // 1   1
    //   /
    // 2   2
    //   /
    // 3   3
    int delt = 0;
    switch (v0) {
    case 0:
      // across or down
      delt = RandTo(&rc, 1);
      break;
    case 1:
      // because we always go down, this is
      // the only general case.
      delt = RandTo(&rc, 2) - 1;
      break;
    case 2:
      // up or across
      delt = RandTo(&rc, 1) - 1;
      break;
    default:
      LOG(FATAL) << "Impossible.";
    }

    // v1 and v2 are in the second row.
    const int v1 = v0 + delt;
    const int v2 = v0 + delt + 1;
    const int v3 = v0 + delt + 1 - delt;
    // the span of points involved.
    int first_edit = std::min({v0, v1, v2, v3});
    int last_edit = std::max({v0, v1, v2, v3});
    // number of indices, inclusive
    int edit_size = last_edit - first_edit + 1;
    CHECK(first_edit < last_edit && (edit_size == 2 || edit_size == 3)) <<
      std::format("vs: {} {} {} {}, delt {}, f {} l {} s {}",
                  v0, v1, v2, v3, delt,
                  first_edit, last_edit, edit_size);

    // edit_size * 3 parameters.
    auto MakeRoot = [&noise_root1, first_edit, edit_size](
        std::span<const double> args) ->
      std::array<vec3, 4> {
      std::array<vec3, 4> root;
      memcpy(root.data(), noise_root1.data(), sizeof (vec3) * root.size());

      for (int i = 0; i < edit_size; i++) {
        root[first_edit + i].x += args[3 * i + 0];
        root[first_edit + i].y += args[3 * i + 1];
        root[first_edit + i].z += args[3 * i + 2];
      }

      return root;
    };

    // Use optimizer. We optimize over four parameters,
    // which are the x,y coordinates of the two involved
    // levels.
    auto Loss = [&](std::span<const double> args) {
        std::array<vec3, 4> root1 = MakeRoot(args);
        // column to the right
        std::array<vec3, 4> root2 = Rootate(root1);

        // Total squared deviation from the original vertices.
        double err = 0.0;
        err += distance_squared(canon_root1[v0], root1[v0]);
        err += distance_squared(canon_root2[v1], root2[v1]);
        err += distance_squared(canon_root2[v2], root2[v2]);
        err += distance_squared(canon_root1[v3], root1[v3]);

        // Now, check the planarity of the quad we're interested
        // in.
        const vec3 &q0 = root1[v0];
        const vec3 &q1 = root2[v1];
        const vec3 &q2 = root2[v2];
        const vec3 &q3 = root1[v3];

        err += NonPlanarity(q0, q1, q2, q3);
        return err;
      };

    // Minimize.
    const int n = edit_size * 3;
    // only move a little
    std::vector<double> lb(n, -0.1);
    std::vector<double> ub(n, +0.1);
    std::vector<double> best_args =
      Opt::Minimize(n, Loss, lb, ub, 2000).first;

    // Then make the polyhedron.
    std::vector<vec3> vertices;
    const auto broot = MakeRoot(best_args);
    for (const vec3 &v : broot) {
      vec2 p = vec2(v.x, v.y);
      for (int i = 0; i < 5; i++) {
        frame2 rot2d = rotation_frame2(
            (i * std::numbers::pi * 2.0) / 5.0);
        vec2 p2 = transform_point(rot2d, p);
        vertices.emplace_back(p2.x, p2.y, v.z);
      }
    }

    if (false) {
      DebugPointCloudAsSTL(vertices,
                           "flat76-debug.stl");
      CHECK(vertices.size() == 20);

      for (int i = 0; i < 4; i++) {
        printf("Root %d: %s\n",
               i, VecString(broot[i]).c_str());
      }
    }

    std::optional<Polyhedron> opoly =
      PolyhedronFromConvexVertices(vertices, "flat76");

    if (!opoly.has_value()) {
      degenerate++;
    } else {
      const Polyhedron &poly = opoly.value();

      // The top and bottom faces are always pentagons. The triangle strip
      // along the side is 6 triangles, so we have 6 * 5 + 2 = 32 faces if
      // the polyhedron's sides consist only of triangles.
      if (poly.faces->v.size() >= 32) {
        notplanar++;
      } else {
        if (solved % 1000 == 0) {
          std::string filename = std::format("flat76.{}.stl", solved);
          SaveAsSTL(poly, filename);
          status->Print("Wrote {}", filename);
        }

        // Multithreaded solve.
        if (auto io = ParallelSolve(12,
                                    &rc, poly,
                                    nullptr, nullptr)) {
          iter_histo.Observe(io.value());
          solved++;
        } else {
          // Nopert!
          SolutionDB db;
          db.AddNopert(poly, SolutionDB::NOPERT_METHOD_FLATTEN76);
          successes++;
          status->Print(AGREEN("Success!") "\n");
          if (successes > 5) {
            status->Print("That is enough noperts.\n");
            return;
          }
        }
      }
    }

    temp += 1e-100;
    temp *= 1.001;

    status_per.RunIf([&]() {

        std::string info =
          std::format("Flatten 76. " AYELLOW("{}") " solved. "
                      APURPLE("{}") ARED("🔥") " "
                      AORANGE("{}") " not planar. "
                      ARED("{}") AFGCOLOR(148, 75, 19, "💩") " ",
                      solved, temp, notplanar, degenerate.Read());

        std::string timing = ANSI::Time(run_time.Seconds());

        status->Status(
            "{}\n"
            "{}\n"
            "{}\n",
            iter_histo.SimpleANSI(HISTO_LINES),
            timing,
            info);
      });
  }

  LOG(FATAL) << "Unimplemented";
}

// Prep
static void Run(uint64_t parameter) {

  // static constexpr int METHOD = SolutionDB::NOPERT_METHOD_RANDOM;
  // static constexpr int METHOD = SolutionDB::NOPERT_METHOD_SYMMETRIC;
  // static constexpr int64_t MAX_SECONDS = 60 * 60;
  static constexpr int64_t MAX_SECONDS = 60 * 60;

  // static constexpr int METHOD = SolutionDB::NOPERT_METHOD_REDUCE_SC;
  // static constexpr int METHOD = SolutionDB::NOPERT_METHOD_RANDOM;
  // static constexpr int METHOD = SolutionDB::NOPERT_METHOD_CYCLIC;
  // static constexpr int METHOD = SolutionDB::NOPERT_METHOD_SYMMETRIC;
  // static constexpr int METHOD = SolutionDB::NOPERT_METHOD_ADVERSARY;
  // static constexpr int METHOD = SolutionDB::NOPERT_METHOD_RHOMBIC;
  // static constexpr int METHOD = SolutionDB::NOPERT_METHOD_UNOPT;
  // static constexpr int METHOD = SolutionDB::NOPERT_METHOD_SYMMETRIC;
static constexpr int METHOD = SolutionDB::NOPERT_METHOD_FLATTEN76;

  switch (METHOD) {
  case SolutionDB::NOPERT_METHOD_RANDOM:
  case SolutionDB::NOPERT_METHOD_CYCLIC:
  case SolutionDB::NOPERT_METHOD_SYMMETRIC:
  case SolutionDB::NOPERT_METHOD_RHOMBIC: {

    CHECK(parameter >= 4) << "Must have at least four vertices.";
    for (;;) {

      std::unique_ptr<CandidateMaker> candidates(
          new RandomCandidateMaker(
              parameter, METHOD, MAX_SECONDS));

      Nopert(candidates.get());
      parameter++;
      std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    break;
  }

  case SolutionDB::NOPERT_METHOD_FLATTEN76: {
    for (;;) {
      DoFlatten76();
    }
    break;
  }

  case SolutionDB::NOPERT_METHOD_REDUCE_SC: {
    constexpr int num_vertices = 24;
    // We do not need to try all bitmasks due to symmetry. Alas, the
    // vertices are just in some arbitrary order, so it's not easy to
    // figure out which masks are equivalent. However, for vertex-
    // transitive shapes, we can say without loss of generality that
    // one of the vertices (the msb) is always deleted. So we only
    // need n - 1 bits.
    constexpr int num_bits = num_vertices - 1;
    constexpr uint64_t reduce_end = 1 << num_bits;
    const char *reference_name = "snubcube";

    // XXX load from file!
    IntervalSet done(false);

    const std::string filename =
      ReduceCandidateMaker::Filename(METHOD, reference_name);
    std::string old = Util::ReadFile(filename);
    if (!old.empty()) {
      done = IntervalCoverUtil::ParseBool(old);
      status->Print("Loaded " AYELLOW("{}") "\n", filename);
    }

    // "done" because there's nothing to do.
    done.SplitRight(reduce_end, true);
    // Done on Startropics, 30 Dec 2024
    done.SetSpan(0, 3989000, true);

    uint64_t reduce_start = parameter;
    while (reduce_start < reduce_end) {
      IntervalSet::Span sp = done.GetPoint(reduce_start);
      if (sp.data == false)
        break;
      reduce_start = sp.end;
    }

    std::unique_ptr<CandidateMaker> candidates(
        new ReduceCandidateMaker(
            METHOD, &done,
            reduce_start, reduce_end,
            reference_name));
    Nopert(candidates.get());

    break;
  }

  case SolutionDB::NOPERT_METHOD_ADVERSARY:
    for (;;) {
      DoAdversary(parameter);
      parameter++;
      if (parameter > 24) parameter = 10;
    }
    break;

  case SolutionDB::NOPERT_METHOD_UNOPT:
    for (;;) {
      UnOpt(parameter);
      parameter++;
      if (parameter > 24) parameter = 10;
    }
    break;
  default:
    LOG(FATAL) << "Bad nopert method";
  }
}

int main(int argc, char **argv) {
  ANSI::Init();
  printf("\n");

  status = new StatusBar(HISTO_LINES + 3);

  int parameter = 6;

  if (argc == 2) {
    parameter = strtol(argv[1], nullptr, 10);
  }

  Run(parameter);

  printf("OK\n");
  return 0;
}
