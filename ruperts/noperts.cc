#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <format>
#include <functional>
#include <memory>
#include <mutex>
#include <numbers>
#include <optional>
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
#include "hull.h"
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

// Try to find counterexamples.

#define ABLOOD(s) AFGCOLOR(148, 0, 0, s)

DECLARE_COUNTERS(polyhedra, attempts, degenerate, skipped, hard, successes);

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

static vec3 RandomVec(ArcFour *rc) {
  return vec3(2.0 * RandDouble(rc) - 1.0,
              2.0 * RandDouble(rc) - 1.0,
              2.0 * RandDouble(rc) - 1.0);
}

// Return the number of iterations taken, or nullopt if we exceeded
// the limit. If solved and the arguments are non-null, sets the outer
// frame and inner frame to some solution.
//
// Note:
static constexpr int NOPERT_ITERS = 200000;
static constexpr int MIN_VERBOSE_ITERS = 5000;
static constexpr bool SAVE_HARD = false;
static std::optional<int> TrySolve(int thread_idx,
                                   ArcFour *rc, const Polyhedron &poly,
                                   frame3 *outer_frame_out,
                                   frame3 *inner_frame_out) {
  CHECK(!poly.faces->v.empty());

  for (int iter = 0; iter < NOPERT_ITERS; iter++) {

    if (iter > 0 && (iter % 5000) == 0) {
      status->Printf("[" APURPLE("%d") "] %d-point polyhedron "
                     AFGCOLOR(190, 220, 190, "not solved")
                     " after " AWHITE("%d") " iters...\n",
                     thread_idx,
                     (int)poly.vertices.size(), iter);

      if (iter == 20000) {
        hard++;
        if (SAVE_HARD) {
          std::string filename = StringPrintf("hard.%lld.%d.stl",
                                              time(nullptr), thread_idx);
          SaveAsSTL(poly, filename);
          status->Printf("[" APURPLE("%d") "] Hard %d-point polyhedron saved "
                         "to " AGREEN("%s") "\n",
                         thread_idx, (int)poly.vertices.size(),
                         filename.c_str());
        }
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
        status->Printf("%d-point polyhedron " AYELLOW("solved") " after "
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
      PolyhedronFromConvexVertices(std::move(pts2), "random");
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
          PointSet3 pointset;

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
            status->Printf("Done with num points = " AYELLOW("%d") "\n",
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
      PolyhedronFromConvexVertices(std::move(points), "randomsymmetric");
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
    // axes of symmetry.

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
    rc(StringPrintf("rand.%d.%lld", method, time(nullptr))),
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
    return StringPrintf(APURPLE("%d") AWHITE("⋮"), num_points);
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
    name = StringPrintf("reduce_%s", reference_name);
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
    return StringPrintf(
        APURPLE("%llu") AGREY("-") APURPLE("%llu"),
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
    status->Printf("Wrote " AGREEN("%s") "\n", filename.c_str());
  }

  static std::string Filename(int method, const char *reference_name) {
    return StringPrintf("noperts-reduce-%d-%s.txt", method, reference_name);
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
        ArcFour rc(StringPrintf("noperts.%d.%lld\n",
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
            status->Printf("No more polyhedra after %s\n",
                           ANSI::Time(timer.Seconds()).c_str());
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
            r.Save(StringPrintf("poly.%d.png", count++));
          }

          Timer solve_timer;
          std::optional<int> iters = TrySolve(thread_idx, &rc, poly,
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

            std::string info = candidates->Info();
            const auto &[numer, denom] = candidates->Frac();

            ANSI::ProgressBarOptions options;
            options.include_frac = false;
            options.include_percent = true;

            std::string timing =
              StringPrintf(
                  "Timing: [" AWHITE("%.1f") "/s] %s " APURPLE("gen")
                  " %s " ABLUE("sol"),
                  tps,
                  ANSI::Time(total_gen_sec).c_str(),
                  ANSI::Time(total_solve_sec).c_str());

            std::string msg =
              StringPrintf(
                  ACYAN("%s") " %s%s" AGREY("|") " "
                  ARED("%s") ABLOOD("×") " "
                  ABLUE("%s") AWHITE("∎"),
                  name.c_str(),
                  info.c_str(), info.empty() ? "" : " ",
                  FormatNum(degenerate.Read()).c_str(),
                  FormatNum(polys).c_str());

            std::string bar =
              ANSI::ProgressBar(numer, denom,
                                msg, total_time, options);

            status->Statusf(
                "%s\n"
                "%s\n"
                "%s\n",
                histo.SimpleANSI(HISTO_LINES).c_str(),
                timing.c_str(),
                bar.c_str());
          }
        }
      });

  return success;
}

static void UnOpt(int64_t num_points) {
  attempts.Reset();

  constexpr double MAX_SECONDS = 60.0 * 60.0;

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
        ArcFour rc(std::format("adv.{}.{}.{}", thread_idx,
                               num_points, time(nullptr)));

        // Here we have nested optimization. The outer optimizer is trying to
        // set vertices such that the inner optimizer takes many iterations to
        // find solutions.

        // Each point is *three* parameters so we use the large optimizer
        // for this. No need for a cache since everything is a double.
        using LargeOpt = LargeOptimizer<false>;

        auto ToPoints = [num_points](const std::vector<double> &args) {
            std::vector<vec3> points;
            CHECK(args.size() == num_points * 3);
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

        auto OuterLoss = [num_points, &ToPoints, &rc, &m, &iter_histo](
            const std::vector<double> &args) -> LargeOpt::return_type {
            std::vector<vec3> points = ToPoints(args);
            CHECK(points.size() == num_points);

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

              // Not feasible.
              /*
              status->Printf("Infeasible (%d pts; %d hull) with badness %f",
                             (int)points.size(), (int)hull.size(),
                             badness);
              */
              return std::pair(badness, false);
            }

            // Otherwise, we have a convex polyhedron.
            std::optional<Polyhedron> opoly = PolyhedronFromConvexVertices(
                std::move(points), "unopt");

            if (!opoly.has_value()) {
              // This can happen if the hull has degenerate faces (colinear points?).
              // status->Printf("wasn't convex??");
              degenerate++;
              return LargeOpt::INFEASIBLE;
            }

            const Polyhedron &poly = opoly.value();
            polyhedra++;
            std::optional<int> oiters = TrySolve(0, &rc, poly,
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
              status->Printf("Feasible! %d iters\n", iters);
              */
              return std::make_pair(NOPERT_ITERS - iters, true);
            }
          };

        // θ, φ, ρ triples
        std::vector<LargeOpt::arginfo> arginfos;
        for (int i = 0; i < num_points; i++) {
          // polar angle
          arginfos.push_back(LargeOpt::Double(0.0, std::numbers::pi));
          // azimuth angle
          arginfos.push_back(LargeOpt::Double(0.0, 2.0 * std::numbers::pi));
          // length
          arginfos.push_back(LargeOpt::Double(0.000001, std::numbers::sqrt2));
        }

        LargeOpt lopt(OuterLoss, num_points * 3, Rand64(&rc));

        // Need a feasible input to start. Anything is feasible as long as
        // it's a convex polyhedron, so use a random cyclic one.
        std::vector<double> start;
        for (int i = 0; i < num_points; i++) {
          // We can easily generate such polyhedra using spherical coordinates,
          // by just setting length = 1.
          start.push_back(RandDouble(&rc) * std::numbers::pi);
          start.push_back(RandDouble(&rc) * std::numbers::pi * 2.0);
          start.push_back(1.0);
        }

        lopt.Sample(start);

        for (;;) {
          {
            MutexLock ml(&m);
            if (should_die) return;
            const double total_time = run_timer.Seconds();
            if (total_time > MAX_SECONDS) {
              should_die = true;
              return;
            }

            status_per.RunIf([&]{
                std::string timing = "TODO: Timing";
                double numer = total_time;
                double denom = MAX_SECONDS;

                std::string msg =
                  StringPrintf(
                      ACYAN("unopt") " " AWHITE("%d") " " AGREY("|") " "
                      ARED("%s") ABLOOD("×") " "
                      ABLUE("%s") AWHITE("∎") " "
                      AGREEN("%s") AWHITE("♚") " "
                      APURPLE("%s") AWHITE("∳"),
                      num_points,
                      FormatNum(degenerate.Read()).c_str(),
                      FormatNum(polyhedra.Read()).c_str(),
                      FormatNum(successes.Read()).c_str(),
                      FormatNum(run_calls).c_str());

                ANSI::ProgressBarOptions options;
                options.include_frac = false;
                options.include_percent = true;

                std::string bar =
                  ANSI::ProgressBar(numer, denom, msg, total_time, options);

                status->Statusf(
                    "%s\n"
                    "%s\n"
                    "%s\n",
                    iter_histo.SimpleANSI(HISTO_LINES).c_str(),
                    timing.c_str(),
                    bar.c_str());
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
            status->Printf("Success!");
            // Success!
            MutexLock ml(&m);
            // If multiple threads finish at the same time, we save all
            // of them.
            should_die = true;

            // Convert to
            std::optional<Polyhedron> opoly =
              PolyhedronFromConvexVertices(ToPoints(args), "unopt");
            CHECK(opoly.has_value()) << "Bug: These should be convex point sets "
              "by construction!";
            const Polyhedron &poly = opoly.value();

            successes++;

            status->Printf(
                "\n\n" ABGCOLOR(200, 0, 200,
                                ANSI_DARK_GREY "***** "
                                ANSI_YELLOW "NOPERT"
                                ANSI_DARK_GREY " with "
                                ANSI_WHITE "%d"
                                ANSI_DARK_GREY " vertices *****")
                "\n", (int)poly.vertices.size());


            SolutionDB db;
            db.AddNopert(poly, SolutionDB::NOPERT_METHOD_UNOPT);
            return;
          }
        }
      });

  status->Printf("Failed " ACYAN("unopt") " after %s\n",
                 ANSI::Time(run_timer.Seconds()).c_str());
  SolutionDB db;
  db.AddNopertAttempt(num_points, attempts.Read(), iter_histo,
                      SolutionDB::NOPERT_METHOD_UNOPT);
}

static void DoAdversary(int64_t num_points) {
  const double TIME_LIMIT = 60.0 * 60.0;
  Timer run_timer;

  polyhedra.Reset();

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

        std::string timing = StringPrintf("%.4f polys/s", tps);

        for (int i = 0; i < thread_turn.size(); i++) {
          AppendFormat(&timing, " {}", thread_turn[i]);
        }

        std::string msg =
          StringPrintf(
              AYELLOW("%lld") AWHITE("⋮") "  |  "
              APURPLE("%s") AWHITE("x") " "
              ABLUE("%s") AWHITE("∎") " "
              AORANGE("%lld") AWHITE("⋆") " "
              AGREEN("%lld") AWHITE("♚"),
              num_points,
              FormatNum(attempts.Read()).c_str(),
              FormatNum(polys).c_str(),
              hard.Read(),
              successes.Read());

        std::string bar =
          ANSI::ProgressBar(total_time, TIME_LIMIT,
                            msg, total_time, options);

        status->Statusf(
            "%s\n"
            "%s\n"
            "%s\n",
            iter_histo.SimpleANSI(HISTO_LINES).c_str(),
            timing.c_str(),
            bar.c_str());
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
            std::optional<int> iters = TrySolve(thread_idx,
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
                status->Printf("Wrote " AGREEN("%s") "\n", filename.c_str());
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

  status->Printf("No nopert (adversarial, %d pts) after %s\n",
                 num_points,
                 ANSI::Time(run_timer.Seconds()).c_str());

  // Record the attempt, but fail.
  SolutionDB db;
  db.AddNopertAttempt(num_points, polyhedra.Read(), iter_histo,
                      SolutionDB::NOPERT_METHOD_ADVERSARY);
}

// Prep
static void Run(uint64_t parameter) {

  // static constexpr int METHOD = SolutionDB::NOPERT_METHOD_RANDOM;
  // static constexpr int METHOD = SolutionDB::NOPERT_METHOD_SYMMETRIC;
  static constexpr int64_t MAX_SECONDS = 60 * 60;

  // static constexpr int METHOD = SolutionDB::NOPERT_METHOD_REDUCE_SC;
  // static constexpr int METHOD = SolutionDB::NOPERT_METHOD_RANDOM;
  // static constexpr int METHOD = SolutionDB::NOPERT_METHOD_SYMMETRIC;
  // static constexpr int METHOD = SolutionDB::NOPERT_METHOD_ADVERSARY;
  // static constexpr int METHOD = SolutionDB::NOPERT_METHOD_RHOMBIC;
  static constexpr int METHOD = SolutionDB::NOPERT_METHOD_UNOPT;

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
      status->Printf("Loaded " AYELLOW("%s") "\n", filename.c_str());
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
