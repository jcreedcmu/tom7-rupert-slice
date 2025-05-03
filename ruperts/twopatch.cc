
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <format>
#include <mutex>
#include <numbers>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "map-util.h"
#include "ansi.h"
#include "arcfour.h"
#include "atomic-util.h"
#include "big-polyhedra.h"
#include "bignum/big.h"
#include "bounds.h"
#include "image.h"
#include "nd-solutions.h"
#include "opt/opt.h"
#include "patches.h"
#include "periodically.h"
#include "polyhedra.h"
#include "randutil.h"
#include "status-bar.h"
#include "threadutil.h"
#include "timer.h"
#include "util.h"
#include "yocto_matht.h"

DECLARE_COUNTERS(sols_done);

// Look for solutions that involve the two specific patches.
// Plot them on both outer and inner patches.
using namespace yocto;

static constexpr int DIGITS = 24;

static constexpr int TARGET_SAMPLES = 1'000'000;

struct TwoPatch {
  static std::string Filename(uint64_t outer_code, uint64_t inner_code) {
    return std::format("{:x}-{:x}.nds", outer_code, inner_code);
  }

  TwoPatch(StatusBar *status,
           const BigPoly &big_poly,
           uint64_t outer_code, uint64_t inner_code,
           uint64_t outer_mask = 0, uint64_t inner_mask = 0) :
    status(status),
    boundaries(big_poly),
    small_poly(SmallPoly(big_poly)),
    outer_code(outer_code), inner_code(inner_code),
    outer_mask(outer_mask), inner_mask(inner_mask),
    sols(Filename(outer_code, inner_code)),
    diameter(Diameter(small_poly)) {

    if (outer_mask == 0) {
      outer_mask = GetCodeMask(boundaries, outer_code);
    }
    if (inner_mask == 0) {
      inner_mask = GetCodeMask(boundaries, inner_code);
    }
    CHECK(outer_mask != 0 && inner_mask != 0);

    // Just need to compute the hulls once.
    outer_hull = ComputeHullForPatch(boundaries, outer_code, outer_mask,
                                     {"twopatch-outer"});
    inner_hull = ComputeHullForPatch(boundaries, inner_code, inner_mask,
                                     {"twopatch-inner"});

    start_sols_size = sols.Size();
    {
      MutexLock ml(&mu);
      MaybeStatus();
    }
  }

  StatusBar *status = nullptr;
  Periodically status_per = Periodically(1);
  Periodically save_per = Periodically(60 * 5);

  std::mutex mu;
  bool should_die = false;
  const Boundaries boundaries;
  const Polyhedron small_poly;
  const uint64_t outer_code = 0, inner_code = 0;
  uint64_t outer_mask = 0, inner_mask = 0;
  double total_sample_sec = 0.0, total_opt_sec = 0.0, total_add_sec = 0.0;
  NDSolutions<6> sols;
  size_t start_sols_size = 0;
  const double diameter = 0.0;
  Timer run_timer;

  std::vector<int> outer_hull, inner_hull;


  void WorkThread(int thread_idx) {
    ArcFour rc(std::format("{}.{}", time(nullptr), thread_idx));

    for (;;) {
      {
        MutexLock ml(&mu);
        if (sols.Size() >= TARGET_SAMPLES) {
          should_die = true;
        }

        if (should_die)
          return;
      }

      Timer sample_timer;
      // Uniformly random view positions in each patch.
      const vec3 outer_view =
        GetVec3InPatch(&rc, boundaries, outer_code, outer_mask);
      const vec3 inner_view =
        GetVec3InPatch(&rc, boundaries, inner_code, inner_mask);
      double sample_sec = sample_timer.Seconds();

      const frame3 outer_frame = FrameFromViewPos(outer_view);
      const frame3 inner_frame = FrameFromViewPos(inner_view);

      // 2D convex polygons (the projected hulls in these view
      // positions).
      const std::vector<vec2> outer_poly =
        PlaceHull(outer_frame, outer_hull);
      const std::vector<vec2> inner_poly =
        PlaceHull(inner_frame, inner_hull);

      // We can precompute inscribed circle etc., although we
      // generally expect these hulls to be pretty close (we
      // have already removed the interior points). Probably
      // better would be to use the inscribed/circumscribed
      // circles to set bounds on the translation.



      // we rotate the inner polygon around zero by theta, and
      // translate it by dx,dy.
      auto Loss = [&outer_poly, &inner_poly](
          const std::array<double, 3> args) {
          const auto &[theta, dx, dy] = args;
          frame2 iframe = rotation_frame2(theta);
          iframe.o = {dx, dy};

          int outside = 0;
          double min_sqdistance = 1.0e30;
          for (const vec2 &v_in : inner_poly) {
            vec2 v_out = transform_point(iframe, v_in);

            // Is the out point in the hull? If not,
            // compute its distance.

            if (!PointInPolygon(v_out, outer_poly)) {
              outside++;
              min_sqdistance =
                std::min(min_sqdistance,
                         SquaredDistanceToPoly(outer_poly, v_out));
            }
          }

          double min_dist = sqrt(min_sqdistance);

          if (outside > 0 && min_dist == 0.0) [[unlikely]] {
            return outside / 1.0e12;
          } else {
            return min_dist;
          }
        };

      // PERF: With better bounds on the OD and ID of the
      // two hulls, we could limit the translation a lot more.
      const double TMAX = diameter;

      static constexpr int D = 3;
      const std::array<double, D> lb =
        {0.0, -TMAX, -TMAX};
      const std::array<double, D> ub =
        {2.0 * std::numbers::pi, +TMAX, +TMAX};

      Timer opt_timer;
      constexpr int ATTEMPTS = 100;
      const auto &[args, error] =
        Opt::Minimize<D>(Loss, lb, ub, 1000, 2, 100);
      [[maybe_unused]] const double opt_sec = opt_timer.Seconds();
      [[maybe_unused]] double aps = ATTEMPTS / opt_sec;

      Timer add_timer;
      // The "solution" is the same outer frame, but we need to
      // include the 2D rotation and translation in the innter frame.
      const auto &[theta, dx, dy] = args;
      const frame3 inner_sol_frame =
        translation_frame(vec3{dx, dy, 0}) *
        // around the z axis
        rotation_frame({0, 0, 1}, theta) *
        inner_frame;

      // solutions itself is synchronized.
      std::array<double, 6> key;
      key[0] = outer_view.x;
      key[1] = outer_view.y;
      key[2] = outer_view.z;
      key[3] = inner_view.x;
      key[4] = inner_view.y;
      key[5] = inner_view.z;
      sols.Add(key, error, outer_frame, inner_sol_frame);
      double add_sec = add_timer.Seconds();

      {
        MutexLock ml(&mu);
        total_sample_sec += sample_sec;
        total_opt_sec += opt_sec;
        total_add_sec += add_sec;
        MaybeStatus();
      }

      sols_done++;
    }
  }

  // With lock.
  void MaybeStatus() {
    status_per.RunIf([&]() {
        double tot_sec = total_sample_sec + total_opt_sec;
        std::string timing =
          std::format("{} sample {} opt ({:.2g}%) {} add. {} sols",
                      ANSI::Time(total_sample_sec),
                      ANSI::Time(total_opt_sec),
                      (100.0 * total_opt_sec) / tot_sec,
                      ANSI::Time(total_add_sec),
                      sols.Size());

        int64_t done = sols_done.Read();
        std::string bar =
          ANSI::ProgressBar(done, TARGET_SAMPLES - start_sols_size,
                            std::format("{} sols here. Save in {}",
                                        done,
                                        ANSI::Time(save_per.SecondsLeft())),
                            run_timer.Seconds());

        status->EmitStatus({timing, bar});
      });

    save_per.RunIf([&]() {
        sols.Save();
      });
  }

  std::vector<vec2> PlaceHull(const frame3 &frame,
                              const std::vector<int> &hull) const {
    std::vector<vec2> out;
    out.resize(hull.size());
    for (int hidx = 0; hidx < hull.size(); hidx++) {
      int vidx = hull[hidx];
      const vec3 &v_in = small_poly.vertices[vidx];
      // PERF: Don't need z coordinate.
      const vec3 v_out = transform_point(frame, v_in);
      out[hidx] = vec2{v_out.x, v_out.y};
    }
    return out;
  }

  void Solve() {
    sols_done.Reset();

    constexpr int NUM_THREADS = 2;
    std::vector<std::thread> threads;
    for (int i = 0; i < NUM_THREADS; i++) {
      threads.emplace_back(&TwoPatch::WorkThread, this, i);
    }

    for (;;) {
      {
        MutexLock ml(&mu);
        if (sols.Size() >= TARGET_SAMPLES)
          break;
      }
      std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    {
      MutexLock ml(&mu);
      should_die = true;
    }

    for (std::thread &t : threads) t.join();
    threads.clear();

    sols.Save();
    status->Printf("Done: Saved %lld sols.\n", (int64_t)sols.Size());
  }

};


int main(int argc, char **argv) {
  ANSI::Init();

  /*
  CHECK(argc == 3) << "./twopatch.exe outer_code inner_code\n";
  std::optional<uint64_t> outer_code = Util::ParseBinary(argv[1]);
  std::optional<uint64_t> inner_code = Util::ParseBinary(argv[2]);
  CHECK(outer_code.has_value() && inner_code.has_value());
  TwoPatch two_patch(BigScube(DIGITS),
                     outer_code.value(), inner_code.value());
  two_patch.Plot();
  */

  StatusBar status = StatusBar(2);

  PatchInfo patchinfo = LoadPatchInfo("scube-patchinfo.txt");
  status.Printf(
      "Total to run: %d * %d = %d\n",
      (int)patchinfo.canonical.size(),
      (int)patchinfo.canonical.size(),
      (int)(patchinfo.canonical.size() * patchinfo.canonical.size()));
  BigPoly scube = BigScube(DIGITS);

  ArcFour rc(std::format("{}", time(nullptr)));
  std::vector<std::pair<uint64_t, PatchInfo::CanonicalPatch>> cc =
    MapToSortedVec(patchinfo.canonical);
  Shuffle(&rc, &cc);

  for (int outer = 0; outer < cc.size(); outer++) {
    const auto &[code1, canon1] = cc[outer];
    for (int inner = 0; inner < cc.size(); inner++) {
      const auto &[code2, canon2] = cc[inner];
      TwoPatch two_patch(&status, scube,
                         code1, code2, canon1.mask, canon2.mask);
      two_patch.Solve();
    }
  }

  // done:
  // 0b0000101000010101110101111011111,0b1101110011101000001011100000101
  // 0b0000101000010101110101111011111,0b0000101000010101110101111011111,

  return 0;
}
