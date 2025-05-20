
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <format>
#include <limits>
#include <mutex>
#include <numbers>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

#include "ansi.h"
#include "arcfour.h"
#include "atomic-util.h"
#include "auto-histo.h"
#include "base/stringprintf.h"
#include "big-polyhedra.h"
#include "bounds.h"
#include "hashing.h"
#include "image.h"
#include "map-util.h"
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

DECLARE_COUNTERS(rounds_done);

// Look for solutions that involve the two specific patches.
// Plot them on both outer and inner patches.
using namespace yocto;

static constexpr int NUM_THREADS = 8;
// static constexpr int NUM_THREADS = 1;

static constexpr int DIGITS = 24;

static constexpr int TARGET_SAMPLES = 1'000'000;

// XXX Specific to snub cube.
[[maybe_unused]]
static constexpr int TOTAL_PATCHES = 36;
static constexpr int TOTAL_PAIRS = TOTAL_PATCHES * TOTAL_PATCHES;

static constexpr std::string_view PATCH_INFO_FILE =
  "scube-patchinfo.txt";

std::string PolyString(const std::vector<vec2> &poly) {
  std::string ret;
  for (const vec2 &v : poly) {
    AppendFormat(&ret, "  ({:.17g}, {:.17g})\n", v.x, v.y);
  }
  return ret;
}

struct HowManyRep {
  static constexpr int ATTEMPTS = 200;
  static constexpr int TARGET_ROUNDS = 25000;

  HowManyRep(StatusBar *status,
             const BigPoly &big_poly,
             uint64_t outer_code, uint64_t inner_code,
             uint64_t outer_mask_arg = 0, uint64_t inner_mask_arg = 0) :
    status(status),
    boundaries(big_poly),
    small_poly(SmallPoly(big_poly)),
    outer_code(outer_code), inner_code(inner_code),
    outer_mask(outer_mask_arg), inner_mask(inner_mask_arg),
    diameter(Diameter(small_poly)) {

    if (outer_mask == 0) {
      outer_mask = GetCodeMask(boundaries, outer_code);
    }
    if (inner_mask == 0) {
      inner_mask = GetCodeMask(boundaries, inner_code);
    }
    CHECK(outer_mask != 0 && inner_mask != 0);

    // Just need to compute the hulls once.
    outer_hull = ComputeHullForPatch(boundaries, outer_code, outer_mask, {});
    inner_hull = ComputeHullForPatch(boundaries, inner_code, inner_mask, {});

    /*
    for (int i = 0; i < ATTEMPTS; i++) {
      loss_histos.emplace_back(100000);
    }
    */
    loss_histos.resize(ATTEMPTS);

    {
      MutexLock ml(&mu);
      MaybeStatus();
    }
  }

  StatusBar *status = nullptr;
  Periodically status_per = Periodically(1);
  Periodically save_per = Periodically(60);

  std::mutex mu;
  bool should_die = false;
  const Boundaries boundaries;
  const Polyhedron small_poly;
  const uint64_t outer_code = 0, inner_code = 0;
  uint64_t outer_mask = 0, inner_mask = 0;
  const double diameter = 0.0;
  Timer run_timer;

  // std::vector<AutoHisto> loss_histos;
  std::vector<std::vector<double>> loss_histos;

  std::vector<int> outer_hull, inner_hull;

  void WorkThread(int thread_idx) {
    ArcFour rc(std::format("{}.{}", time(nullptr), thread_idx));

    for (;;) {
      {
        MutexLock ml(&mu);

        if (should_die)
          return;
      }

      // Uniformly random view positions in each patch.
      const vec3 outer_view =
        GetVec3InPatch(&rc, boundaries, outer_code, outer_mask);
      const vec3 inner_view =
        GetVec3InPatch(&rc, boundaries, inner_code, inner_mask);

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

      PolyTester2D outer_tester(outer_poly);
      CHECK(outer_tester.IsInside(vec2{0, 0}));

      // we rotate the inner polygon around the origin by theta, and
      // translate it by dx,dy.
      //
      // The loss function finds the maximum distance among the points
      // that are outside the outer poly. Overall, we're trying to
      // minimize that difference.
      auto Loss = [&outer_tester, &inner_poly](
          const std::array<double, 3> &args) {
          const auto &[theta, dx, dy] = args;
          frame2 iframe = rotation_frame2(theta);
          iframe.o = {dx, dy};

          int outside = 0;
          double max_sqdistance = 0.0;
          for (const vec2 &v_in : inner_poly) {
            vec2 pt = transform_point(iframe, v_in);

            // Is the transformed point in the hull? If not,
            // compute its distance.

            std::optional<double> osqdist =
              outer_tester.SquaredDistanceOutside(pt);

            if (osqdist.has_value()) {
              outside++;
              max_sqdistance = std::max(max_sqdistance, osqdist.value());
            }
          }

          double min_dist = sqrt(max_sqdistance);

          if (outside > 0 && min_dist == 0.0) [[unlikely]] {
            return outside / 1.0e16;
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

      double error = std::numeric_limits<double>::infinity();

      std::vector<double> error_at;
      error_at.reserve(ATTEMPTS);

      for (int att = 0; att < ATTEMPTS; att++) {
        const auto &[a_args, a_error] =
          Opt::Minimize<D>(Loss, lb, ub, 1000, 2, 1, Rand32(&rc));

        if (a_error < error) {
          error = a_error;
        }

        // We are interested in the minimum error after this
        // number of attempts.
        error_at.push_back(error);
      }


      {
        MutexLock ml(&mu);
        for (int a = 0; a < error_at.size(); a++) {
          // Absolute loss.
          loss_histos[a].push_back(error_at[a] - error_at.back());
          // Relative to final error. It'll always be >= 1.
          // loss_histos[a].push_back(error_at[a] / error_at.back());
        }
        MaybeStatus();
      }

      rounds_done++;
    }
  }

  // With lock.
  void MaybeStatus() {
    status_per.RunIf([&]() {
        double wall_sec = run_timer.Seconds();
        int64_t numer = rounds_done.Read();
        int64_t denom = TARGET_ROUNDS;

        std::string oline =
          std::format(AWHITE("outer") " {:016x} = {}",
                       outer_code,
                       boundaries.ColorMaskedBits(outer_code, outer_mask));
        std::string iline =
          std::format(AWHITE("inner") " {:016x} = {}",
                      inner_code,
                      boundaries.ColorMaskedBits(inner_code, inner_mask));

        std::string bar =
          ANSI::ProgressBar(numer, denom,
                            "Running.",
                            wall_sec);

        status->EmitStatus({
            AWHITE("———————————————————————————"),
            oline, iline, bar});
      });

    save_per.RunIf([&]() {
        Render();
      });
  }

  std::vector<vec2> PlaceHull(const frame3 &frame,
                              const std::vector<int> &hull) const {
    std::vector<vec2> out;
    out.resize(hull.size());
    for (int hidx = 0; hidx < hull.size(); hidx++) {
      int vidx = hull[hidx];
      const vec3 &v_in = small_poly.vertices[vidx];
      const vec3 v_out = transform_point(frame, v_in);
      out[hidx] = vec2{v_out.x, v_out.y};
    }
    return out;
  }

  void Solve() {
    std::vector<std::thread> threads;
    for (int i = 0; i < NUM_THREADS; i++) {
      threads.emplace_back(&HowManyRep::WorkThread, this, i);
    }

    for (;;) {
      {
        MutexLock ml(&mu);
        if (rounds_done.Read() >= TARGET_ROUNDS)
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

    Render();
  }

  // With lock.
  void Render() {
    const int PW = 1;
    ImageRGBA img(PW * ATTEMPTS, 2048);
    img.Clear32(0x000000FF);
    Bounds bounds;
    bounds.Bound(0, 0);
    // bounds.Bound(0, 2.0);
    for (int a = 6; a < ATTEMPTS; a++) {
      for (double r : loss_histos[a]) {
        // r = std::min(r, 2.0);
        bounds.Bound(a, r);
      }
    }

    Bounds::Scaler scaler = bounds.Stretch(img.Width(), img.Height()).FlipY();

    int64_t samples = 0;
    for (int a = 0; a < ATTEMPTS; a++) {
      for (double r : loss_histos[a]) {
        r = std::min(r, 2.0);
        const auto &[sx, sy] = scaler.Scale(a, r);
        // for (int dx = 0; dx < PW; dx++) {
        int x = std::clamp((int)std::round(sx), 0, img.Width() - 1);
        int y = std::clamp((int)std::round(sy), 0, img.Height() - 1);
        img.BlendPixel32(x, y, 0xFFFFFF44);
      }
      samples += loss_histos[a].size();
    }

    img.BlendText32(0, 0, 0x00FF00AA,
                    std::format("max {:.11g}", bounds.MaxY()));

    img.Save("howmanyrep.png");
    status->Printf("Wrote howmanyrep.png. %lld samples. Max %.11g",
                   samples, bounds.MaxY());
  }

};

int main(int argc, char **argv) {
  ANSI::Init();

  PatchInfo patchinfo = LoadPatchInfo(PATCH_INFO_FILE);
  BigPoly scube = BigScube(DIGITS);

  ArcFour rc(std::format("{}", time(nullptr)));
  std::vector<std::pair<uint64_t, PatchInfo::CanonicalPatch>> cc =
    MapToSortedVec(patchinfo.canonical);
  // Shuffle(&rc, &cc);

  CHECK(cc.size() >= 2);

  const auto &[code1, canon1] = cc[0];
  const auto &[code2, canon2] = cc[1];

  StatusBar status(4);
  HowManyRep how_many(&status, scube, code1, code2,
                      canon1.mask, canon2.mask);
  how_many.Solve();

  return 0;
}
