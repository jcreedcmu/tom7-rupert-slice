
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
#include <string>
#include <thread>
#include <vector>

#include "ansi.h"
#include "arcfour.h"
#include "big-polyhedra.h"
#include "bignum/big.h"
#include "bounds.h"
#include "image.h"
#include "nd-solutions.h"
#include "opt/opt.h"
#include "patches.h"
#include "periodically.h"
#include "polyhedra.h"
#include "rendering.h"
#include "status-bar.h"
#include "threadutil.h"
#include "timer.h"
#include "vector-util.h"
#include "yocto_matht.h"
#include "atomic-util.h"

DECLARE_COUNTERS(sols_done);

// Look for solutions that involve the two specific patches.
// Plot them on both outer and inner patches.
using namespace yocto;

static constexpr int DIGITS = 24;

struct TwoPatch {
  static std::string Filename(uint64_t outer_code, uint64_t inner_code) {
    return std::format("{:x}-{:x}.nds", outer_code, inner_code);
  }

  TwoPatch(const BigPoly &big_poly,
           uint64_t outer_code, uint64_t inner_code) :
    boundaries(big_poly),
    small_poly(SmallPoly(big_poly)),
    outer_code(outer_code), inner_code(inner_code),
    sols(Filename(outer_code, inner_code)) {

    outer_mask = GetCodeMask(boundaries, outer_code);
    inner_mask = GetCodeMask(boundaries, inner_code);

    // Just need to compute the hulls once.
    outer_hull = ComputeHull(outer_code, outer_mask);
    inner_hull = ComputeHull(inner_code, inner_mask);

    {
      MutexLock ml(&mu);
      MaybeStatus();
    }
  }

  StatusBar status = StatusBar(2);
  Periodically status_per = Periodically(1);
  Periodically save_per = Periodically(60 * 5);

  std::mutex mu;
  bool should_die = false;
  const Boundaries boundaries;
  const Polyhedron small_poly;
  const uint64_t outer_code = 0, inner_code = 0;
  uint64_t outer_mask = 0, inner_mask = 0;
  double total_sample_sec = 0.0, total_opt_sec = 0.0;

  std::vector<int> outer_hull, inner_hull;

  NDSolutions<6> sols;

  void WorkThread(int thread_idx) {
    ArcFour rc(std::format("{}.{}", time(nullptr), thread_idx));

    for (;;) {
      {
        MutexLock ml(&mu);
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

      // XXX get correct translation bounds from polys
      static constexpr int D = 3;
      const std::array<double, D> lb =
        {0.0, -0.15, -0.15};
      const std::array<double, D> ub =
        {2.0 * std::numbers::pi, +0.15, +0.15};

      Timer opt_timer;
      constexpr int ATTEMPTS = 100;
      const auto &[args, error] =
        Opt::Minimize<D>(Loss, lb, ub, 1000, 2, 100);
      [[maybe_unused]] const double opt_sec = opt_timer.Seconds();
      double aps = ATTEMPTS / opt_sec;

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

      {
        MutexLock ml(&mu);
        total_sample_sec += sample_sec;
        total_opt_sec += opt_sec;
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
          std::format("{} sample {} opt ({:.2}%)",
                      ANSI::Time(total_sample_sec),
                      ANSI::Time(total_opt_sec),
                      (100.0 * total_opt_sec) / tot_sec);

        std::string counts =
          std::format("{} sols done.", sols_done.Read());

        status.EmitStatus({timing, counts});
      });

    save_per.RunIf([&]() {
        sols.Save();
      });
  }

  static constexpr bool RENDER_HULL = true;

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

  // Get the points on the hull when in this view patch. Exact.
  // Clockwise winding order.
  std::vector<int> ComputeHull(uint64_t code,
                               uint64_t mask) const {
    BigQuat example_view = GetBigQuatInPatch(boundaries, code, mask);
    BigFrame frame = NonUnitRotationFrame(example_view);

    BigMesh2D full_shadow = Shadow(Rotate(frame, boundaries.big_poly));
    std::vector<int> hull = BigQuickHull(full_shadow.vertices);
    CHECK(hull.size() >= 3);

    if (RENDER_HULL) {
      Rendering rendering(SmallPoly(boundaries.big_poly), 1920, 1080);
      auto small_shadow = SmallMesh(full_shadow);
      rendering.RenderMesh(small_shadow);
      rendering.RenderHull(small_shadow, hull);
      rendering.Save(std::format("twopatch-hull-{:b}.png", code));
    }

    BigRat area = SignedAreaOfHull(full_shadow, hull);
    printf("Area for example hull: %.17g\n", area.ToDouble());

    if (BigRat::Sign(area) == -1) {
      VectorReverse(&hull);
      area = SignedAreaOfHull(full_shadow, hull);
      printf("Reversed hull to get area: %.17g\n", area.ToDouble());
    }

    CHECK(BigRat::Sign(area) == 1);

    if (RENDER_HULL) {
      std::vector<vec2> phull = PlaceHull(SmallFrame(frame), hull);

      Bounds bounds;
      for (const auto &vec : phull) {
        bounds.Bound(vec.x, vec.y);
      }
      bounds.AddMarginFrac(0.05);

      ImageRGBA ref(1920, 1080);
      ref.Clear32(0x000000FF);
      Bounds::Scaler scaler =
        bounds.ScaleToFit(ref.Width(), ref.Height()).FlipY();
      for (int i = 0; i < phull.size(); i++) {
        const vec2 &v0 = phull[i];
        const vec2 &v1 = phull[(i + 1) % phull.size()];
        const auto &[x0, y0] = scaler.Scale(v0.x, v0.y);
        const auto &[x1, y1] = scaler.Scale(v1.x, v1.y);
        ref.BlendLine32(x0, y0, x1, y1, 0xFFFFFFAA);
      }
      ref.Save(std::format("twopatch-phull-{:b}.png", code));
    }

    return hull;
  }

  void Plot() {
    constexpr int NUM_THREADS = 8;
    std::vector<std::thread> threads;
    for (int i = 0; i < NUM_THREADS; i++) {
      threads.emplace_back(&TwoPatch::WorkThread, this, i);
    }

    while (sols.Size() < 10'000'000) {
      std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    {
      MutexLock ml(&mu);
      should_die = true;
    }

    for (std::thread &t : threads) t.join();
    threads.clear();
  }

};



int main(int argc, char **argv) {
  ANSI::Init();

  TwoPatch two_patch(BigScube(DIGITS),
                     0b0000101000010101110101111011111,
                     0b1101110011101000001011100000101);
  two_patch.Plot();

  return 0;
}
