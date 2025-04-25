
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <format>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "ansi.h"
#include "arcfour.h"
#include "base/logging.h"
#include "base/stringprintf.h"
#include "big-polyhedra.h"
#include "bignum/big.h"
#include "bounds.h"
#include "color-util.h"
#include "image.h"
#include "patches.h"
#include "periodically.h"
#include "polyhedra.h"
#include "randutil.h"
#include "rendering.h"
#include "run-z3.h"
#include "status-bar.h"
#include "threadutil.h"
#include "timer.h"
#include "util.h"
#include "vector-util.h"
#include "yocto_matht.h"
#include "z3.h"

static constexpr int DIGITS = 24;

// Visualize a patch under its parameterization. This is
// using doubles and samples.
static void PlotPatch(const Boundaries &boundaries,
                      BigVec3 &bigv,
                      uint64_t code, uint64_t mask) {
  Timer timer;
  uint64_t example_code = boundaries.GetCode(bigv);
  CHECK(example_code == code);
  std::string code_string = std::format("{:b}", code);
  Polyhedron small_poly = SmallPoly(boundaries.big_poly);

  const vec3 v = normalize(SmallVec(bigv));

  const std::vector<int> hull = [&]() {
      frame3 frame = FrameFromViewPos(v);
      Mesh2D shadow = Shadow(Rotate(small_poly, frame));
      return QuickHull(shadow.vertices);
    }();

  // Generate vectors that have the same code.
  const int NUM_SAMPLES = 10000;
  ArcFour rc(std::format("plot_{}", code));
  StatusBar status(1);

  int64_t hits = 0, attempts = 0;
  Periodically status_per(5.0);
  std::vector<vec3> samples;
  samples.reserve(NUM_SAMPLES);
  while (samples.size() < NUM_SAMPLES) {
    attempts++;

    vec3 s = GetVec3InPatch(&rc, boundaries, code, mask);
    samples.push_back(s);
    hits++;

    #if 0
    vec3 s;
    std::tie(s.x, s.y, s.z) = RandomUnit3D(&rc);
    BigVec3 bs(BigRat::ApproxDouble(s.x, 1000000),
               BigRat::ApproxDouble(s.y, 1000000),
               BigRat::ApproxDouble(s.z, 1000000));

    // Try all the signs. At most one of these will be
    // in the patch, but this should increase the
    // efficiency (because knowing that other signs
    // are *not* in the patch increases the chance
    // that we are).
    for (uint8_t b = 0b000; b < 0b1000; b++) {
      BigVec3 bbs((b & 0b100) ? -bs.x : bs.x,
                  (b & 0b010) ? -bs.y : bs.y,
                  (b & 0b001) ? -bs.z : bs.z);
      uint64_t sample_code = boundaries.GetCode(bbs);
      if (sample_code == code) {
        // Sample is in range.
        samples.push_back(
            vec3((b & 0b100) ? -s.x : s.x,
                 (b & 0b010) ? -s.y : s.y,
                 (b & 0b001) ? -s.z : s.z));
        hits++;
        break;
      }
    }
    #endif

    status_per.RunIf([&]() {
        status.Progressf(samples.size(), NUM_SAMPLES,
                         ACYAN("%s") " (%.3f%% eff)",
                         code_string.c_str(),
                         (hits * 100.0) / attempts);
      });
  }

  // separate bounds for each parameter. Just using X dimension.
  Bounds rbounds;
  Bounds gbounds;
  Bounds bbounds;

  Bounds bounds;
  // First pass to compute bounds.
  for (const vec3 &s : samples) {
    rbounds.BoundX(s.x);
    gbounds.BoundX(s.y);
    bbounds.BoundX(s.z);

    frame3 frame = FrameFromViewPos(s);
    for (int vidx : hull) {
      const vec3 &vin = small_poly.vertices[vidx];
      vec3 vout = transform_point(frame, vin);
      bounds.Bound(vout.x, vout.y);
    }
  }
  bounds.AddMarginFrac(0.05);

  ImageRGBA img(3840, 2160);
  img.Clear32(0x000000FF);

  Bounds::Scaler scaler = bounds.ScaleToFit(img.Width(), img.Height()).FlipY();

  const double rspan = rbounds.MaxX() - rbounds.MinX();
  const double gspan = gbounds.MaxX() - gbounds.MinX();
  const double bspan = bbounds.MaxX() - bbounds.MinX();
  auto Color = [&](const vec3 &s) {
      float r = (s.x - rbounds.MinX()) / rspan;
      float g = (s.y - gbounds.MinX()) / gspan;
      float b = (s.z - bbounds.MinX()) / bspan;
      return ColorUtil::FloatsTo32(r * 0.9 + 0.1,
                                   g * 0.9 + 0.1,
                                   b * 0.9 + 0.1,
                                   1.0);
    };

  std::vector<vec2> starts;

  for (const vec3 &s : samples) {
    frame3 frame = FrameFromViewPos(s);
    uint32_t color = Color(s);

    auto GetOut = [&](int hidx) {
        const vec3 &vin = small_poly.vertices[hull[hidx]];
        return transform_point(frame, vin);
      };

    for (int hullidx = 0; hullidx < hull.size(); hullidx++) {
      vec3 vout1 = GetOut(hullidx);
      vec3 vout2 = GetOut((hullidx + 1) % hull.size());

      const auto &[x1, y1] = scaler.Scale(vout1.x, vout1.y);
      const auto &[x2, y2] = scaler.Scale(vout2.x, vout2.y);

      img.BlendLine32(x1, y1, x2, y2, color & 0xFFFFFF80);
    }

    vec3 start = GetOut(0);
    const auto &[x, y] = scaler.Scale(start.x, start.y);
    starts.emplace_back(x, y);
  }

  CHECK(samples.size() == starts.size());
#if 0
  for (int i = 0; i < samples.size(); i++) {
    uint32_t color = Color(samples[i]);
    const vec2 &start = starts[i];
    img.BlendCircle32(start.x, start.y, 4, color);
  }
#endif

  img.BlendText2x32(8, 16, 0xFFFF77AA,
                    std::format("{:032b}", code));
  // Draw scale.
  double oneu = scaler.ScaleX(1.0);
  {
    int Y = 32;
    int x0 = 8;
    int x1 = 8 + oneu;
    img.BlendLine32(x0, Y, x1, Y, 0xFFFF7766);
    img.BlendLine32(x0, Y - 4, x0, Y - 1, 0xFFFF7766);
    img.BlendLine32(x1, Y - 4, x1, Y - 1, 0xFFFF7766);
  }

  std::string filename = std::format("patch-{:b}.png", code);
  img.Save(filename);
  printf("Wrote %s in %s\n", filename.c_str(),
         ANSI::Time(timer.Seconds()).c_str());
}

static std::string Usage() {
  return "Usage:\n\n"
    "plotpatch.exe code-in-binary\n";
}

static void PlotCode(uint64_t code) {

  BigPoly scube = BigScube(DIGITS);
  Boundaries boundaries(scube);

  uint64_t mask = GetCodeMask(boundaries, code);
  BigVec3 example = GetBigVec3InPatch(boundaries, code, mask);

  printf("Code: %s\n"
         "Mask: %s\n"
         "Full: %s\n",
         std::format("{:b}", code).c_str(),
         std::format("{:b}", mask).c_str(),
         boundaries.ColorMaskedBits(code, mask).c_str());

  PlotPatch(boundaries, example, code, mask);
}

int main(int argc, char **argv) {
  ANSI::Init();

  CHECK(argc == 2) << Usage();

  std::optional<uint64_t> code = Util::ParseBinary(argv[1]);
  CHECK(code.has_value()) << Usage();

  PlotCode(code.value());

  return 0;
}
