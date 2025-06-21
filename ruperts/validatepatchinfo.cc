
#include <cstdint>
#include <cstdio>
#include <format>
#include <string>
#include <vector>

#include "ansi.h"
#include "arcfour.h"
#include "big-polyhedra.h"
#include "map-util.h"
#include "patches.h"
#include "polyhedra.h"
#include "rendering.h"
#include "status-bar.h"
#include "yocto_matht.h"

constexpr int DIGITS = 24;

static std::vector<vec2> PlaceHull(const Polyhedron &small_poly,
                                   const frame3 &frame,
                                   const std::vector<int> &hull) {
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

static void ValidatePatchInfo() {
  BigPoly poly = BigScube(DIGITS);
  Polyhedron small_poly = SmallPoly(poly);
  Boundaries boundaries(poly);
  PatchInfo patchinfo = LoadPatchInfo("scube-patchinfo.txt");
  const auto canonical = MapToSortedVec(patchinfo.canonical);
  StatusBar status(1);
  ArcFour rc("validate");
  int64_t codes_ok = 0, ok = 0, bad_area = 0, bad_conv = 0,
    bad_conv_cw = 0;
  for (int c = 0; c < canonical.size(); c++) {
    bool code_ok = true;
    const auto &[code, can] = canonical[c];
    std::vector<int> hull = ComputeHullForPatch(
        boundaries, code, can.mask, {"validate"});
    for (int i = 0; i < 1000; i++) {
      vec3 view = GetVec3InPatch(&rc,
                                 boundaries,
                                 code,
                                 can.mask);

      const frame3 frame = FrameFromViewPos(view);

      const std::vector<vec2> outer_poly =
        PlaceHull(small_poly, frame, hull);

      bool bad = false;

      if (SignedAreaOfConvexPoly(outer_poly) <= 0.0) {
        bad = true;
        bad_area++;
      }

      if (!IsPolyConvex(outer_poly)) {
        if (code_ok) {
          std::string cs = std::format("{:b}", code);
          status.Print(ARED("{}") " not convex\n", cs);
          Polyhedron rpoly = Rotate(small_poly, frame);
          Rendering rendering(small_poly, 1920, 1080);
          rendering.RenderMesh(Shadow(rpoly));
          rendering.RenderPolygon(outer_poly, 0xFF00FFCC);
          rendering.Save(std::format("non-convex-polygon-{}.png", cs));
          LOG(FATAL) << "EXIT EARLY";
        }
        bad = true;
        bad_conv++;
      }

      if (!IsConvexAndScreenClockwise(outer_poly)) {
        bad = true;
        bad_conv_cw++;
      }

      if (bad) {
        code_ok = false;
      } else {
        ok++;
      }
    }
    if (code_ok) codes_ok++;

    status.Progressf(c, canonical.size(),
                     "%s",
                     std::format("{:b} {} ok", code, codes_ok).c_str());
  }

  printf("Codes ok: %lld. Total ok: %lld\n"
         "Bad (Non-positive area): %lld\n"
         "Bad (Not convex): %lld\n"
         "Bad (Not convex or not clockwise): %lld\n",
         codes_ok, ok, bad_area, bad_conv, bad_conv_cw);
}

int main(int argc, char **argv) {
  ANSI::Init();

  ValidatePatchInfo();

  return 0;
}
