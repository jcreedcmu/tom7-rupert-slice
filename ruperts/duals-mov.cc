
// Generate a movie of the named shape rotating in space.

#include <cmath>
#include <cstdio>
#include <format>
#include <numbers>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "ansi.h"
#include "arcfour.h"
#include "image.h"
#include "mov-recorder.h"
#include "mov.h"
#include "periodically.h"
#include "polyhedra.h"
#include "rendering.h"
#include "status-bar.h"
#include "yocto_matht.h"

using vec2 = yocto::vec<double, 2>;
using vec3 = yocto::vec<double, 3>;
using vec4 = yocto::vec<double, 4>;
using mat4 = yocto::mat<double, 4>;
using quat4 = yocto::quat<double, 4>;
using frame3 = yocto::frame<double, 3>;

static StatusBar status(2);

static void AnimateDuals(std::string_view filename) {
  std::vector<Polyhedron> top, bot;
  for (std::string_view s : {
      // arbitrary choice of platonic
      "tetrahedron",
      "cube",
      "dodecahedron",
      // and archimedean
      "cuboctahedron",
      "icosidodecahedron",
      "rhombicosidodecahedron",
      "rhombicuboctahedron",
      "snubcube",
      "snubdodecahedron",
      "truncatedcube",
      "truncatedcuboctahedron",
      "truncateddodecahedron",
      "truncatedicosahedron",
      "truncatedicosidodecahedron",
      "truncatedoctahedron",
      "truncatedtetrahedron",
    }) {
    top.push_back(PolyhedronByName(s));
    bot.push_back(PolyhedronByName(DualPolyhedron(s)));
  }

  // TODO: Sort by vertices?

  ArcFour rc("animate");

  std::vector<quat4> rots;
  for (int i = 0; i < top.size(); i++) {
    rots.push_back(RandomQuaternion(&rc));
  }

  constexpr int WIDTH = 3840;
  constexpr int HEIGHT = 2160;
  constexpr int FRAMES = 10 * 60;
  MovRecorder rec(filename, WIDTH, HEIGHT,
                  MOV::DURATION_60, MOV::Codec::PNG_CCLIB);
  rec.SetEncodingThreads(12);
  rec.SetMaxQueueSize(60);

  Periodically status_per(1.0);
  for (int i = 0; i < FRAMES; i++) {
    if (status_per.ShouldRun()) {
      status.Progress(i, FRAMES, "rotate " ACYAN("duals"));
    }

    CHECK(top.size() == bot.size());
    const int cols = top.size();
    const double one_width = WIDTH / (double)cols;

    double t = i / (double)FRAMES;
    double angle = t * 2.0 * std::numbers::pi;

    // rotation quat actually returns vec4; isomorphic to quat4.
    quat4 frame_rot =
      QuatFromVec(yocto::rotation_quat<double>({0.0, 1.0, 0.0}, angle));

    ImageRGBA frame(WIDTH, HEIGHT);
    frame.Clear32(0x000000FF);
    for (int c = 0; c < cols; c++) {
      int x = std::round(one_width * c);

      const quat4 &initial_rot = rots[c];
      quat4 final_rot = normalize(initial_rot * frame_rot);

      auto Place = [&frame, one_width, x, &final_rot](
          const Polyhedron &poly_in, int y) {
          Polyhedron poly = Rotate(poly_in, yocto::rotation_frame(final_rot));

          Rendering rendering(poly, one_width, one_width);
          Mesh2D mesh = Shadow(poly);
          rendering.RenderMesh(mesh, 2.25f);
          frame.CopyImage(x, y, rendering.img);
        };

      Place(top[c], 0);
      Place(bot[c], HEIGHT - one_width);
    }

    rec.AddFrame(std::move(frame));
  }

  status.Printf("Finalizing.");
}


int main(int argc, char **argv) {
  ANSI::Init();
  printf("\n");

  CHECK(argc == 2) << "./duals-mov.exe output.mov";

  std::string filename = argv[1];

  AnimateDuals(filename);

  return 0;
}
