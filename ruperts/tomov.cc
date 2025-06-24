
// Generate a movie of the named shape rotating in space.

#include <cstdio>
#include <format>
#include <numbers>
#include <string>
#include <string_view>
#include <utility>

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

static void AnimateMesh(const Polyhedron &poly, std::string_view filename) {
  ArcFour rc("animate");
  quat4 initial_rot = RandomQuaternion(&rc);

  constexpr int SIZE = 2160;
  constexpr int FRAMES = 10 * 60;
  MovRecorder rec(filename, SIZE, SIZE,
                  MOV::DURATION_60, MOV::Codec::PNG_CCLIB);

  Periodically status_per(1.0);
  for (int i = 0; i < FRAMES; i++) {
    if (status_per.ShouldRun()) {
      status.Progress(i, FRAMES, "rotate " ACYAN("{}"), poly.name);
    }

    double t = i / (double)FRAMES;
    double angle = t * 2.0 * std::numbers::pi;

    // rotation quat actually returns vec4; isomorphic to quat4.
    quat4 frame_rot =
      QuatFromVec(yocto::rotation_quat<double>({0.0, 1.0, 0.0}, angle));

    quat4 final_rot = normalize(initial_rot * frame_rot);
    Polyhedron rpoly = Rotate(poly, yocto::rotation_frame(final_rot));

    Rendering rendering(poly, SIZE, SIZE);
    Mesh2D mesh = Shadow(rpoly);
    rendering.RenderMesh(mesh, 6.0f);
    rec.AddFrame(std::move(rendering.img));
  }
}


int main(int argc, char **argv) {
  ANSI::Init();
  printf("\n");

  CHECK(argc == 3) << "./tomov.exe polyhedron output.mov";

  Polyhedron target = PolyhedronByName(argv[1]);
  std::string filename = argv[2];

  AnimateMesh(target, filename);

  return 0;
}
