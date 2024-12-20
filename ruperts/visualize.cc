
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <numbers>
#include <string>
#include <utility>
#include <vector>

#include "ansi.h"
#include "arcfour.h"
#include "base/stringprintf.h"
#include "image.h"
#include "mov-recorder.h"
#include "periodically.h"
#include "polyhedra.h"
#include "randutil.h"
#include "rendering.h"
#include "status-bar.h"
#include "timer.h"
#include "yocto_matht.h"

using vec2 = yocto::vec<double, 2>;
using vec3 = yocto::vec<double, 3>;
using vec4 = yocto::vec<double, 4>;
using mat4 = yocto::mat<double, 4>;
using quat4 = yocto::quat<double, 4>;
using frame3 = yocto::frame<double, 3>;

[[maybe_unused]]
static void AnimateMesh(const Polyhedron &poly) {
  ArcFour rc("animate");
  quat4 initial_rot = RandomQuaternion(&rc);

  constexpr int SIZE = 1080;
  constexpr int FRAMES = 10 * 60;
  MovRecorder rec(StringPrintf("animate-%s.mov", poly.name), SIZE, SIZE);

  StatusBar status(2);
  Periodically status_per(1.0);
  for (int i = 0; i < FRAMES; i++) {
    if (status_per.ShouldRun()) {
      status.Progressf(i, FRAMES, "rotate");
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
    rendering.RenderMesh(mesh);
    rec.AddFrame(std::move(rendering.img));
  }
}

[[maybe_unused]]
static void AnimateHull() {
  ArcFour rc("animate");

  constexpr int WIDTH = 1920;
  constexpr int HEIGHT = 1080;
  constexpr int SIZE = HEIGHT;
  constexpr int FRAMES = 10 * 60;
  constexpr int POINTS = 100;
  MovRecorder rec("animate-hull.mov", WIDTH, HEIGHT);

  std::vector<vec2> points;
  std::vector<vec2> vels;

  RandomGaussian gauss(&rc);
  for (int i = 0; i < POINTS; i++) {
    double x =
      std::clamp(gauss.Next() * SIZE * 0.1 + SIZE * 0.5, 0.0, (double)SIZE);
    double y =
      std::clamp(gauss.Next() * SIZE * 0.1 + SIZE * 0.5, 0.0, (double)SIZE);
    points.emplace_back(vec2{x, y});
    vels.emplace_back(
        vec2{
          .x = RandDouble(&rc) * 8.0 - 4.0,
          .y = RandDouble(&rc) * 8.0 - 4.0,
        });
  }

  double sec1 = 0.0, sec2 = 0.0;

  StatusBar status(2);
  Periodically status_per(1.0);
  for (int i = 0; i < FRAMES; i++) {
    if (status_per.ShouldRun()) {
      status.Progressf(i, FRAMES, "hull");
    }

    ImageRGBA img(WIDTH, HEIGHT);
    img.Clear32(0x000000FF);

    for (int i = 0; i < (int)points.size(); i++) {
      img.BlendFilledCircle32(points[i].x, points[i].y, 6.0f,
                              (Rendering::Color(i) & 0xFFFFFFAA) |
                              0x33333300);
      img.BlendCircle32(points[i].x, points[i].y, 6.0f, 0xFFFFFF44);
    }

    Timer timer1;
    std::vector<int> hull1 = GrahamScan(points);
    sec1 += timer1.Seconds();

    Timer timer2;
    std::vector<int> hull2 = QuickHull(points);
    sec2 += timer2.Seconds();

    // printf("Got hull sized %d, %d\n", (int)hull1.size(), (int)hull2.size());

    auto DrawHull = [&](const std::vector<int> &hull, int32_t color) {
        for (int i = 0; i < hull.size(); i++) {
          const vec2 &a = points[hull[i]];
          const vec2 &b = points[hull[(i + 1) % hull.size()]];

          img.BlendThickLine32(a.x, a.y, b.x, b.y, 2.0f, color);
        }
      };

    DrawHull(hull1, 0xFFFFFF33);
    // DrawHull(hull2, 0x00FFFF33);

    // img.Save(StringPrintf("hull%d.png", i));

    rec.AddFrame(std::move(img));

    for (int i = 0; i < (int)points.size(); i++) {
      points[i] += vels[i];
      if (points[i].x < 0.0) {
        points[i].x = 0.0;
        vels[i].x = -vels[i].x;
      }
      if (points[i].y < 0.0) {
        points[i].y = 0.0;
        vels[i].y = -vels[i].y;
      }

      if (points[i].x > SIZE) {
        points[i].x = SIZE;
        vels[i].x = -vels[i].x;
      }

      if (points[i].y > SIZE) {
        points[i].y = SIZE;
        vels[i].y = -vels[i].y;
      }
    }

    // gravity :)
    for (vec2 &d : vels) {
      d += vec2{0.0, 0.05};
    }

  }

  printf("Hull1: %s. Hull2: %s\n",
         ANSI::Time(sec1).c_str(), ANSI::Time(sec2).c_str());

}

[[maybe_unused]]
static void Visualize(const Polyhedron &poly) {
  // ArcFour rc(StringPrintf("seed.%lld", time(nullptr)));
  ArcFour rc("fixed-seed");

  CHECK(PlanarityError(poly) < 1.0e-10);
  printf("Planarity OK.\n");

  {
    Rendering rendering(poly, 1920, 1080);
    for (int i = 0; i < 5; i++) {
      frame3 frame = yocto::rotation_frame(RandomQuaternion(&rc));
      Polyhedron rpoly = Rotate(poly, frame);

      CHECK(PlanarityError(rpoly) < 1.0e10);
      rendering.RenderPerspectiveWireframe(rpoly, Rendering::Color(i));
    }

    rendering.Save(StringPrintf("wireframe-%s.png", poly.name));
  }

  {
    Rendering rendering(poly, 1920, 1080);
    // quat4 q = RandomQuaternion(&rc);
    // frame3 frame = yocto::rotation_frame(q);
    // Polyhedron rpoly = Rotate(poly, frame);

    Mesh2D mesh = Shadow(poly);
    rendering.RenderMesh(mesh);

    printf("Get convex hull (%d vertices):\n",
           (int)mesh.vertices.size());
    std::vector<int> hull = QuickHull(mesh.vertices);
    printf("Hull size %d\n", (int)hull.size());
    // rendering.RenderHull(mesh, hull);

    rendering.Save(StringPrintf("shadow-%s.png", poly.name));
  }
}

int main(int argc, char **argv) {
  ANSI::Init();
  printf("\n");

  Polyhedron target = DeltoidalHexecontahedron();

  // These generate visualizations of the polyhedron;
  // they are unrelated to solving.

  // AnimateHull();
  Visualize(target);
  AnimateMesh(target);

  return 0;
}
