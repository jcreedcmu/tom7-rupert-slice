
#include "rendering.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <initializer_list>
#include <string>
#include <vector>

#include "ansi.h"
#include "base/logging.h"
#include "color-util.h"
#include "image.h"

#include "yocto_matht.h"
#include "polyhedra.h"

using vec2 = yocto::vec<double, 2>;
using vec3 = yocto::vec<double, 3>;
using vec4 = yocto::vec<double, 4>;
using mat4 = yocto::mat<double, 4>;
using quat4 = yocto::quat<double, 4>;
using frame3 = yocto::frame<double, 3>;

constexpr double FIT_SCALE = 0.80;

static std::vector<uint32_t> COLORS = {
  0xFF0000FF,
  0xFFFF00FF,
  0x00FF00FF,
  0x00FFFFFF,
  0x0000FFFF,
  0xFF00FFFF,
  0x770000FF,
  0x777700FF,
  0x007700FF,
  0x007777FF,
  0x000088FF, // distinguish more from 55 below
  0x770077FF,
  0xFF7777FF,
  0xFFFF77FF,
  0x77FF77FF,
  0x77FFFFFF,
  0x7777FFFF,
  0xFF77FFFF,
  0x330000FF,
  0x333300FF,
  0x003300FF,
  0x003333FF,
  0x000055FF, // 33 is a little too dark
  0x330033FF,
  0x773333FF,
  0x777733FF,
  0x337733FF,
  0x337777FF,
  0x333377FF,
  0x773377FF,
  0xFFAAAAFF,
  0xFFFFAAFF,
  0xAAFFAAFF,
  0xAAFFFFFF,
  0xAAAAFFFF,
  0xFFAAFFFF,
  0xFF3333FF,
  0xFFFF33FF,
  0x33FF33FF,
  0x33FFFFFF,
  0x3333FFFF,
  0xFF33FFFF,
  0x773333FF,
  0x77FF33FF,
  0x3377FFFF,
  0x7733FFFF,
  0xFF7733FF,
  0x33FF77FF,
  0x3333FFFF,
  0xFF3377FF,
  0x77AAAAFF,
  0x77FFAAFF,
  0xAA77FFFF,
  0x77AAFFFF,
  0xFF77AAFF,
  0xAAFF77FF,
  0xAAAAFFFF,
  0xFFAA77FF,
  0x33AAAAFF,
  0x33FFAAFF,
  0xAA33FFFF,
  0x33AAFFFF,
  0xFF33AAFF,
  0xAAFF33FF,
  0xAAAAFFFF,
  0xFFAA33FF,
};

Rendering::Rendering(const Polyhedron &p, int width_in, int height_in) :
  width(width_in), height(height_in), img(width, height) {
  img.Clear32(0x000000FF);

  double dia = Diameter(p);
  polyscale = (std::min(width, height) / dia) * FIT_SCALE;
}

void Rendering::RenderPerspectiveWireframe(const Polyhedron &p,
                                           uint32_t color) {
  constexpr double aspect = 1.0;
  const mat4 proj = yocto::perspective_mat(
      yocto::radians(60.0), aspect, 0.1, 100.0);

  const frame3 camera_frame = yocto::lookat_frame<double>(
      {0, 0, 5}, {0, 0, 0}, {0, 1, 0});
  const mat4 view_matrix = yocto::frame_to_mat(camera_frame);
  const mat4 model_view_projection = proj * view_matrix;

  for (const std::vector<int> &face : p.faces->v) {
    for (int i = 0; i < (int)face.size(); i++) {
      const vec3 v0 = p.vertices[face[i]];
      const vec3 v1 = p.vertices[face[(i + 1) % face.size()]];

      const vec2 p0 = Project(v0, model_view_projection);
      const vec2 p1 = Project(v1, model_view_projection);

      // Note that polyscale will fit an orthographic projection,
      // but it might not fit a perspective one.
      const float x0 = (p0.x * polyscale + width * 0.5);
      const float y0 = (p0.y * polyscale + height * 0.5);
      const float x1 = (p1.x * polyscale + width * 0.5);
      const float y1 = (p1.y * polyscale + height * 0.5);
      img.BlendThickLine32(x0, y0, x1, y1, 4.0f, color & 0xFFFFFF88);
    }
  }
}

void Rendering::RenderMesh(const Mesh2D &mesh) {
  // Draw filled polygons first.
  for (int sy = 0; sy < height; sy++) {
    for (int sx = 0; sx < width; sx++) {
      const vec2 pt = ToWorld(sx, sy);
      for (int i = 0; i < mesh.faces->v.size(); i++) {
        if (PointInPolygon(pt, mesh.vertices, mesh.faces->v[i])) {
          img.BlendPixel32(sx, sy, Color(i) & 0xFFFFFF22);
        }
      }
    }
  }

  // Draw lines on top.
  for (const std::vector<int> &face : mesh.faces->v) {
    for (int i = 0; i < face.size(); i++) {
      const vec2 &a = mesh.vertices[face[i]];
      const vec2 &b = mesh.vertices[face[(i + 1) % face.size()]];

      const auto &[ax, ay] = ToScreen(a);
      const auto &[bx, by] = ToScreen(b);
      img.BlendThickLine32(ax, ay, bx, by, 3.0, 0xFFFFFF99);
    }
  }
}

void Rendering::RenderBadPoints(const Mesh2D &sinner, const Mesh2D &souter) {
  for (const vec2 &v : sinner.vertices) {
    if (!InMesh(souter, v)) {
      const auto &[sx, sy] = ToScreen(v);
      img.BlendThickCircle32(sx, sy, 20, 4, 0xFF0000AA);
    }
  }
}

void Rendering::RenderTriangulation(const Mesh2D &mesh, uint32_t color) {
  for (const auto &[a, b, c] : mesh.faces->triangulation) {
    const auto &[x0, y0] = ToScreen(mesh.vertices[a]);
    const auto &[x1, y1] = ToScreen(mesh.vertices[b]);
    const auto &[x2, y2] = ToScreen(mesh.vertices[c]);
    img.BlendThickLine32(x0, y0, x1, y1, 3.0, color);
    img.BlendThickLine32(x1, y1, x2, y2, 3.0, color);
    img.BlendThickLine32(x2, y2, x0, y0, 3.0, color);
  }
}

void Rendering::RenderHull(const Mesh2D &mesh,
                           const std::vector<int> &hull,
                           uint32_t color) {
  for (int i = 0; i < hull.size(); i++) {
    const vec2 &v0 = mesh.vertices[hull[i]];
    const vec2 &v1 = mesh.vertices[hull[(i + 1) % hull.size()]];

    const auto &[x0, y0] = ToScreen(v0);
    const auto &[x1, y1] = ToScreen(v1);

    img.BlendThickLine32(x0, y0, x1, y1, 2.0, color);
  }

  /*
  for (int i = 0; i < hull.size(); i++) {
    const vec2 &v0 = mesh.vertices[hull[i]];
    const auto &[x, y] = ToScreen(v0);
    img.BlendText32(x - 12, y - 12, 0xFFFF00FF,
                     StringPrintf("%d", i));
  }
  */
}

// Red for negative, black for 0, green for positive.
// nominal range [-1, 1].
static constexpr ColorUtil::Gradient DISTANCE{
  GradRGB(-4.0f, 0xFFFF88),
  GradRGB(-2.0f, 0xFFFF00),
  GradRGB(-1.0f, 0xFF0000),
  GradRGB( 0.0f, 0x440044),
  GradRGB( 1.0f, 0x00FF00),
  GradRGB(+2.0f, 0x00FFFF),
  GradRGB(+4.0f, 0x88FFFF),
};

void Rendering::RenderHullDistance(const Mesh2D &mesh,
                                   const std::vector<int> &hull) {
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      const vec2 v = ToWorld(x, y);
      const double d = std::abs(DistanceToHull(mesh.vertices, hull, v));
      img.BlendPixel32(x, y, ColorUtil::LinearGradient32(DISTANCE, d));
    }
  }
}

void Rendering::DarkenBG() {
  for (int y = 0; y < img.Height(); y++) {
    for (int x = 0; x < img.Width(); x++) {
      img.BlendPixel32(x, y, 0x550000AA);
    }
  }
}

uint32_t Rendering::Color(int idx) {
  return COLORS[idx % COLORS.size()];
}

void Rendering::Save(const std::string &filename) {
  img.Save(filename);
  printf("Wrote " AGREEN("%s") "\n", filename.c_str());
}
