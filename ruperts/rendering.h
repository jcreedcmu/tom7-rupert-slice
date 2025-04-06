
#ifndef _RUPERTS_RENDERING_H
#define _RUPERTS_RENDERING_H

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "polyhedra.h"
#include "yocto_matht.h"

#include "image.h"

struct Rendering {
  using vec2 = yocto::vec<double, 2>;

  Rendering(const Polyhedron &p, int width, int height);

  void RenderPerspectiveWireframe(const Polyhedron &p, uint32_t color);
  void RenderMesh(const Mesh2D &mesh);
  void RenderBadPoints(const Mesh2D &sinner, const Mesh2D &souter);
  void RenderHull(const Mesh2D &mesh, const std::vector<int> &hull,
                  uint32_t color = 0x00FF00AA);
  void RenderHullDistance(const Mesh2D &mesh, const std::vector<int> &hull);
  void RenderTriangulation(const Mesh2D &mesh, uint32_t color = 0x4444FFAA);
  void MarkPoints(const Mesh2D &mesh, const std::vector<int> &points,
                  float r = 20.0f,
                  uint32_t color = 0xFFFF00AA);

  void RenderSolution(const Polyhedron &p,
                      const frame3 &outer_frame,
                      const frame3 &inner_frame);

  void RenderTriangle(const Mesh2D &mesh, int ai, int bi, int ci,
                      uint32_t color);

  void DarkenBG();

  void Save(const std::string &filename, bool verbose = true);

  static uint32_t Color(int idx);

  vec2 ToWorld(int sx, int sy) {
    // Center of screen should be 0,0.
    double cy = sy - height / 2.0;
    double cx = sx - width / 2.0;
    return vec2{.x = cx / polyscale, .y = cy / polyscale};
  }

  std::pair<int, int> ToScreen(const vec2 &pt) {
    double cx = pt.x * polyscale;
    double cy = pt.y * polyscale;
    return std::make_pair(cx + width / 2.0, cy + height / 2.0);
  }

  int width = 0, height = 0;
  // Scale of the polyhedron to fit it comfortably in the image
  // when projected from any angle. (This is based on its diameter.)
  // Multiply world dimensions by this to get screen dimensions.
  double polyscale = 1.0;
  ImageRGBA img;
};


#endif
