
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numbers>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <unordered_set>

#include "base/logging.h"
#include "base/stringprintf.h"
#include "bounds.h"
#include "hashing.h"
#include "image.h"
#include "mesh.h"
#include "polyhedra.h"
#include "textsvg.h"
#include "util.h"
#include "yocto_matht.h"

using vec3 = yocto::vec<double, 3>;
using frame3 = yocto::frame<double, 3>;
using mat3 = yocto::mat<double, 3>;

static constexpr bool VERBOSE = false;

void TransformMesh(const frame3 &frame, TriangularMesh3D *mesh) {
  for (vec3 &v : mesh->vertices) v = yocto::transform_point(frame, v);
}

void TransformMesh(const mat4 &mtx, TriangularMesh3D *mesh) {
  for (vec3 &v : mesh->vertices) v = yocto::transform_point(mtx, v);
}

static void SaveShadowSVG(std::string_view infile, std::string_view outfile) {

  TriangularMesh3D mesh = LoadSTL(infile);

  // Should always be safe to reorient the mesh.
  OrientMesh(&mesh);

  // Now make sure it has positive volume.
  {
    double vol = MeshVolume(mesh);
    if (VERBOSE) {
      printf("Volume %s: %.11g\n", std::string(infile).c_str(), vol);
    }
    if (MeshVolume(mesh) < 0.0) {
      FlipNormals(&mesh);
    }
  }

  CHECK(MeshVolume(mesh) > 0.0);
  MeshEdgeInfo edge_info(mesh);

  constexpr double WIDTH = 512.0;
  constexpr double HEIGHT = 512.0;

  std::string svg = TextSVG::HeaderEx(0.0, 0.0,
                                      WIDTH, HEIGHT,
                                      "px",
                                      "ruperts");

  std::string view_file = Util::Replace(infile, ".stl", ".view");
  MeshView view;
  if (Util::ExistsFile(view_file)) {
    if (VERBOSE) {
      printf("Loading view from %s.\n", view_file.c_str());
    }
    view = MeshView::FromString(Util::ReadFile(view_file));
  }

  // flip Y axis to get screen coordinates.
  for (vec3 &v : mesh.vertices) {
    v.y = -v.y;
  }

  frame3 f = inverse(yocto::lookat_frame(
      // eye, elevated off xy plane and
      // backed away from the model along y.
      view.camera_pos,
      // always looking at origin
      vec3{0, 0, 0},
      // up is up
      view.up_vector,
      /* inv_xz */ false));

  // mat4 persp = yocto::perspective_mat(view.fov, 1.0, view.near_plane, view.far_plane);

  TransformMesh(f, &mesh);

  Bounds bounds;
  for (const vec3 &v : mesh.vertices) {
    bounds.Bound(v.x, v.y);
  }

  bounds.AddMarginFrac(0.05);

  if (VERBOSE) {
    printf("Bounds: min (%.11g, %.11g) max (%.11g, %.11g)\n",
           bounds.MinX(), bounds.MinY(),
           bounds.MaxX(), bounds.MaxY());
  }

  // FlipY is buggy when some are negative?
  Bounds::Scaler scaler = bounds.ScaleToFit(WIDTH, HEIGHT); // .FlipY();

  // mat3 rotation = yocto::rotation(f);
  // vec3 camera_dir = yocto::normalize(vec3{0, 0, 0} - view.camera_pos);

  std::vector<vec2> v2d;
  v2d.reserve(mesh.vertices.size());
  for (const vec3 &v : mesh.vertices) {
    v2d.emplace_back(v.x, v.y);
  }

  std::vector<int> hull = QuickHull(v2d);

  std::unordered_set<int> on_hull(hull.begin(), hull.end());

  static constexpr int PX = 2;
  static constexpr int IMAGE_WIDTH = 1024, IMAGE_HEIGHT = 1024;
  ImageRGBA img(IMAGE_WIDTH * PX, IMAGE_HEIGHT * PX);
  img.Clear32(0xFFFFFFFF);
  Bounds::Scaler iscaler = bounds.ScaleToFit(img.Width(), img.Height());
  for (const auto &[a, b, c] : mesh.triangles) {
    vec3 v0 = mesh.vertices[a];
    vec3 v1 = mesh.vertices[b];
    vec3 v2 = mesh.vertices[c];

    bool planar01 = edge_info.EdgeAngle(a, b) < 1e-5;
    bool planar12 = edge_info.EdgeAngle(b, c) < 1e-5;
    bool planar20 = edge_info.EdgeAngle(c, a) < 1e-5;

    bool on_hull01 = on_hull.contains(a) && on_hull.contains(b);
    bool on_hull12 = on_hull.contains(b) && on_hull.contains(c);
    bool on_hull20 = on_hull.contains(c) && on_hull.contains(a);

    vec3 normal = yocto::normalize(yocto::cross(v1 - v0, v2 - v0));
    // vec3 normal_camera = yocto::transform_direction(rotation, normal);

    // Backface culling.
    bool backface = normal.z < 0;
    // bool backface = normal_camera.z < 0;
    // bool backface = dot(camera_dir, normal) < 0.0;

    const auto &[v0x, v0y] = scaler.Scale(v0.x, v0.y);
    const auto &[v1x, v1y] = scaler.Scale(v1.x, v1.y);
    const auto &[v2x, v2y] = scaler.Scale(v2.x, v2.y);

    AppendFormat(
        &svg,
        "<polygon points=\"{:.6},{:.6} {:.6},{:.6} {:.6},{:.6}\" "
        "{} stroke=\"#DDD\" />\n",
        v0x, v0y, v1x, v1y, v2x, v2y,
        backface ? "fill=\"#ffaaaa\" fill-opacity=\"0.9\"" :
                   "fill=\"#aaaaff\" fill-opacity=\"0.3\""
                 );

    {
      // And on the image.
      const auto &[v0x, v0y] = iscaler.Scale(v0.x, v0.y);
      const auto &[v1x, v1y] = iscaler.Scale(v1.x, v1.y);
      const auto &[v2x, v2y] = iscaler.Scale(v2.x, v2.y);

      if (!planar01 && !on_hull01)
        img.BlendThickLine32(v0x, v0y, v1x, v1y, PX * 3,
                             0xDDDDDDFF);

      if (!planar12 && !on_hull12)
        img.BlendThickLine32(v1x, v1y, v2x, v2y, PX * 3,
                             0xDDDDDDFF);

      if (!planar20 && !on_hull20)
        img.BlendThickLine32(v2x, v2y, v0x, v0y, PX * 3,
                             0xDDDDDDFF);
    }
  }

  // Draw hull.
  for (int i = 0; i < hull.size(); i++) {
    int prev = (i == 0) ? hull.size() - 1 : i - 1;
    const vec2 &v0 = v2d[hull[prev]];
    const vec2 &v1 = v2d[hull[i]];


    {
      // Image
      const auto &[v0x, v0y] = iscaler.Scale(v0.x, v0.y);
      const auto &[v1x, v1y] = iscaler.Scale(v1.x, v1.y);

      img.BlendThickLine32(v0x, v0y, v1x, v1y, PX * 3,
                           0x000000FF);

    }


  }

  // Vertices last.
  for (const vec3 &v : mesh.vertices) {

    // SVG
    {
      const auto &[vx, vy] = scaler.Scale(v.x, v.y);
      AppendFormat(
          &svg,
          "<circle cx=\"{:.6}\" cy=\"{:.6}\" r=\"{:.6}\" "
          "fill = \"#000000\" />\n",
          vx, vy, WIDTH / 64);
    }

    {
      const auto &[vx, vy] = iscaler.Scale(v.x, v.y);
      img.BlendFilledCircle32(vx, vy, std::max(img.Width() / 64.0, 1.0),
                              0x000000FF);
    }

  }

  std::string png_file = Util::Replace(outfile, ".svg", ".png");
  img.ScaleDownBy(PX).Save(png_file);
  if (VERBOSE) {
    printf("Wrote %s\n", png_file.c_str());
  }

  svg += TextSVG::Footer();

  Util::WriteFile(outfile, svg);
  if (VERBOSE) {
    printf("Wrote %s\n", std::string(outfile).c_str());
  }
}

int main(int argc, char **argv) {

  CHECK(argc == 3) << "./makesvg input.stl out.svg\n";

  std::string infile = argv[1];
  std::string outfile = argv[2];

  SaveShadowSVG(infile, outfile);

  return 0;
}
