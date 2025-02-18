
#include <algorithm>
#include <cstdio>
#include <string>
#include <string_view>

#include "base/logging.h"
#include "base/stringprintf.h"
#include "bounds.h"
#include "mesh.h"
#include "polyhedra.h"
#include "textsvg.h"
#include "util.h"
#include "yocto_matht.h"

using vec3 = yocto::vec<double, 3>;
using frame3 = yocto::frame<double, 3>;
using mat3 = yocto::mat<double, 3>;

void TransformMesh(const frame3 &frame, TriangularMesh3D *mesh) {
  for (vec3 &v : mesh->vertices) v = yocto::transform_point(frame, v);
}

static void SaveSVG(TriangularMesh3D mesh, std::string_view filename) {

  constexpr double WIDTH = 512.0;
  constexpr double HEIGHT = 512.0;

  std::string svg = TextSVG::HeaderEx(0.0, 0.0,
                                      WIDTH, HEIGHT,
                                      "px",
                                      "ruperts");

  vec3 camera_pos = vec3{-1, 4, 2};

  frame3 f = yocto::lookat_frame(
      // eye, elevated off xy plane and
      // backed away from the model along y.
      camera_pos,
      // looking at origin
      vec3{0, 0, 0},
      // up is up
      vec3{0, 0, 1},
      /* inv_xz */ false);

  printf("Frame:\n%s\n", FrameString(f).c_str());

  TransformMesh(f, &mesh);

  Bounds bounds;
  for (const vec3 &v : mesh.vertices) {
    bounds.Bound(v.x, v.y);
  }

  // bounds.AddMarginFrac(0.05);

  printf("Bounds: min (%.11g, %.11g) max (%.11g, %.11g)\n",
         bounds.MinX(), bounds.MinY(),
         bounds.MaxX(), bounds.MaxY());

  // FlipY is buggy when some are negative?
  Bounds::Scaler scaler = bounds.ScaleToFit(WIDTH, HEIGHT); // .FlipY();

  // Sort triangles, putting larger z coordinates first so that they
  // are drawn behind.
  std::sort(mesh.triangles.begin(),
            mesh.triangles.end(),
            [&mesh](const auto &t1, const auto &t2) -> bool {
              const auto &[a1, b1, c1] = t1;
              const auto &[a2, b2, c2] = t2;
              double z1 = std::max(mesh.vertices[a1].z,
                                   std::max(mesh.vertices[b1].z,
                                            mesh.vertices[c1].z));
              double z2 = std::max(mesh.vertices[a2].z,
                                   std::max(mesh.vertices[b2].z,
                                            mesh.vertices[c2].z));
              return z1 > z2;
            });

  mat3 rotation = yocto::rotation(f);
  vec3 camera_dir = yocto::normalize(vec3{0, 0, 0} -
                                             camera_pos);

  for (const auto &[a, b, c] : mesh.triangles) {
    vec3 v0 = mesh.vertices[a];
    vec3 v1 = mesh.vertices[b];
    vec3 v2 = mesh.vertices[c];

    vec3 normal = yocto::normalize(yocto::cross(v1 - v0, v2 - v0));
    vec3 normal_camera = yocto::transform_direction(rotation, normal);

    // Backface culling.
    // bool backface = normal_camera.z < 0;
    bool backface = dot(camera_dir, normal) < 0.0;

    const auto &[v0x, v0y] = scaler.Scale(v0.x, v0.y);
    const auto &[v1x, v1y] = scaler.Scale(v1.x, v1.y);
    const auto &[v2x, v2y] = scaler.Scale(v2.x, v2.y);



    AppendFormat(
        &svg,
        "<polygon points=\"{:.6},{:.6} {:.6},{:.6} {:.6},{:.6}\" "
        "{} stroke=\"#000\" />\n",
        v0x, v0y, v1x, v1y, v2x, v2y,
        backface ? "fill=\"#ffaaaa\" fill-opacity=\"0.9\"" :
                   "fill=\"#aaaaff\" fill-opacity=\"0.9\""
                 );
  }

  svg += TextSVG::Footer();

  Util::WriteFile(filename, svg);
  printf("Wrote %s\n", std::string(filename).c_str());
}

int main(int argc, char **argv) {

  CHECK(argc == 3) << "./makesvg input.stl out.svg\n";

  std::string infile = argv[1];
  std::string outfile = argv[2];

  TriangularMesh3D mesh = LoadSTL(infile);

  SaveSVG(mesh, outfile);

  return 0;
}
