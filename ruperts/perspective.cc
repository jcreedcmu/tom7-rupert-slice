
#include <cstdio>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include "ansi.h"
#include "base/logging.h"
#include "base/stringprintf.h"
#include "mesh.h"
#include "polyhedra.h"
#include "image.h"
#include "util.h"
#include "yocto_matht.h"

using vec3 = yocto::vec<double, 3>;
using frame3 = yocto::frame<double, 3>;
using mat3 = yocto::mat<double, 3>;

void TransformMesh(const frame3 &frame, TriangularMesh3D *mesh) {
  for (vec3 &v : mesh->vertices) v = yocto::transform_point(frame, v);
}

void TransformMesh(const mat4 &mtx, TriangularMesh3D *mesh) {
  for (vec3 &v : mesh->vertices) v = yocto::transform_point(mtx, v);
}

inline vec2 ProjectPt(const mat4 &proj, const vec3 &point) {
  vec4 pp = proj * vec4{.x = point.x, .y = point.y, .z = point.z, .w = 1.0};
  return vec2{.x = pp.x / pp.w, .y = pp.y / pp.w};
}


static void PerspectiveTest(TriangularMesh3D mesh) {

  // Should always be safe to reorient the mesh.
  OrientMesh(&mesh);

  vec3 camera_pos = vec3{-1.0, -.03, 4.0};

  #if 0
  frame3 f = yocto::lookat_frame(
      camera_pos,
      // looking at origin
      vec3{0, 0, 0},
      // up is up
      vec3{0, 0, 1},
      /* inv_xz */ false);
  #endif

  frame3 f = translation_frame(-camera_pos);

  constexpr double FOVY = 1.0; // 1 radian is about 60 deg
  constexpr double ASPECT_RATIO = 1.0;
  constexpr double NEAR_PLANE = 0.1;
  constexpr double FAR_PLANE = 1000.0;
  mat4 persp = yocto::perspective_mat(FOVY, ASPECT_RATIO, NEAR_PLANE,
                                      FAR_PLANE);

  printf("Frame:\n%s\n", FrameString(f).c_str());

  // for (vec3 &v : mesh.vertices) v = v + vec3{1.0, 0.3, -4.0};
  TransformMesh(f, &mesh);

  // TransformMesh(persp, &mesh);

  std::vector<vec2> vert2;
  vert2.reserve(mesh.vertices.size());
  // indexed by triangle!
  std::vector<std::tuple<vec2, vec2, bool>> normals;
  for (const vec3 &point : mesh.vertices) {
    vert2.push_back(ProjectPt(persp, point));
  }
  for (const auto &[a, b, c] : mesh.triangles) {
    vec3 v0 = mesh.vertices[a];
    vec3 v1 = mesh.vertices[b];
    vec3 v2 = mesh.vertices[c];

    vec3 ctr = (v0 + v1 + v2) / 3.0;
    vec3 normal = normalize(cross(v1 - v0, v2 - v0)) * 0.25;
    bool backface = normal.z < 0;
    normals.emplace_back(ProjectPt(persp, ctr),
                         ProjectPt(persp, ctr + normal),
                         backface);
  }


  for (const vec2 &v : vert2) {
    printf("  vertex: %.6g %.6g\n",
           v.x, v.y);
  }

  vec2 xpos = ProjectPt(persp, transform_point(f, vec3{1, 0, 0}));
  vec2 ypos = ProjectPt(persp, transform_point(f, vec3{0, 1, 0}));
  vec2 zpos = ProjectPt(persp, transform_point(f, vec3{0, 0, 1}));
  vec2 origin = ProjectPt(persp, transform_point(f, vec3{0, 0, 0}));

  printf("axes\n"
         "x: %s\n"
         "y: %s\n"
         "z: %s\n",
         VecString(xpos).c_str(),
         VecString(ypos).c_str(),
         VecString(zpos).c_str());



  ImageRGBA img(1024, 1024);
  img.Clear32(0x000000FF);

  vec2 center = vec2{img.Width() * 0.5, (double)img.Height() * 0.5};
  double scale = 480.0;

  for (int t = 0; t < mesh.triangles.size(); t++) {
    const auto &[a, b, c] = mesh.triangles[t];
    const auto &[ns, ne, backface] = normals[t];
    if (!backface) {
      vec2 v0 = vert2[a] * scale + center;
      vec2 v1 = vert2[b] * scale + center;
      vec2 v2 = vert2[c] * scale + center;
      img.BlendLine32(v0.x, v0.y, v1.x, v1.y, 0xFFFFFF77);
      img.BlendLine32(v1.x, v1.y, v2.x, v2.y, 0xFFFFFF77);
      img.BlendLine32(v2.x, v2.y, v0.x, v0.y, 0xFFFFFF77);

      vec2 n0 = ns * scale + center;
      vec2 n1 = ne * scale + center;
      img.BlendLine32(n0.x, n0.y, n1.x, n1.y,
                      backface ? 0xFF770055 : 0x33FF5555);
    }
  }

  origin = origin * scale + center;
  xpos = xpos * scale + center;
  ypos = ypos * scale + center;
  zpos = zpos * scale + center;

  img.BlendCircle32(origin.x, origin.y, 10, 0xFF33FFCC);

  img.BlendLine32(origin.x, origin.y,
                  xpos.x, xpos.y,
                  0xFF0000AA);
  img.BlendLine32(origin.x, origin.y,
                  ypos.x, ypos.y,
                  0x00AA00AA);
  img.BlendLine32(origin.x, origin.y,
                  zpos.x, zpos.y,
                  0x0000FFAA);

  img.Save("perspective.png");
  printf("Wrote " AGREEN("perspective.png") "\n");
}

int main(int argc, char **argv) {
  TriangularMesh3D mesh = LoadSTL("platonic-dodecahedron.stl");

  PerspectiveTest(mesh);

  return 0;
}
