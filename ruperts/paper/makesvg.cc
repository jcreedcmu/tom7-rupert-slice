
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numbers>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>

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

struct FaceInfo {
  // Ordered a < b.
  using Edge = std::pair<int, int>;

  static Edge MakeEdge(int a, int b) {
    if (a > b) std::swap(a, b);
    return std::make_pair(a, b);
  }

  // Always non-negative.
  std::unordered_map<Edge, double, Hashing<Edge>> dihedral_angle;
  // The other points of the two triangles that touch this edge.
  std::unordered_map<Edge, std::pair<int, int>, Hashing<Edge>> other_points;

  // The edge a-b or b-a must exist!
  double EdgeAngle(int a, int b) {
    Edge e = MakeEdge(a, b);
    auto it = dihedral_angle.find(e);
    CHECK(it != dihedral_angle.end());
    return it->second;
  }

  FaceInfo(const TriangularMesh3D &mesh) {
    for (const auto &[a, b, c] : mesh.triangles) {
      auto AddEdge = [this](int a, int b, int c) {
          Edge e = MakeEdge(a, b);
          auto &[p1, p2] = other_points[e];

          // It's not possible for the other points to both be
          // zero, so we know it must have been default-constructed.
          if (p1 == 0 && p2 == 0) {
            // No value yet.
            p1 = -1;
            p2 = -1;
          }

          if (p1 == -1) {
            p1 = c;
          } else if (p2 == -1) {
            p2 = c;
          } else {
            LOG(FATAL) << "Non-manifold: More than two "
              "triangles share an edge. " <<
              StringPrintf("Triangle: %d,%d,%d. Already have "
                           "p1 = %d, p2 = %d",
                           a, b, c, p1, p2);
          }
        };

      AddEdge(a, b, c);
      AddEdge(b, c, a);
      AddEdge(c, a, b);
    }

    for (const auto &[e, p] : other_points) {
      const auto &[a, b] = e;
      const auto &[c, d] = p;
      CHECK(c != -1 && d != -1) << "Non-manifold: An edge "
        "that does not connect two triangles.";

      const vec3 &va = mesh.vertices[a];
      const vec3 &vb = mesh.vertices[b];
      const vec3 &vc = mesh.vertices[c];
      const vec3 &vd = mesh.vertices[d];

      vec3 nabc = normalize(cross(vb - va, vc - va));
      vec3 nabd = normalize(cross(vb - va, vd - va));

      // Angle between face normals.
      double angle = std::acos(
          std::clamp(dot(nabc, nabd), -1.0, 1.0));

      angle = std::min(angle, std::numbers::pi - angle);

      CHECK(angle >= 0.0);
      // printf("Angle %d-%d: %.11g\n", a, b, angle);
      dihedral_angle[e] = angle;
    }
  }
};

static void SaveSVG(std::string_view infile, std::string_view outfile) {

  TriangularMesh3D mesh = LoadSTL(infile);

  // Should always be safe to reorient the mesh.
  OrientMesh(&mesh);

  // Now make sure it has positive volume.
  {
    double vol = MeshVolume(mesh);
    printf("Volume %s: %.11g\n", std::string(infile).c_str(), vol);
    if (MeshVolume(mesh) < 0.0) {
      FlipNormals(&mesh);
    }
  }

  CHECK(MeshVolume(mesh) > 0.0);
  FaceInfo face_info(mesh);

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

  frame3 f = inverse(yocto::lookat_frame(
      // eye, elevated off xy plane and
      // backed away from the model along y.
      view.camera_pos,
      // always looking at origin
      vec3{0, 0, 0},
      // up is up
      view.up_vector,
      /* inv_xz */ false));

  mat4 persp = yocto::perspective_mat(view.fov, 1.0, view.near_plane, view.far_plane);

  mat4 mtx = persp;

  TransformMesh(f, &mesh);
  TransformMesh(mtx, &mesh);

  /*
  for (const vec3 &v : mesh.vertices) {
    printf("  vertex: %.6g %.6g %.6g\n",
           v.x, v.y, v.z);
  }
  */

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

  // Sort triangles by ascending z coordinate, after projection.
  // Positive z is closer to the camera.
  std::sort(mesh.triangles.begin(),
            mesh.triangles.end(),
            [&mesh](const auto &t1, const auto &t2) -> bool {
              const auto &[a1, b1, c1] = t1;
              const auto &[a2, b2, c2] = t2;

              /*
              double z1 = std::max({
                  mesh.vertices[a1].z,
                  mesh.vertices[b1].z,
                  mesh.vertices[c1].z});
              double z2 = std::max({
                  mesh.vertices[a2].z,
                  mesh.vertices[b2].z,
                  mesh.vertices[c2].z});
              */

              double avg1 =
                (mesh.vertices[a1].z +
                 mesh.vertices[b1].z +
                 mesh.vertices[c1].z);
              double avg2 =
                (mesh.vertices[a2].z +
                 mesh.vertices[b2].z +
                 mesh.vertices[c2].z);

              // XXX
              return avg1 > avg2;
            });

  // mat3 rotation = yocto::rotation(f);
  // vec3 camera_dir = yocto::normalize(vec3{0, 0, 0} - view.camera_pos);


  static constexpr int PX = 2;
  static constexpr int IMAGE_WIDTH = 1024, IMAGE_HEIGHT = 1024;
  ImageRGBA img(IMAGE_WIDTH * PX, IMAGE_HEIGHT * PX);
  img.Clear32(0xFFFFFFFF);
  Bounds::Scaler iscaler = bounds.ScaleToFit(img.Width(), img.Height());
  for (const auto &[a, b, c] : mesh.triangles) {
    vec3 v0 = mesh.vertices[a];
    vec3 v1 = mesh.vertices[b];
    vec3 v2 = mesh.vertices[c];

    bool planar01 = face_info.EdgeAngle(a, b) < 1e-5;
    bool planar12 = face_info.EdgeAngle(b, c) < 1e-5;
    bool planar20 = face_info.EdgeAngle(c, a) < 1e-5;

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
        "{} stroke=\"#000\" />\n",
        v0x, v0y, v1x, v1y, v2x, v2y,
        backface ? "fill=\"#ffaaaa\" fill-opacity=\"0.9\"" :
                   "fill=\"#aaaaff\" fill-opacity=\"0.3\""
                 );

    {
      // And on the image.
      const auto &[v0x, v0y] = iscaler.Scale(v0.x, v0.y);
      const auto &[v1x, v1y] = iscaler.Scale(v1.x, v1.y);
      const auto &[v2x, v2y] = iscaler.Scale(v2.x, v2.y);

      img.BlendTriangle32(v0x, v0y, v1x, v1y, v2x, v2y,
                          0xFFFFFFBB);

      if (!planar01)
        img.BlendThickLine32(v0x, v0y, v1x, v1y, PX * 3,
                             0x000000AA);

      if (!planar12)
        img.BlendThickLine32(v1x, v1y, v2x, v2y, PX * 3,
                             0x000000AA);

      if (!planar20)
        img.BlendThickLine32(v2x, v2y, v0x, v0y, PX * 3,
                             0x000000AA);
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

  SaveSVG(infile, outfile);

  return 0;
}
