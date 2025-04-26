
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <tuple>
#include <utility>
#include <vector>

#include "ansi.h"
#include "bounds.h"
#include "color-util.h"
#include "image.h"
#include "mesh.h"
#include "nd-solutions.h"
#include "patches.h"
#include "polyhedra.h"
#include "status-bar.h"
#include "threadutil.h"
#include "yocto_matht.h"
#include "atomic-util.h"

using namespace yocto;

// Color map for spherical coordinates.
struct SphereMap {
  // For x,y,z on unit sphere.
  virtual uint32_t GetRGBA(const vec3 &v) {
    auto To = [](double c) {
        return 0.1 + ((c + 1.0) * 0.5 * 0.9);
      };

    uint32_t rgba = ColorUtil::FloatsTo32(
        To(v.x), To(v.y), To(v.z), 1.0);

    return rgba;
  }
};

/*
struct TriangularMesh3D {
  using vec3 = yocto::vec<double, 3>;
  std::vector<vec3> vertices;
  std::vector<std::tuple<int, int, int>> triangles;
};
*/

struct Texture {
  Texture() : image(1, 1) {}

  std::pair<int, int> Allocate(int width, int height) {
    // XXX use both dimensions.
    int xo = image.Width();
    image = image.Crop32(0, 0, image.Width() + width,
                         std::max(image.Height(), height));
    return std::make_pair(xo, 0);
  }

  void SetPixel32(int x, int y, uint32_t color) {
    image.SetPixel32(x, y, color);
  }

  vec2 NormalizeUV(const vec2 &uv) {
    return vec2{
      .x = uv.x / image.Width(),
      .y = uv.y / image.Height(),
    };
  }

  ImageRGBA image;
};

static vec3 BarycentricCoords(const vec2 &p, const vec2 &a, const vec2 &b,
                              const vec2 &c) {
  vec2 v0 = b - a, v1 = c - a, v2 = p - a;
  double d00 = dot(v0, v0);
  double d01 = dot(v0, v1);
  double d11 = dot(v1, v1);
  double d20 = dot(v2, v0);
  double d21 = dot(v2, v1);
  double denom = d00 * d11 - d01 * d01;
  if (std::abs(denom) < 1.0e-8) {
    // If degenerate, use the closest vertex.
    double dist_a = length_squared(p - a);
    double dist_b = length_squared(p - b);
    double dist_c = length_squared(p - c);
    if (dist_a <= dist_b && dist_a <= dist_c)
      return {1.0, 0.0, 0.0};
    if (dist_b <= dist_c)
      return {0.0, 1.0, 0.0};
    return {0.0, 0.0, 1.0};
  }
  double v = (d11 * d20 - d01 * d21) / denom;
  double w = (d00 * d21 - d01 * d20) / denom;
  double u = 1.0 - v - w;
  return {u, v, w};
}

// Generate a "sphere" as a pair of OBJ+MTL files.
static TexturedMesh TextureSphere() {
  TexturedMesh ret;
  ret.mesh = ApproximateSphere(0);

  SphereMap sphere_map;

  Texture texture;

  // pixel coordinates in the image.
  // Later we must scale these to [0, 1].
  std::vector<std::tuple<vec2, vec2, vec2>> puvs;

  // Approximate number of pixels of texture data we
  // try to get per triangle.
  constexpr int TARGET_PIXELS_PER_TRIANGLE = 256 * 256;

  for (const auto &[a, b, c] : ret.mesh.triangles) {
    const vec3 &va = ret.mesh.vertices[a];
    const vec3 &vb = ret.mesh.vertices[b];
    const vec3 &vc = ret.mesh.vertices[c];

    // Arbitrarily treat a as the home vertex.
    vec3 edge1 = vb - va;
    vec3 edge2 = vc - va;

    double edge1_len_sq = length_squared(edge1);
    double edge2_len_sq = length_squared(edge2);

    // Calculate the normal vector before normalization.
    vec3 triangle_perp = cross(edge1, edge2);
    double normal_len_sq = length_squared(triangle_perp);

    // If the triangle is degenerate, output zeroes for the uv
    // coordinates.
    if (normal_len_sq < 1.0e-18 ||
        edge1_len_sq < 1.0e-18 || edge2_len_sq < 1.0e-18) {
      ret.uvs.emplace_back(vec2{0,0}, vec2{0,0}, vec2{0,0});
      continue;
    }

    const vec3 triangle_normal = triangle_perp / sqrt(normal_len_sq);

    // Create an orthonormal basis (u_axis, v_axis) in the triangle's
    // plane.
    vec3 u_axis;

    // Choose the longer edge for defining u_axis for better stability.
    // Note: Always choosing edge1 to try to track down the bug!
    if (true || edge1_len_sq >= edge2_len_sq) {
      u_axis = edge1 / sqrt(edge1_len_sq);
    } else {
      u_axis = edge2 / sqrt(edge2_len_sq);
    }

    // Choose v axis to be perpendicular to normal and u axes (both unit).
    vec3 v_axis = cross(triangle_normal, u_axis);

    // Project vertices onto the 2D basis. va is the origin.
    vec2 pa{0.0, 0.0};
    vec2 pb{dot(edge1, u_axis), dot(edge1, v_axis)};
    vec2 pc{dot(edge2, u_axis), dot(edge2, v_axis)};

    double area = 0.5 * std::abs(pb.x * pc.y - pb.y * pc.x);

    if (area < 1e-12f) {
      ret.uvs.emplace_back(vec2{0,0}, vec2{0,0}, vec2{0,0});
      continue;
    }

    double scale = sqrt(TARGET_PIXELS_PER_TRIANGLE / area);

    // Calculate scaled 2D vertices (local pixel coordinates for the patch).
    vec2 uvA_local = pa * scale;
    vec2 uvB_local = pb * scale;
    vec2 uvC_local = pc * scale;

    // Calculate the bounding box of the local UVs.
    double min_x_local = std::min({uvA_local.x, uvB_local.x, uvC_local.x});
    double max_x_local = std::max({uvA_local.x, uvB_local.x, uvC_local.x});
    double min_y_local = std::min({uvA_local.y, uvB_local.y, uvC_local.y});
    double max_y_local = std::max({uvA_local.y, uvB_local.y, uvC_local.y});

    // Dimensions of the patch's bounding box in pixels.
    double patch_width_f = max_x_local - min_x_local;
    double patch_height_f = max_y_local - min_y_local;
    int patch_width = (int)std::ceil(patch_width_f);
    int patch_height = (int)std::ceil(patch_height_f);

    // Translate local UVs so the min corner is at (0,0).
    vec2 local_origin_offset = {min_x_local, min_y_local};
    uvA_local -= local_origin_offset;
    uvB_local -= local_origin_offset;
    uvC_local -= local_origin_offset;

    const auto [xoff, yoff] = texture.Allocate(patch_width, patch_height);
    vec2 offset{(double)xoff, (double)yoff};

    // Pixel coordinates in texture. We'll remap these to [0, 1] after
    // we know the final texture dimensions.
    vec2 final_uvA = offset + uvA_local;
    vec2 final_uvB = offset + uvB_local;
    vec2 final_uvC = offset + uvC_local;
    ret.uvs.emplace_back(final_uvA, final_uvB, final_uvC);

    // Rasterize the triangle patch into the atlas.
    // Calculate integer bounds for rasterization loop in atlas space.
    int start_x = (int)std::floor(offset.x);
    int end_x = (int)std::ceil(offset.x + patch_width_f);
    int start_y = (int)std::floor(offset.y);
    int end_y = (int)std::ceil(offset.y + patch_height_f);

    // Clamp loop bounds to texture dimensions.
    start_x = std::max(0, start_x);
    end_x = std::min(texture.image.Width(), end_x);
    start_y = std::max(0, start_y);
    end_y = std::min(texture.image.Height(), end_y);

    for (int py = start_y; py < end_y; py++) {
      for (int px = start_x; px < end_x; px++) {
        // Center of the current pixel in atlas coordinates.
        vec2 p_atlas = {(float)px + 0.5, (float)py + 0.5};
        // Convert pixel center to local patch coordinates.
        vec2 p_local = p_atlas - offset;

        // Calculate barycentric coordinates relative to the local patch UVs.
        vec3 bary =
          BarycentricCoords(p_local, uvA_local, uvB_local, uvC_local);

        // Check if the pixel center is inside the triangle.
        double epsilon = 1e-4f;
        if (bary.x >= -epsilon && bary.y >= -epsilon && bary.z >= -epsilon) {
          // Clamp weights to ensure they are valid [0, 1] multipliers.
          bary = {
            .x = std::clamp(bary.x, 0.0, 1.0),
            .y = std::clamp(bary.y, 0.0, 1.0),
            .z = std::clamp(bary.z, 0.0, 1.0),
          };

          // The whole point is to get the original 3D coordinate,
          // which we can then project to the sphere.
          vec3 p = normalize(va * bary.x + vb * bary.y + vc * bary.z);

          uint32_t color = sphere_map.GetRGBA(p);
          texture.SetPixel32(px, py, color);
        }
      }
    }
  }

  texture.image.Save("debug-texture.png");
  printf("Saved debug-texture.png\n");

  // Now renormalize uvs.
  for (auto &[uva, uvb, uvc] : ret.uvs) {
    uva = texture.NormalizeUV(uva);
    uvb = texture.NormalizeUV(uvb);
    uvc = texture.NormalizeUV(uvc);
  }

  ret.texture = std::move(texture.image);

  // ...
  return ret;
}




int main(int argc, char **argv) {
  ANSI::Init();

  TexturedMesh tmesh = TextureSphere();
  SaveAsOBJ(tmesh, "sphere");

  return 0;
}
