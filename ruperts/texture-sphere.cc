
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <limits>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "ansi.h"
#include "arcfour.h"
#include "auto-histo.h"
#include "bounds.h"
#include "color-util.h"
#include "image.h"
#include "map-util.h"
#include "mesh.h"
#include "nd-solutions.h"
#include "patches.h"
#include "periodically.h"
#include "polyhedra.h"
#include "randutil.h"
#include "status-bar.h"
#include "threadutil.h"
#include "timer.h"
#include "yocto_matht.h"
#include "atomic-util.h"
#include "geom/tree-nd.h"
#include "geom/tree-3d.h"
#include "patches.h"

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

  virtual ~SphereMap() {}
};

#define USE_ND 1

#if 0
struct TwoPatchOuterSphereMap : public SphereMap {
  TwoPatchOuterSphereMap() {
    PatchInfo patch_info = LoadPatchInfo("scube-patchinfo.txt");
    AutoHisto histo(100000);

    StatusBar status(1);
    int progress = 0;

    std::vector<std::pair<uint64_t, PatchInfo::CanonicalPatch>> cc =
      MapToSortedVec(patch_info.canonical);

    min_bound = std::numeric_limits<double>::infinity();
    max_bound = -std::numeric_limits<double>::infinity();

    for (const auto &[ocode, opatch] : cc) {
      for (const auto &[icode, ipatch] : cc) {
        std::string filename = TwoPatchFilename(ocode, icode);
        NDSolutions<6> shard(filename);
        status.Printf("Shard %s has %lld", filename.c_str(), shard.Size());
        for (int i = 0; i < shard.Size(); i++) {
          const auto &[key, score, outer_, inner_] = shard[i];
          // As the outer view.
          #if USE_ND
          tree.Insert(std::span(key.data(), 3), score);
          #else
          tree.Insert(key[0], key[1], key[2], score);
          #endif
          min_bound = std::min(min_bound, score);
          max_bound = std::max(max_bound, score);
          histo.Observe(score);
        }
      }
      progress++;
      status.Progressf(progress, patch_info.canonical.size(),
                       "Loaded %lld", tree.Size());

      status.Printf("Histo:\n"
                    "%s\n",
                    histo.SimpleANSI(32).c_str());
    }

    bound_span = max_bound - min_bound;
    printf("%lld samples total.\n", tree.Size());
    printf("Spanning %.17g\n"
           "Min: %s%.17g" ANSI_RESET "\n"
           "Max: %s%.17g" ANSI_RESET "\n",
           bound_span,
           ANSI::ForegroundRGB32(ColorScore(min_bound)).c_str(), min_bound,
           ANSI::ForegroundRGB32(ColorScore(max_bound)).c_str(), max_bound);

    printf("Histo:\n"
           "%s\n",
           histo.SimpleANSI(32).c_str());

    {
      Timer sample_timer;
      uint32_t c = GetRGBA(vec3{1.0, 0.0, 0.0});
      printf("One sample took %s; got %08x\n",
             ANSI::Time(sample_timer.Seconds()).c_str(), c);
    }
  }

  uint32_t ColorScore(double score) {
    return ColorUtil::LinearGradient32(ColorUtil::VISIBLE_SPECTRUM,
                                       (score - min_bound) / bound_span);
  }

  uint32_t GetRGBA(const vec3 &v) override {
    if (tree.Empty()) return 0;

    #if USE_ND
    const auto &[pos, score, dist] = tree.Closest(v);
    #else
    const auto &[pos, score, dist] = tree.Closest(v.x, v.y, v.z);
    #endif
    return ColorScore(score);
  }

 private:
  double min_bound = 0, max_bound = 0, bound_span = 0;
  #if USE_ND
  TreeND<double, double> tree = TreeND<double, double>(3);
  #else
  Tree3D<double, double> tree;
  #endif
};

struct Texture {
  Texture() : image(1, 1) {}
  Texture(int width, int height) : image(width, height) {}

  std::pair<int, int> Allocate(int width, int height) {
    if (xpos + width > MAX_WIDTH) {
      ypos += row_height;
      xpos = 0;
    }

    if (xpos + width > image.Width() || ypos + height > image.Height()) {
      image = image.Crop32(0, 0,
                           std::max(image.Width(), xpos + width),
                           std::max(image.Height(), ypos + height));
    }

    std::pair<int, int> ret = {xpos, ypos};
    xpos += width;
    row_height = std::max(row_height, height);
    return ret;
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

  int xpos = 0, ypos = 0;
  static constexpr int MAX_WIDTH = 8192;
  int row_height = 1;

  ImageRGBA image;
};

// UDIM textures store more than one unit square, which can also be of
// different resolution.
struct UDimTexture {
  UDimTexture() {}

  // The image is always square; power of two is conventional.
  std::pair<int, int> Allocate(int side_pixels) {
    CHECK(side_pixels > 0);
    int size = (int)textures.size();
    CHECK(size < 9000);
    textures.emplace_back(side_pixels, side_pixels);
    return UnpackUV(size);
  }

  // From an index in the textures vector.
  static std::pair<int, int> UnpackUV(int index) {
    int u = index % 10;
    int v = index / 10;
    return std::make_pair(u, v);
  }

  static int PackUV(int u, int v) {
    return v * 10 + u;
  }

  // Always a four-digit number in [1001, 9999].
  static int Filenum(int u, int v) {
    CHECK(u >= 0 && v >= 0 && u < 10 && v < 899);
    return 1001 + (v * 10) + u;
  }

  Texture &Texture(int u, int v) {
    CHECK(u >= 0 && v >= 0 && u < 10);
    int idx = PackUV(u, v);
    CHECK(idx < textures.size());
    return textures[idx];
  }

 private:
  // Row-major. There are always 10 columns (max U is 10.0), but V can
  // go to about 900.
  std::vector<Texture> textures;
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

struct TexturedTriangle {
  Texture texture;
  std::tuple<vec2, vec2, vec2> uvs;
};

static constexpr int TRIANGLE_TEXTURE_EDGE = 1024;
TexturedTriangle TextureOneTriangle(const TriangularMesh3D &mesh,
                                    int triangle_idx) {
  const auto &[a, b, c] = mesh.triangles[triangle_idx];
  const vec3 &va = mesh.vertices[a];
  const vec3 &vb = mesh.vertices[b];
  const vec3 &vc = mesh.vertices[c];

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
    TexturedTriangle tex{
      .texture = Texture(),
      .uvs = {vec2{0,0}, vec2{0,0}, vec2{0,0}}
    };
    return tex;
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
    TexturedTriangle tex{
      .texture = Texture(1, 1),
      .uvs = {vec2{0,0}, vec2{0,0}, vec2{0,0}}
    };
    return tex;
  }

  // Approximate number of pixels of texture data we
  // try to get per triangle. XXX: Use the texture size.
  constexpr int TARGET_PIXELS_PER_TRIANGLE = 256 * 256;
  // constexpr int TARGET_PIXELS_PER_TRIANGLE = 32 * 32;

  double scale = sqrt(TARGET_PIXELS_PER_TRIANGLE / area);

  // Calculate scaled 2D vertices (local pixel coordinates for the patch).
  vec2 uva = pa * scale;
  vec2 uvb = pb * scale;
  vec2 uvc = pc * scale;

  // uv bounding box.
  double min_x_local = std::min({uva.x, uvb.x, uvc.x});
  double max_x_local = std::max({uva.x, uvb.x, uvc.x});
  double min_y_local = std::min({uva.y, uvb.y, uvc.y});
  double max_y_local = std::max({uva.y, uvb.y, uvc.y});

  // Dimensions of the patch's bounding box in pixels.
  double patch_width_f = max_x_local - min_x_local;
  double patch_height_f = max_y_local - min_y_local;
  int patch_width = (int)std::ceil(patch_width_f);
  int patch_height = (int)std::ceil(patch_height_f);

  // Translate local UVs so the min corner is at (0,0).
  vec2 local_origin_offset = {min_x_local, min_y_local};
  uva -= local_origin_offset;
  uvb -= local_origin_offset;
  uvc -= local_origin_offset;

  Texture texture(TRIANGLE_TEXTURE_EDGE,
                  TRIANGLE_TEXTURE_EDGE);

  // const auto [xoff, yoff] = texture.Allocate(patch_width, patch_height);
  // vec2 offset{(double)xoff, (double)yoff};

  // Pixel coordinates in texture. We'll remap these to [0, 1] after
  // we know the final texture dimensions.
  vec2 texture_uva = offset + uva;
  vec2 texture_uvb = offset + uvb;
  vec2 texture_uvc = offset + uvc;

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

  // Guard parallel access to texture.
  std::mutex mu;
  Texture texture(TRIANGLE_TEXTURE_EDGE, TRIANGLE_TEXTURE_EDGE);

  if (end_y > start_y && end_x > start_x) {
    ParallelComp2D(
        end_x - start_x,
        end_y - start_y,
        [&](int64_t ox, int64_t oy) {
          const int px = ox + start_x;
          const int py = oy + start_y;

          // Center of the current pixel in atlas coordinates.
          vec2 p_atlas = {(float)px + 0.5, (float)py + 0.5};
          // Convert pixel center to local patch coordinates.
          vec2 p_local = p_atlas - offset;

          // Calculate barycentric coordinates relative to the local
          // patch UVs.
          vec3 bary =
            BarycentricCoords(p_local, uva, uvb, uvc);

          // Check if the pixel center is inside the triangle.
          double epsilon = 1e-4f;
          if (bary.x >= -epsilon &&
              bary.y >= -epsilon &&
              bary.z >= -epsilon) {
            // Clamp weights to ensure they are valid [0, 1] multipliers.
            bary = {
              .x = std::clamp(bary.x, 0.0, 1.0),
              .y = std::clamp(bary.y, 0.0, 1.0),
              .z = std::clamp(bary.z, 0.0, 1.0),
            };

            // The whole point is to get the original 3D coordinate,
            // which we can then project to the sphere.
            vec3 p = normalize(va * bary.x + vb * bary.y + vc * bary.z);

            uint32_t color = sphere_map->GetRGBA(p);
            {
              MutexLock ml(&mu);
              texture.SetPixel32(px, py, color);
            }
          }
        },
        8);
  }

  return TexturedTriangle{
    .texture = std::move(texture),
    // XX should probably remap here.
    .uvs = { texture_uva, texture_uvb, texture_uvc },
  };
}

// Generate a "sphere" as a pair of OBJ+MTL files.
static TexturedMesh TextureSphere(SphereMap *sphere_map) {
  TexturedMesh ret;
  ret.mesh = ApproximateSphere(0);

  Texture texture;
  StatusBar status(1);
  Periodically status_per(1.0);
  status.Statusf("Texturing %lld triangles.",
                 ret.mesh.triangles.size());

  // pixel coordinates in the image.
  // Later we must scale these to [0, 1].
  std::vector<std::tuple<vec2, vec2, vec2>> puvs;

  for (int triangle_idx = 0; triangle_idx < ret.mesh.triangles.size();
       triangle_idx++) {

    TexturedTriangle tex = TextureOneTriangle(ret.mesh, triangle_idx);
    // XXX copy tex into UDims.
    LOG(FATAL) << "Broken in refactoring";

    status_per.RunIf([&]() {
        status.Progressf(triangle_idx,
                         ret.mesh.triangles.size(),
                         "Texturing.");
      });
  }

  // texture.image.Save("debug-texture.png");
  // printf("Saved debug-texture.png\n");

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

static void Generate() {
  std::unique_ptr<SphereMap> sphere_map(
      new TwoPatchOuterSphereMap);
  TexturedMesh tmesh = TextureSphere(sphere_map.get());
  SaveAsOBJ(tmesh, "sphere");

}
#endif

static void TreeBench() {
  PatchInfo patch_info = LoadPatchInfo("scube-patchinfo.txt");
  AutoHisto histo(100000);

  StatusBar status(1);
  int progress = 0;

  std::vector<std::pair<uint64_t, PatchInfo::CanonicalPatch>> cc =
    MapToSortedVec(patch_info.canonical);

  double min_bound = 0, max_bound = 0;
  min_bound = std::numeric_limits<double>::infinity();
  max_bound = -std::numeric_limits<double>::infinity();

  TreeND<double, double> tree = TreeND<double, double>(3);

  Timer load_timer;

  std::vector<std::string> filenames;
  for (const auto &[ocode, opatch] : cc) {
    for (const auto &[icode, ipatch] : cc) {
      std::string filename = TwoPatchFilename(ocode, icode);
      filenames.push_back(filename);
    }
  }
  {
    ArcFour rc("shuffle");
    Shuffle(&rc, &filenames);
  }

  Periodically status_per(5.0);
  for (const std::string &filename: filenames) {
    // std::string filename = TwoPatchFilename(ocode, icode);
    NDSolutions<6> shard(filename);
    status.Printf("Shard %s has %lld", filename.c_str(), shard.Size());
    for (int i = 0; i < shard.Size(); i++) {
      const auto &[key, score, outer_, inner_] = shard[i];
      // As the outer view.
      tree.Insert(std::span(key.data(), 3), score);
      min_bound = std::min(min_bound, score);
      max_bound = std::max(max_bound, score);
      histo.Observe(score);
    }

    progress++;
    status_per.RunIf([&]() {
        status.Progressf(progress, filenames.size(),
                         "Loaded %lld", tree.Size());

        status.Printf("Histo:\n"
                      "%s\n",
                      histo.SimpleANSI(32).c_str());
      });
  }

  printf("Loaded all shards in %s\n",
         ANSI::Time(load_timer.Seconds()).c_str());

  double bound_span = max_bound - min_bound;
  auto ColorScore = [min_bound, bound_span](double score) -> uint32_t {
    return ColorUtil::LinearGradient32(ColorUtil::VISIBLE_SPECTRUM,
                                       (score - min_bound) / bound_span);
    };

  [[maybe_unused]]
  auto GetRGBA = [&](const vec3 &v) -> uint32_t {
    if (tree.Empty()) return 0;

    const auto &[pos, score, dist] = tree.Closest(v);
    return ColorScore(score);
    };

  printf("%lld samples total.\n", tree.Size());
  printf("Spanning %.17g\n"
         "Min: %s%.17g" ANSI_RESET "\n"
         "Max: %s%.17g" ANSI_RESET "\n",
         bound_span,
         ANSI::ForegroundRGB32(ColorScore(min_bound)).c_str(), min_bound,
         ANSI::ForegroundRGB32(ColorScore(max_bound)).c_str(), max_bound);

  printf("Histo:\n"
         "%s\n",
         histo.SimpleANSI(32).c_str());

  CHECK(!tree.Empty());

  ArcFour rc("deterministic");
  {
    for (int i = 0; i < 5; i++) {
      vec3 v{RandDouble(&rc), RandDouble(&rc), RandDouble(&rc)};
      v /= length(v);

      Timer sample_timer;
      auto dcr = tree.DebugClosest(v);
      const auto &[pos, score, dist] = dcr.res;

      printf("One sample took %s; got\n"
             "Score: %.17g\n"
             "Dist: %.17g\n",
             ANSI::Time(sample_timer.Seconds()).c_str(), score, dist);

      printf("leaves_searched: %lld (%.5f%%)\n",
             dcr.leaves_searched,
             (dcr.leaves_searched * 100.0) / tree.Size());
      printf("new_best: %lld\n", dcr.new_best);
      printf("max_depth: %lld\n", dcr.max_depth);
      printf("heap_pops: %lld\n", dcr.heap_pops);
      printf("max_heap_size: %lld\n", dcr.max_heap_size);
      printf("-----------------\n");
    }
  }

}

int main(int argc, char **argv) {
  ANSI::Init();

  TreeBench();

  // Generate();

  return 0;
}
