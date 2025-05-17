
#include <algorithm>
#include <array>
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
#include <unordered_map>
#include <utility>
#include <vector>

#include "ansi.h"
#include "arcfour.h"
#include "array-util.h"
#include "atomic-util.h"
#include "auto-histo.h"
#include "big-polyhedra.h"
#include "bounds.h"
#include "color-util.h"
#include "geom/tree-nd.h"
#include "image.h"
#include "map-util.h"
#include "mesh.h"
#include "nd-solutions.h"
#include "patches.h"
#include "periodically.h"
#include "polyhedra.h"
#include "randutil.h"
#include "status-bar.h"
#include "textured-mesh.h"
#include "threadutil.h"
#include "timer.h"
#include "yocto_matht.h"

using namespace yocto;

#define USE_ND 1

static constexpr int DIGITS = 24;

static inline vec2 VecFromPair(const std::pair<double, double> &p) {
  return vec2{.x = p.first, .y = p.second};
}

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

// Lazy scoremap using canonical patches.
struct ScoreMap {
  using Tree = TreeND<double, double>;

  // PERF: Can do more fine-grained locking!
  std::mutex m;

  ScoreMap(PatchInfo patch_info) :
    patch_info(patch_info), poly(BigScube(DIGITS)), boundaries(poly) {
  }

  double ClosestScore(const vec3 &v) {
    // PERF
    uint64_t code = boundaries.GetCode(v);

    auto sit = patch_info.all_codes.find(code);
    CHECK(sit != patch_info.all_codes.end());

    uint64_t canon_code = sit->second.canonical_code;
    vec3 canon_v = sit->second.patch_to_canonical.TransformPoint(v);

    return ClosestScoreCanonical(canon_code, canon_v);
  }

  double ClosestScoreCanonical(uint64_t canon_code,
                               const vec3 &canon_v) {
    Tree *tree = nullptr;
    m.lock();
    auto it = trees.find(canon_code);
    if (it == trees.end()) {
      Timer timer;
      printf("Load new tree for %llx\n", canon_code);
      tree = new Tree(3);
      trees[canon_code].reset(tree);

      // Insert all of the files in that patch.
      int64_t entries = 0;
      std::vector<std::unique_ptr<NDSolutions<6>>> shards;

      for (const auto &[icode, ipatch] : patch_info.canonical) {
        std::string filename = TwoPatchFilename(canon_code, icode);
        auto shard = std::make_unique<NDSolutions<6>>(filename);
        entries += shard->Size();
        shards.emplace_back(std::move(shard));
      }

      std::vector<std::pair<std::array<double, 3>, double>> batch;
      batch.reserve(entries);

      for (auto &shard : shards) {
        for (int i = 0; i < shard->Size(); i++) {
          const auto &[key, score, outer_, inner_] = (*shard)[i];
          // As the outer view.
          // tree->Insert(std::span(key.data(), 3), score);
          batch.emplace_back(SliceArray<0, 3>(key), score);
        }
      }

      printf("Init batch size (%lld):\n", (int64_t)batch.size());
      // ugh...
      std::vector<std::pair<std::span<const double>, double>> batch_arg;
      batch_arg.reserve(batch.size());
      for (const auto &[key, val] : batch) {
        batch_arg.emplace_back(std::span<const double>(key.data(), 3), val);
      }
      tree->InitBatch(std::move(batch_arg));

      printf("Loaded new tree (%lld entries) in %s.\n",
             entries,
             ANSI::Time(timer.Seconds()).c_str());
      m.unlock();

    } else {
      m.unlock();
      tree = it->second.get();
    }

    CHECK(tree != nullptr);

    const auto &[pos, score, dist] = tree->Closest(canon_v);

    #if 0
    Timer sample_timer;
    auto dcr = tree->DebugClosest(canon_v);
    const auto &[pos, score, dist] = dcr.res;

    printf("One sample took %s; got\n"
           "Score: %.17g\n"
           "Dist: %.17g\n",
           ANSI::Time(sample_timer.Seconds()).c_str(), score, dist);

    printf("leaves_searched: %lld (%.5f%%)\n",
           dcr.leaves_searched,
           (dcr.leaves_searched * 100.0) / tree->Size());
    printf("new_best: %lld\n", dcr.new_best);
    printf("max_depth: %lld\n", dcr.max_depth);
    printf("heap_pops: %lld\n", dcr.heap_pops);
    printf("max_heap_size: %lld\n", dcr.max_heap_size);
    printf("-----------------\n");
    #endif

    return score;
  }

  std::unordered_map<uint64_t, std::unique_ptr<Tree>> trees;

  PatchInfo patch_info;
  BigPoly poly;
  Boundaries boundaries;
};


struct TwoPatchOuterSphereMap : public SphereMap {
  // Precomputed, since we don't want to have to load all shards
  // just to get the bounds.
  static constexpr double min_score = 4.3511678576336583e-15;
  static constexpr double max_score = 4.8285280901252183e-08;
  static constexpr double bound_span = max_score - min_score;

  TwoPatchOuterSphereMap() : score_map(LoadPatchInfo("scube-patchinfo.txt")) {

  }

  uint32_t ColorScore(double score) const {
    return ColorUtil::LinearGradient32(ColorUtil::VISIBLE_SPECTRUM,
                                       (score - min_score) / bound_span);
  }

  uint32_t GetRGBA(const vec3 &v) override {
    return ColorScore(score_map.ClosestScore(v));
  }

 private:
  ScoreMap score_map;
};

#if 0
struct Texture {
  Texture() : Texture(1, 1) {}
  Texture(int width, int height) : image(width, height) {}
  Texture(Texture &&other) = default;
  Texture(const Texture &other) = default;
  Texture &operator=(const Texture &other) = default;
  Texture &operator=(Texture &&other) = default;

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
#endif

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
  ImageRGBA texture;
  std::tuple<vec2, vec2, vec2> uvs;
};

static constexpr int TRIANGLE_TEXTURE_EDGE = 32;
TexturedTriangle TextureOneTriangle(const TriangularMesh3D &mesh,
                                    SphereMap *sphere_map,
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
      edge1_len_sq < 1.0e-18 ||
      edge2_len_sq < 1.0e-18) {
    TexturedTriangle tex{
      .texture = ImageRGBA(),
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

  if (area < 1.0e-12) {
    TexturedTriangle tex{
      .texture = ImageRGBA(1, 1),
      .uvs = {vec2{0,0}, vec2{0,0}, vec2{0,0}}
    };
    return tex;
  }

  // Use the bounding box to determine the scale.
  Bounds bounds;
  bounds.Bound(pa);
  bounds.Bound(pb);
  bounds.Bound(pc);

  std::mutex mu;
  ImageRGBA texture(TRIANGLE_TEXTURE_EDGE, TRIANGLE_TEXTURE_EDGE);

  Bounds::Scaler scaler = bounds.ScaleToFit(TRIANGLE_TEXTURE_EDGE,
                                            TRIANGLE_TEXTURE_EDGE);

  // Calculate scaled 2D vertices (pixel coordinates in the texture).
  vec2 pxa = VecFromPair(scaler.Scale(pa));
  vec2 pxb = VecFromPair(scaler.Scale(pb));
  vec2 pxc = VecFromPair(scaler.Scale(pc));

  // Bounds in pixel space.
  int start_x = (int)std::floor(scaler.ScaleX(bounds.MinX()));
  int start_y = (int)std::floor(scaler.ScaleY(bounds.MinY()));
  int end_x = (int)std::ceil(scaler.ScaleX(bounds.MaxX()));
  int end_y = (int)std::ceil(scaler.ScaleY(bounds.MaxY()));


  if (end_y > start_y && end_x > start_x) {
    ParallelComp2D(
        end_x - start_x,
        end_y - start_y,
        [&](int64_t ox, int64_t oy) {
          const int px = ox + start_x;
          const int py = oy + start_y;

          // Center of the current pixel in uv coordinates.
          vec2 pt = VecFromPair(scaler.Unscale(px + 0.5, py + 0.5));

          // Calculate barycentric coordinates relative to the local
          // patch UVs.
          vec3 bary = BarycentricCoords(pt, pa, pb, pc);

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
    // Map to [0,1] rectangle. XXX might need to flip Y?
    .uvs = {
      pxa / TRIANGLE_TEXTURE_EDGE,
      pxb / TRIANGLE_TEXTURE_EDGE,
      pxc / TRIANGLE_TEXTURE_EDGE,
    },
  };
}

// Generate a "sphere" as a pair of OBJ+MTL files.
static TexturedMesh TextureSphere(SphereMap *sphere_map) {
  TexturedMesh ret;
  ret.mesh = ApproximateSphere(0);

  StatusBar status(1);
  Periodically status_per(1.0);
  status.Statusf("Texturing %lld triangles.",
                 ret.mesh.triangles.size());

  for (int triangle_idx = 0; triangle_idx < ret.mesh.triangles.size();
       triangle_idx++) {

    // One texture image per triangle.
    TexturedTriangle tex =
      TextureOneTriangle(ret.mesh, sphere_map, triangle_idx);

    const auto &[uu, vv] = ret.texture.AddTexture(std::move(tex.texture));
    vec2 uvidx(uu, vv);
    const auto &[uva, uvb, uvc] = tex.uvs;
    ret.uvs.push_back(
        std::make_tuple(uvidx + uva, uvidx + uvb, uvidx + uvc));

    status_per.RunIf([&]() {
        status.Progress(triangle_idx,
                        ret.mesh.triangles.size(),
                        "Texturing.");
      });
  }

  // ...
  return ret;
}

static void Generate() {
  // std::unique_ptr<SphereMap> sphere_map(
  // new TwoPatchOuterSphereMap);
  std::unique_ptr<SphereMap> sphere_map(new SphereMap);
  TexturedMesh tmesh = TextureSphere(sphere_map.get());
  SaveAsOBJ(tmesh, "sphere");
}

[[maybe_unused]]
static void TreeBench1() {
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

static void TreeBench2() {
  PatchInfo patch_info = LoadPatchInfo("scube-patchinfo.txt");
  ScoreMap score_map(patch_info);

  StatusBar status(1);

  ArcFour rc("deterministic");
  std::vector<vec3> samples;
  for (int i = 0; i < 5; i++) {
    vec3 v{RandDouble(&rc), RandDouble(&rc), RandDouble(&rc)};
    v /= length(v);
    samples.push_back(v);
  }

  for (int i = 0; i < 2; i++) {
    printf("==== Pass %d ====\n", i + 1);
    for (const vec3 &v : samples) {
      [[maybe_unused]]
      double d = score_map.ClosestScore(v);
    }
  }
}

int main(int argc, char **argv) {
  ANSI::Init();

  // TreeBench2();

  Generate();

  return 0;
}
