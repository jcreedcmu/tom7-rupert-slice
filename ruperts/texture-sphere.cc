
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <limits>
#include <memory>
#include <mutex>
#include <numbers>
#include <optional>
#include <span>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>
#include <semaphore>

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
#include "util.h"
#include "yocto_matht.h"

DECLARE_COUNTERS(outside_triangle);

using namespace yocto;

#define USE_ND 1

static constexpr int DIGITS = 24;

// on each side
// static constexpr int MULTISAMPLE = 16;
static constexpr int MULTISAMPLE = 16;

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

// Global status.
static StatusBar status(1);

// Mainly to prevent exceeding system memory, since we also
// store a temporary copy of the data while processing it
// into the nd-tree.
static std::counting_semaphore load_semaphore(8);

// Lazy scoremap using canonical patches.
struct ScoreMap {
  using Tree = TreeND<double, double>;

  // Locks just the tree map. The trees may not be mutated once they
  // are loaded.
  std::mutex m;

  ScoreMap(PatchInfo patch_info) :
    patch_info(patch_info), poly(BigScube(DIGITS)), boundaries(poly) {
  }

  double ClosestScore(const vec3 &v) {
    // PERF
    uint64_t code = boundaries.GetCodeSloppy(v);

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
      // Load, but not holding the lock. We mark that we
      // are loading by putting a nullptr in the tree.
      trees[canon_code].reset(nullptr);
      const int task_num = (int)trees.size();
      m.unlock();

      Timer load_wait;
      load_semaphore.acquire();
      Timer timer;
      status.Print("#{} (waited {}) Load " ACYAN("{:x}") " (for {})\n",
                   task_num,
                   ANSI::Time(load_wait.Seconds()),
                   canon_code, VecString(canon_v));
      tree = new Tree(3);

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

      status.Print("Init batch size " AWHITE("{}") ":\n", (int64_t)batch.size());
      // ugh...
      std::vector<std::pair<std::span<const double>, double>> batch_arg;
      batch_arg.reserve(batch.size());
      for (const auto &[key, val] : batch) {
        batch_arg.emplace_back(std::span<const double>(key.data(), 3), val);
      }
      tree->InitBatch(std::move(batch_arg));

      // Free memory before releasing semaphore.
      shards.clear();
      load_semaphore.release();

      m.lock();
      auto &up = trees[canon_code];
      CHECK(up.get() == nullptr) << "Someone else initialized?";
      up.reset(tree);
      m.unlock();

      status.Print("Loaded new tree ({} entries) in {}.\n",
                   entries,
                   ANSI::Time(timer.Seconds()));

    } else {
      auto &up = it->second;

      for (;;) {
        if (up.get() != nullptr) {
          tree = up.get();
          m.unlock();
          break;
        }

        // Otherwise, someone else is loading it. Let them.
        m.unlock();

        // PERF: Could use condition variable here, but this
        // is only happening during load time and a 100ms delay
        // is trivial in comparison.
        using namespace std::chrono_literals;
        std::this_thread::sleep_for(100ms);

        m.lock();
      }
    }

    CHECK(tree != nullptr);

    const auto &[pos, score, dist] = tree->Closest(canon_v);
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
  /*
  static constexpr double min_score = 4.3511678576336583e-15;
  static constexpr double max_score = 4.8285280901252183e-08;
  static constexpr double bound_span = max_score - min_score;
  */

  std::vector<double> quantiles;

  TwoPatchOuterSphereMap() : score_map(LoadPatchInfo("scube-patchinfo.txt")) {
    for (const std::string &line :
           Util::ReadFileToLines("scube-score-quantiles.txt")) {
      std::optional<double> od = Util::ParseDoubleOpt(line);
      if (od.has_value()) {
        quantiles.push_back(od.value());
      }
    }
  }

  double ScoreToRank(double score) const {
    auto it = std::upper_bound(quantiles.begin(), quantiles.end(), score);
    size_t index = it - quantiles.begin();
    return index / (double)quantiles.size();
  }

  uint32_t ColorScore(double score) const {
    /*
    return ColorUtil::LinearGradient32(ColorUtil::VISIBLE_SPECTRUM,
                                       (score - min_score) / bound_span);
    */
    return ColorUtil::LinearGradient32(ColorUtil::VISIBLE_SPECTRUM,
                                       ScoreToRank(score));
  }

  uint32_t GetRGBA(const vec3 &v) override {
    double s = ScoreToRank(score_map.ClosestScore(v));
    // XXX ad hoc
    // s = std::clamp(s, 0.0, 1.0);
    // s = std::clamp(s * s, 0.0, 1.0);
    return ColorScore(s);
  }

 private:
  ScoreMap score_map;
};


struct PatchIDMap : public SphereMap {
  PatchInfo patch_info;
  BigPoly poly;
  Boundaries boundaries;
  PatchIDMap() :
    patch_info(LoadPatchInfo("scube-patchinfo.txt")),
    poly(BigScube(DIGITS)), boundaries(poly) {
    ArcFour rc("deterministic");
    std::unordered_map<uint64_t, double> canonical_hue;
    int h = 0;
    for (const auto &[code, _] : patch_info.canonical) {
      canonical_hue[code] = h / (double)patch_info.canonical.size();
      h++;
    }

    for (const auto &[code, patch] : patch_info.all_codes) {
      double hue = canonical_hue[patch.canonical_code];
      bool is_canon = canonical_hue.contains(code);
      colored_codes[code] =
        ColorUtil::HSVAToRGBA32(
            hue,
            (is_canon ? 1.0 : 0.25 + RandDouble(&rc) * 0.5),
            (is_canon ? 1.0 : 0.4 + RandDouble(&rc) * 0.5),
            1.0);
    }
  }

  uint32_t GetRGBA(const vec3 &v) override {
    uint64_t code = boundaries.GetCodeSloppy(v);
    auto it = colored_codes.find(code);
    if (it == colored_codes.end()) return 0x000000FF;
    return it->second;
  }

 private:
  std::unordered_map<uint64_t, uint32_t> colored_codes;
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
  ImageRGBA texture;
  std::tuple<vec2, vec2, vec2> uvs;
};

// Move va away from the edge vb-vc by the given amount.
static vec2 ExpandVertex(
    const vec2 &va,
    const vec2 &vb,
    const vec2 &vc,
    double amount) {

  vec2 edge = vc - vb;
  double edge_len_sq = length_squared(edge);

  // Handle degenerate opposite edge by moving away from the point
  // instead.
  if (edge_len_sq < 1.0e-12) {
    vec2 dir_away = va - vb;
    if (length_squared(dir_away) < 1.0e-12) {
      return va;
    }
    return va + normalize(dir_away) * amount;
  }

  // Normal to the edge vector.
  vec2 normal = {edge.y, -edge.x};
  normal = normalize(normal);

  // Handle both winding orders by flipping if necessary.
  vec2 vec_to_vertex = va - vb;
  if (dot(normal, vec_to_vertex) < 0.0) {
    normal = -normal;
  }

  return va + normal * amount;
}

static std::tuple<vec2, vec2, vec2> ExpandTriangle(
    const vec2 &va, const vec2 &vb, const vec2 &vc,
    double amount) {
  return std::make_tuple(ExpandVertex(va, vb, vc, amount),
                         ExpandVertex(vb, va, vc, amount),
                         ExpandVertex(vc, va, vb, amount));
}

// static constexpr int TRIANGLE_TEXTURE_EDGE = 8192;
// static constexpr int TRIANGLE_TEXTURE_EDGE = 8192;
static constexpr int TRIANGLE_TEXTURE_EDGE = 4096;
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

  static constexpr int MARGIN_PIXELS = TRIANGLE_TEXTURE_EDGE >> 7;
  Bounds::Scaler scaler = bounds.ScaleToFitWithMargin(
      TRIANGLE_TEXTURE_EDGE,
      TRIANGLE_TEXTURE_EDGE,
      MARGIN_PIXELS);

  // Calculate scaled 2D vertices (pixel coordinates in the texture).
  vec2 pxa = VecFromPair(scaler.Scale(pa));
  vec2 pxb = VecFromPair(scaler.Scale(pb));
  vec2 pxc = VecFromPair(scaler.Scale(pc));

  // Screen triangle used for bounding.
  const auto &[ta, tb, tc] =
    ExpandTriangle(pxa, pxb, pxc, MARGIN_PIXELS * std::numbers::sqrt2);

  // Rasterization rectangle.
  Bounds raster_bounds;
  raster_bounds.Bound(ta);
  raster_bounds.Bound(tb);
  raster_bounds.Bound(tc);

  int start_x = std::max(0, (int)std::floor(raster_bounds.MinX()));
  int start_y = std::max(0, (int)std::floor(raster_bounds.MinY()));
  int end_x = std::min(texture.Width(),
                       (int)std::ceil(raster_bounds.MaxX()));
  int end_y = std::min(texture.Height(),
                       (int)std::ceil(raster_bounds.MaxY()));

  if (end_y > start_y && end_x > start_x) {
    ParallelComp2D(
        end_x - start_x,
        end_y - start_y,
        [&](int64_t ox, int64_t oy) {
          const int px = ox + start_x;
          const int py = oy + start_y;

          // It has to be in the bounding triangle.
          if (!InTriangle(ta, tb, tc, vec2(px + 0.5, py + 0.5))) {
            outside_triangle++;
            return;
          }

          // Assuming a = 0xFF.
          uint32_t r = 0;
          uint32_t g = 0;
          uint32_t b = 0;
          for (int yy = 0; yy < MULTISAMPLE; yy++) {
            for (int xx = 0; xx < MULTISAMPLE; xx++) {

              // Sample the center of the pixel if multisample=1,
              // but one quarter and three quarters if multisample=2, etc.
              constexpr float subdiv = 1.0f / (1 + MULTISAMPLE);

              // The center of the pixel.
              vec2 screen_pt = vec2(px + (xx + 1) * subdiv,
                                    py + (yy + 1) * subdiv);

              // That point in the triangle's local coordinate system.
              vec2 pt = VecFromPair(scaler.Unscale(screen_pt));

              // The barycentric coordinates of the sample point. This may
              // be outside the triangle.
              vec3 bary = BarycentricCoords(pt, pa, pb, pc);

              // This is the interpolated (or extrapolated) point on the
              // triangle's plane (world coordinates).
              vec3 plane_pt = va * bary.x + vb * bary.y + vc * bary.z;

              // The whole point is to get the original 3D coordinate,
              // which we can then project to the sphere.
              vec3 p = normalize(plane_pt);

              const auto &[rr, gg, bb, aa_] = ColorUtil::Unpack32(
                  sphere_map->GetRGBA(p));
              r += rr;
              g += gg;
              b += bb;
            }
          }

          r /= MULTISAMPLE * MULTISAMPLE;
          g /= MULTISAMPLE * MULTISAMPLE;
          b /= MULTISAMPLE * MULTISAMPLE;

          auto Clamp = [](int c) -> uint8_t { return std::clamp(c, 0, 255); };

          uint32_t color = ColorUtil::Pack32(Clamp(r), Clamp(g), Clamp(b),
                                             0xFF);

          {
            MutexLock ml(&mu);
            texture.SetPixel32(px, py, color);
          }
        },
        8);
  } else {
    status.Printf(ARED("Empty triangle?") " %d %d to %d %d\n"
                  "pxa: %s\n"
                  "pxb: %s\n"
                  "pxc: %s\n"
                  "ta: %s\n"
                  "tb: %s\n"
                  "tc: %s\n",
                  start_x, start_y,
                  end_x, end_y,
                  VecString(pxa).c_str(),
                  VecString(pxb).c_str(),
                  VecString(pxc).c_str(),
                  VecString(ta).c_str(),
                  VecString(tb).c_str(),
                  VecString(tc).c_str());
  }

  return TexturedTriangle{
    .texture = std::move(texture),
    // Map to [0,1] rectangle. Note that SaveAsOBJ flips the v coordinate.
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
  // ret.mesh = ApproximateSphere(2);
  ret.mesh = PolyToTriangularMesh(SnubCube());

  Periodically status_per(1.0);
  status.Status("Texturing {} triangles...",
                ret.mesh.triangles.size());

  for (int triangle_idx = 0; triangle_idx < ret.mesh.triangles.size();
       triangle_idx++) {

    Timer timer;
    // One texture image per triangle.
    TexturedTriangle tex =
      TextureOneTriangle(ret.mesh, sphere_map, triangle_idx);

    std::string tmpfile = "texture-sphere.tmp.png";
    tex.texture.Save(tmpfile);
    status.Print("Saved " AGREEN("{}"), tmpfile);

    const auto &[uu, vv] = ret.texture.AddTexture(std::move(tex.texture));
    vec2 uvidx(uu, vv);
    const auto &[uva, uvb, uvc] = tex.uvs;
    ret.uvs.push_back(
        std::make_tuple(uvidx + uva, uvidx + uvb, uvidx + uvc));

    status_per.RunIf([&]() {
        status.Print("Finished triangle {} in {}",
                     triangle_idx, ANSI::Time(timer.Seconds()));
        status.Progress(triangle_idx + 1,
                        ret.mesh.triangles.size(),
                        "Texturing.");
      });
  }

  printf("Done. %lld outside\n",
         outside_triangle.Read());

  // ...
  return ret;
}

static void Generate() {
  Timer timer;
  std::unique_ptr<SphereMap> sphere_map(new TwoPatchOuterSphereMap);
  // std::unique_ptr<SphereMap> sphere_map(new PatchIDMap);
  // std::unique_ptr<SphereMap> sphere_map(new SphereMap);
  TexturedMesh tmesh = TextureSphere(sphere_map.get());
  SaveAsOBJ(tmesh, "snubcube");
  printf("Took %s\n", ANSI::Time(timer.Seconds()).c_str());
}


int main(int argc, char **argv) {
  ANSI::Init();

  // TreeBench2();

  Generate();

  return 0;
}
