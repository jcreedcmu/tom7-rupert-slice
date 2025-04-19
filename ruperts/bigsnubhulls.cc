
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <format>
#include <limits>
#include <mutex>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "ansi.h"
#include "arcfour.h"
#include "atomic-util.h"
#include "base/logging.h"
#include "base/stringprintf.h"
#include "big-polyhedra.h"
#include "bignum/big-overloads.h"
#include "bignum/big.h"
#include "color-util.h"
#include "geom/tree-3d.h"
#include "hashing.h"
#include "hull3d.h"
#include "periodically.h"
#include "polyhedra.h"
#include "randutil.h"
#include "status-bar.h"
#include "threadutil.h"
#include "util.h"
#include "yocto_matht.h"
#include "vector-util.h"

using vec3 = yocto::vec<double, 3>;

DECLARE_COUNTERS(too_close);

struct Hulls {
  size_t Num() { return canonical.size(); }

  size_t GetHullId(const std::vector<int> &hull) {
    CHECK(hull.size() >= 3);

    // Canonicalize:
    //  - The hull should start with the smallest
    //    vertex, numerically. (Handled here)
    //  - We should not include colinear vertices.
    //    (TODO: handled elsewhere?)
    //  - With duplicate points, use the minimum
    //    index (TODO: handled elsewhere?)
    //  - Winding order: The hull could wind in
    //    either direction. Choose the one that
    //    puts the smaller vertex second.

    // It would be great if we modded out by symmetry,
    // too. This might require a more thoughtful
    // assignment of vertex indices?

    int besti = 0;
    for (int i = 1; i < hull.size(); i++) {
      if (hull[i] < hull[besti]) {
        besti = i;
      }
    }

    std::vector<int> hull_key;
    hull_key.reserve(hull.size());
    for (int i = 0; i < hull.size(); i++) {
      hull_key.push_back(hull[(besti + i) % hull.size()]);
    }

    if (hull_key[1] > hull_key.back()) {
      VectorReverse(&hull_key);
      // Because that put the first element last.
      VectorRotateRight(&hull_key, 1);
    }

    // Do we have it?
    auto it = indices.find(hull_key);
    if (it != indices.end()) {
      return it->second;
    } else {
      size_t id = canonical.size();
      indices[hull_key] = id;
      canonical.emplace_back(std::move(hull_key));
      return id;
    }
  }

  std::unordered_map<std::vector<int>, size_t,
                     Hashing<std::vector<int>>> indices;
  std::vector<std::vector<int>> canonical;
};

inline vec3 QuaternionToSpherePoint(const quat4 &q) {
  // The z-column of the rotation matrix represents the rotated Z-axis.
  // return normalize(rotation_frame(q).z);
  return transform_point(rotation_frame(normalize(q)), vec3{0, 0, 1});
}

// Visualize a sample of the distinct convex hulls.
// By distinct convex hull, we mean a specific set of
// points on the hull.
static void SnubHulls() {
  // BigPoly scube = BigCube(32); // BigScube(32);
  Polyhedron cube = Cube();

  Periodically status_per(1.0);
  Hulls hulls;
  constexpr int64_t ITERS = 10'000;
  StatusBar status(1);

  std::mutex m;
  int64_t next_work_idx = 0;
  constexpr int NUM_THREADS = 8;

  // Parallel.
  std::vector<vec3> samples;
  std::vector<size_t> ids;

  Tree3D<double, bool> tree;
  static constexpr double TOO_CLOSE = 1.0e-3;

  ParallelFan(
      NUM_THREADS,
      [&](int thread_idx) {
        ArcFour rc(std::format("snubhulls.{}.{}", thread_idx,
                               time(nullptr)));
        for (;;) {
          int64_t work_idx = 0;
          {
            MutexLock ml(&m);
            if (next_work_idx == ITERS)
              return;
            work_idx = next_work_idx;
            next_work_idx++;
          }

          const quat4 small_quat = normalize(RandomQuaternion(&rc));
          vec3 sphere_point(100.0 * QuaternionToSpherePoint(small_quat));

          if (AllZero(sphere_point))
            continue;

          {
            MutexLock ml(&m);


            if (!tree.Empty()) {
              const auto &[pos_, value_, dist] = tree.Closest(sphere_point.x,
                                                              sphere_point.y,
                                                              sphere_point.z);
              if (dist < TOO_CLOSE) {
                too_close++;
                continue;
              }
            }

            // Reserve it.
            tree.Insert(sphere_point.x,
                        sphere_point.y,
                        sphere_point.z,
                        true);
          }

          #if 0
          BigQuat q = ApproxBigQuat(small_quat,
                                    int64_t{100'000'000'000});


          BigMesh2D shadow =
            RotateAndProject(
                NonUnitRotationFrame(q),
                scube);

          std::vector<int> hull = BigQuickHull(shadow.vertices);
          #endif

          frame3 frame = rotation_frame(small_quat);
          Mesh2D shadow = Shadow(Rotate(cube, frame));
          std::vector<int> hull = QuickHull(shadow.vertices);

          {
            MutexLock ml(&m);
            size_t id = hulls.GetHullId(hull);
            CHECK(ids.size() == samples.size());
            ids.push_back(id);
            samples.push_back(sphere_point);
          }
          status_per.RunIf([&]{
              status.Progressf(work_idx, ITERS,
                               "%lld/%lld  %lld" ARED("≈"),
                               work_idx, ITERS,
                               too_close.Read());
            });
        }
      });

  for (const vec3 &v : samples) {
    CHECK(!AllZero(v));
  }

  CHECK(ids.size() == samples.size());

  printf("There are %lld distinct hulls in this sample.\n",
         (int64_t)hulls.Num());

  ArcFour rc("color");
  std::vector<uint32_t> colors;
  colors.reserve(hulls.Num());
  if (hulls.Num() <= 4) {
    colors.push_back(0xFF0000FF);
    colors.push_back(0x00FF00FF);
    colors.push_back(0x0000FFFF);
    colors.push_back(0xFF00FFFF);
  } else {
    for (int i = 0; i < hulls.Num(); i++) {
      colors.push_back(ColorUtil::HSVAToRGBA32(
                           RandDouble(&rc),
                           0.5 + RandDouble(&rc) * 0.5,
                           0.5 + RandDouble(&rc) * 0.5,
                           1.0));
    }
  }

  if (colors.size() > hulls.Num()) colors.resize(hulls.Num());

  for (int i = 0; i < hulls.Num(); i++) {
    printf("%d. (%s▉" ANSI_RESET "):",
           i, ANSI::ForegroundRGB32(colors[i]).c_str());
    for (int vidx : hulls.canonical[i]) {
      CHECK(vidx >= 0 && vidx < cube.vertices.size());
      printf(" %d", vidx);
    }
    printf("\n");
  }

  // As point cloud.
  if (true) {
    std::string outply =
      std::format(
          "ply\n"
          "format ascii 1.0\n"
          "element vertex {}\n"
          "property float x\n"
          "property float y\n"
          "property float z\n"
          "property uchar red\n"
          "property uchar green\n"
          "property uchar blue\n"
          "end_header\n", samples.size());

    // outply.append("# vertices\n");
    for (size_t i = 0; i < samples.size(); i++) {
      CHECK(i < samples.size());
      const vec3 &v = samples[i];
      const int id = ids[i];
      CHECK(id < colors.size());
      uint32_t color32 = colors[id];
      /*
      uint32_t color32 =
        ColorUtil::HSVAToRGBA32(
            (v.x + 100.0) / 200.0,
            (v.y + 100.0) / 200.0,
            (v.z + 100.0) / 200.0,
            1.0);
      */

      const auto &[r, g, b, _] = ColorUtil::Unpack32(color32);
      AppendFormat(&outply,
                   "{} {} {} {} {} {}\n",
                   v.x, v.y, v.z,
                   r, g, b);
    }

    std::string filename = "snubcloud.ply";
    Util::WriteFile(filename, outply);
    printf("Wrote %lld bytes to %s.\n",
           outply.size(),
           filename.c_str());
  }

  if (false) {
    // Get the convex hull. We expect every point to be on the convex
    // hull (since they are all on the unit sphere) but we want the
    // connectivity of the faces.
    printf("Compute hull from %lld samples:\n", (int64_t)samples.size());
    std::vector<std::tuple<int, int, int>> faces =
      Hull3D::HullFaces(samples);
    printf("%lld triangles in hull.\n", (int64_t)faces.size());

    std::string outply =
      std::format(
          "ply\n"
          "format ascii 1.0\n"
          "element vertex {}\n"
          "property float x\n"
          "property float y\n"
          "property float z\n"
          "property uchar red\n"
          "property uchar green\n"
          "property uchar blue\n"
          "element face {}\n"
          "property list uchar int vertex_indices\n"
          "end_header\n",
          samples.size(),
          faces.size());

    // outply.append("# vertices\n");
    for (size_t i = 0; i < samples.size(); i++) {
      CHECK(i < samples.size());
      const vec3 &v = samples[i];
      const int id = ids[i];
      CHECK(id < colors.size());

      const auto &[r, g, b, _] = ColorUtil::Unpack32(colors[id]);
      AppendFormat(&outply,
                   "{} {} {} {} {} {}\n",
                   v.x, v.y, v.z,
                   r, g, b);
    }

    // outply.append("# faces\n");
    for (const auto &[a, b, c] : faces) {
      CHECK(a >= 0 && a < samples.size());
      CHECK(b >= 0 && b < samples.size());
      CHECK(c >= 0 && c < samples.size());

      // const int id = ids[a];
      // const auto &[rr, gg, bb, _] = ColorUtil::Unpack32(colors[id]);

      AppendFormat(&outply,
                   "3 {} {} {}\n",
                   a, b, c);

      /*
      AppendFormat(&outply,
                   "3 {} {} {} {} {} {}\n",
                   a, b, c,
                   rr, gg, bb);
      */
    }

    std::string filename = "snubhulls.ply";
    Util::WriteFile(filename, outply);
    printf("Wrote %lld bytes to %s.\n",
           outply.size(),
           filename.c_str());
  }
}


int main(int argc, char **argv) {
  ANSI::Init();

  SnubHulls();

  return 0;
}
