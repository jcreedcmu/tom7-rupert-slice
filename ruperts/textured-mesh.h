#ifndef _RUPERTS_TEXTURED_MESH_H
#define _RUPERTS_TEXTURED_MESH_H

#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include "yocto_matht.h"
#include "mesh.h"
#include "image.h"
#include "base/logging.h"

// UDIM textures store more than one unit square, which can also be of
// different resolution.
struct UDimTexture {
  UDimTexture() {}

  // The image is always square; power of two is conventional.
  std::pair<int, int> Allocate(int side_pixels) {
    CHECK(side_pixels > 0);
    const int size = (int)textures.size();
    CHECK(size < 9000);
    textures.emplace_back(side_pixels, side_pixels);
    return UnpackUV(size);
  }

  std::pair<int, int> AddTexture(ImageRGBA texture) {
    const int size = (int)textures.size();
    CHECK(size < 9000);
    textures.push_back(std::move(texture));
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

  ImageRGBA &GetTexture(int u, int v) {
    CHECK(u >= 0 && v >= 0 && u < 10);
    int idx = PackUV(u, v);
    CHECK(idx < textures.size());
    return textures[idx];
  }

  const ImageRGBA &GetTexture(int u, int v) const {
    CHECK(u >= 0 && v >= 0 && u < 10);
    int idx = PackUV(u, v);
    CHECK(idx < textures.size());
    return textures[idx];
  }

  int NumTextures() const {
    return textures.size();
  }

 private:
  // Row-major. There are always 10 columns (max U is 10.0), but V can
  // go to about 900.
  std::vector<ImageRGBA> textures;
};

struct TexturedMesh {
  using vec2 = yocto::vec<double, 2>;
  TriangularMesh3D mesh;
  // Parallel with triangles. The uv coordinates of each vertex.
  std::vector<std::tuple<vec2, vec2, vec2>> uvs;
  UDimTexture texture;
};

// Writes base.obj and base.mtl.
void SaveAsOBJ(const TexturedMesh &tmesh, std::string_view filename_base);

#endif
