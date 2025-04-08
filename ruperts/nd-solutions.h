
#ifndef _RUPERTS_ND_SOLUTIONS_H
#define _RUPERTS_ND_SOLUTIONS_H

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

#include "base/logging.h"
#include "image.h"
#include "threadutil.h"
#include "util.h"
#include "bounds.h"

// Maps N doubles to a score.
// Supports multi-threading, but not multiple processes.
template<size_t N>
struct NDSolutions {
  explicit NDSolutions(std::string_view filename);

  void Add(const std::array<double, N> &key, double val);
  int64_t Size() const { return data.size(); }

  // Distance to the closest sample; Euclidean.
  double Distance(const std::array<double, N> &key);
  void Save();

  // Plot the scores given a single dimension index.
  void Plot1D(int dim,
              int image_width, int image_height,
              std::string_view filename);

 private:
  static double ReadDouble(const uint8_t *bytes) {
    static_assert(sizeof(double) == 8);
    std::array<uint8_t, 8> buf;
    memcpy(buf.data(), bytes, 8);
    return std::bit_cast<double>(buf);
  }

  static void AppendDouble(std::vector<uint8_t> *bytes, double d) {
    static_assert(sizeof(double) == 8);
    std::array<uint8_t, 8> buf = std::bit_cast<std::array<uint8_t, 8>>(d);
    for (int i = 0; i < 8; i++) {
      bytes->push_back(buf[i]);
    }
  }

  static double SqDist(const std::array<double, N + 1> &row,
                       const std::array<double, N> &key) {
    double sum = 0.0;
    for (int i = 0; i < N; i++) {
      double d = row[i] - key[i];
      sum += d * d;
    }
    return sum;
  }

  static constexpr char MAGIC[] = "NdS0";

  std::mutex m;
  std::string filename;
  std::vector<std::array<double, N + 1>> data;
};

#endif


// Template implementations follow.

template<size_t N>
void NDSolutions<N>::Save() {
  MutexLock ml(&m);
  std::vector<uint8_t> contents;
  for (int i = 0; i < 4; i++) {
    contents.push_back(MAGIC[i]);
  }
  CHECK(contents.size() == 4) << contents.size();
  for (const auto &a : data) {
    for (const double d : a) {
      AppendDouble(&contents, d);
    }
  }

  CHECK(Util::WriteFileBytes(filename, contents)) << filename;
}

template<size_t N>
void NDSolutions<N>::Add(const std::array<double, N> &key, double val) {
  std::array<double, N + 1> row;
  for (int i = 0; i < N; i++) row[i] = key[i];
  row[N] = val;

  {
    MutexLock ml(&m);
    data.push_back(std::move(row));
  }
}

template<size_t N>
double NDSolutions<N>::Distance(const std::array<double, N> &key) {
  MutexLock ml(&m);
  double min_distance_sq = std::numeric_limits<double>::infinity();
  for (const auto &row : data) {
    double dd = SqDist(row, key);
    min_distance_sq = std::min(min_distance_sq, dd);
  }

  return sqrt(min_distance_sq);
}

template<size_t N>
NDSolutions<N>::NDSolutions(std::string_view filename) : filename(filename) {
  std::vector<uint8_t> bytes = Util::ReadFileBytes(filename);
  if (!bytes.empty()) {
    CHECK(bytes.size() >= 4 &&
          0 == memcmp(bytes.data(), MAGIC, 4) &&
          (bytes.size() - 4) % (8 * (N + 1)) == 0)
        << filename << "Not an nd-solutions file!";

    for (int idx = 4; idx < bytes.size(); idx += (8 * (N + 1))) {
      std::array<double, N + 1> row;
      for (int c = 0; c < N + 1; c++) {
        row[c] = ReadDouble(&bytes[idx + c * 8]);
      }
      data.push_back(std::move(row));
    }
  }
}

template<size_t N>
void NDSolutions<N>::Plot1D(int dim,
                            int image_width, int image_height,
                            std::string_view filename) {
  CHECK(dim >= 0 && dim < N);
  MutexLock ml(&m);

  Bounds bounds;
  // Make sure x axis is included.
  bounds.BoundY(0.0);
  for (const auto &row : data) {
    double x = row[dim];
    double y = row[N];
    if (y >= 0.0) {
      bounds.Bound(x, y);
    }
  }
  bounds.AddTwoMarginsFrac(0.02, 0.0);

  constexpr float PX = 2.0f;
  constexpr float CIRCLE = 3.0f * PX;
  constexpr float DOT = 2.0f * PX;
  ImageRGBA image(image_width, image_height);
  image.Clear32(0x000000FF);

  Bounds::Scaler scaler =
    bounds.Stretch(image.Width(), image.Height()).FlipY();;

  // x axis
  {
    const auto y = scaler.ScaleY(0);
    image.BlendLine32(0, std::round(y), image.Width() - 1, std::round(y),
                      0xFF0000AA);
  }

  for (double x = 0.0; x < bounds.MaxX(); x += 0.25) {
    double xx = scaler.ScaleX(x);
    image.BlendLine32(std::round(xx), 0, std::round(xx), image.Height() - 1,
                      0x00770099);
  }

  for (int x = (int)bounds.MinX(); x < bounds.MaxX(); x++) {
    double xx = scaler.ScaleX(x);
    image.BlendLine32(std::round(xx), 0, std::round(xx), image.Height() - 1,
                      0x33FF33AA);
  }

  for (const auto &row : data) {
    const double x = row[dim];
    const double y = row[N];
    if (y < 0.0) {
      const auto &[sx, sy] = scaler.Scale(x, 0.0);
      image.BlendThickCircleAA32(
          std::round(sx), std::round(sy) + 2 * PX, CIRCLE, PX, 0xFF000099);
    } else {
      const auto &[sx, sy] = scaler.Scale(x, y);
      image.BlendFilledCircleAA32(
          std::round(sx), std::round(sy), DOT, 0xFFFFFF99);
    }
  }

  image.Save(filename);
}
