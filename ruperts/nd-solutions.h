
#ifndef _RUPERTS_ND_SOLUTIONS_H
#define _RUPERTS_ND_SOLUTIONS_H

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>

#include "auto-histo.h"
#include "base/logging.h"
#include "bounds.h"
#include "color-util.h"
#include "geom/tree-nd.h"
#include "hashing.h"
#include "image.h"
#include "integer-voronoi.h"
#include "threadutil.h"
#include "util.h"
#include "yocto_matht.h"

// Maps N doubles to a score and solution.
// Supports multi-threading, but not multiple processes.
template<size_t N>
struct NDSolutions {
  using frame3 = yocto::frame<double, 3>;
  using vec3 = yocto::vec<double, 3>;

  explicit NDSolutions(std::string_view filename, bool verbose = false);

  // Same as NDSolutions(filename).Size(), but much faster
  // because it doesn't need to read the whole file.
  static std::optional<size_t> SolutionsInFile(std::string_view filename,
                                               std::string *error = nullptr);

  bool Empty() { return Size() == 0; }

  void Add(const std::array<double, N> &key, double val,
           const frame3 &outer_frame, const frame3 &inner_frame);
  int64_t Size() {
    MutexLock ml(&m);
    return data.size();
  }

  // Distance to the closest sample; Euclidean.
  double Distance(const std::array<double, N> &key);

  // Aborts if empty.
  std::tuple<std::array<double, N>, double, frame3, frame3>
  Closest(const std::array<double, N> &key);

  void Save();

  // Plot the scores given a single dimension index.
  void Plot1D(int dim,
              int image_width, int image_height,
              std::string_view filename);

  void Plot1DColor2(int xdim, int cdim1, int cdim2,
                    int image_width, int image_height,
                    std::string_view filename);

  void Plot2D(int xdim, int ydim,
              int image_width, int image_height,
              std::string_view filename);

  // XXX should probably not keep this, since it would be
  // much better to use a spatial data structure.
  std::tuple<std::array<double, N>, double, frame3, frame3>
  operator[] (size_t idx);

  // Get the entire vector (as a copy), which may be quite big!
  std::vector<std::tuple<std::array<double, N>, double, frame3, frame3>>
  GetVec();

 private:
  static constexpr size_t ROW_SIZE = N + 1 + 12 + 12;
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

  static double SqDist(const std::array<double, ROW_SIZE> &row,
                       const std::array<double, N> &key) {
    double sum = 0.0;
    for (int i = 0; i < N; i++) {
      double d = row[i] - key[i];
      sum += d * d;
    }
    return sum;
  }

  static std::tuple<std::array<double, N>, double, frame3, frame3>
  DecodeRow(const std::array<double, ROW_SIZE> &row) {
    int idx = 0;
    auto ReadFrame = [&row, &idx](frame3 &f) {
        for (int y = 0; y < 4; y++) {
          vec3 &v = f[y];
          for (int x = 0; x < 3; x++) {
            v[x] = row[idx++];
          }
        }
      };

    std::array<double, N> key;
    double score;
    frame3 outer, inner;
    for (int i = 0; i < N; i++) {
      key[i] = row[idx++];
    }
    score = row[idx++];
    ReadFrame(outer);
    ReadFrame(inner);
    CHECK(idx == ROW_SIZE);
    return std::make_tuple(key, score, outer, inner);
  }

  // Holding lock.
  void CreateIndex();

  static constexpr char MAGIC[] = "NdS1";

  std::mutex m;
  std::string filename;

  // PERF: Probably should just store this in the tree directly
  // to reduce memory pressure.
  std::vector<std::array<double, ROW_SIZE>> data;

  // The index (kd tree), which we generate lazily. Once it
  // exists, it is updated incrementally (some uses do not look
  // up points at all).
  std::unique_ptr<TreeND<double, size_t>> index;
};



// Template implementations follow.

template<size_t N>
auto NDSolutions<N>::GetVec() ->
std::vector<std::tuple<std::array<double, N>, double, frame3, frame3>> {
  std::vector<
    std::tuple<std::array<double, N>, double, frame3, frame3>> ret;
  ret.resize(data.size());
  MutexLock ml(&m);

  for (size_t i = 0; i < data.size(); i++) {
    ret[i] = DecodeRow(data[i]);
  }

  return ret;
}


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
void NDSolutions<N>::CreateIndex() {
  CHECK(index.get() == nullptr);
  index.reset(new TreeND<double, size_t>(N));

  for (size_t idx = 0; idx < data.size(); idx++) {
    const auto &[key, dist, outer, inner] = DecodeRow(data[idx]);
    index->Insert(key, idx);
  }
}

template<size_t N>
void NDSolutions<N>::Add(
    const std::array<double, N> &key, double val,
    const frame3 &outer_frame, const frame3 &inner_frame) {
  std::array<double, ROW_SIZE> row;
  for (int i = 0; i < N; i++)
    row[i] = key[i];
  row[N] = val;

  auto AddFrame = [&](int start_idx, const frame3 &f) {
      for (int y = 0; y < 4; y++) {
        const vec3 &v = f[y];
        for (int x = 0; x < 3; x++) {
          row[start_idx++] = v[x];
        }
      }
    };

  AddFrame(N + 1, outer_frame);
  AddFrame(N + 1 + 12, inner_frame);

  {
    MutexLock ml(&m);
    size_t idx = data.size();
    data.push_back(std::move(row));
    if (index.get() != nullptr) {
      index->Insert(key, idx);
    }
  }
}

template<size_t N>
double NDSolutions<N>::Distance(const std::array<double, N> &key) {
  MutexLock ml(&m);
  if (index.get() == nullptr)
    CreateIndex();

  if (index->Empty())
    return std::numeric_limits<double>::infinity();

  const auto &[pos, idx, dist] = index->Closest(key);
  return dist;
}

template<size_t N>
auto NDSolutions<N>::Closest(const std::array<double, N> &key) ->
  std::tuple<std::array<double, N>, double, frame3, frame3> {
  MutexLock ml(&m);

  if (index.get() == nullptr)
    CreateIndex();

  CHECK(!index->Empty());

  const auto &[pos, idx, dist] = index->Closest(key);
  return DecodeRow(data[idx]);
}

template<size_t N>
NDSolutions<N>::NDSolutions(std::string_view filename, bool verbose) : filename(filename) {
  // Note that these will easily exceed 32-bit byte indices.
  std::vector<uint8_t> bytes = Util::ReadFileBytes(filename);
  if (verbose) { printf("Read %lld bytes.\n", bytes.size()); }
  if (!bytes.empty()) {
    CHECK(bytes.size() >= 4 &&
          0 == memcmp(bytes.data(), MAGIC, 4) &&
          (bytes.size() - 4) % (8 * ROW_SIZE) == 0)
        << filename << "Not an nd-solutions file!";

    for (int64_t idx = 4; idx < bytes.size(); idx += (8 * ROW_SIZE)) {
      std::array<double, ROW_SIZE> row;
      for (int c = 0; c < ROW_SIZE; c++) {
        row[c] = ReadDouble(&bytes[idx + c * 8]);
      }
      data.push_back(std::move(row));
      // if (verbose && (idx % 1000 == 0)) printf("idx %lld\n", idx);
    }
  }
  if (verbose) printf("Read %lld bytes OK\n", bytes.size());

  // Index is created on demand.
}

template<size_t N>
std::optional<size_t>
NDSolutions<N>::SolutionsInFile(std::string_view filename,
                                std::string *error) {
  namespace fs = std::filesystem;
  std::error_code ec;
  ec.clear();

  std::uintmax_t size = fs::file_size(filename, ec);
  if (ec) {
    if (error != nullptr) *error = "Couldn't get file size";
    return std::nullopt;
  }

  // Check header.
  if (!Util::HasMagic(filename, std::string_view(MAGIC, 4))) {
    if (error != nullptr) *error = "Bad header";
    return std::nullopt;
  }

  if ((size - 4) % (8 * ROW_SIZE) != 0) {
    if (error != nullptr) *error = "Incorrect size";
    return std::nullopt;
  }

  size -= 4;
  size /= (8 * ROW_SIZE);

  return {size};
}

template<size_t N>
void NDSolutions<N>::Plot1D(int dim,
                            int image_width, int image_height,
                            std::string_view filename) {
  CHECK(dim >= 0 && dim < N);
  MutexLock ml(&m);

  AutoHisto histo(10000);

  Bounds bounds;
  // Make sure x axis is included.
  bounds.BoundY(0.0);
  for (const auto &row : data) {
    double x = row[dim];
    double y = row[N];
    if (y >= 0.0) {
      bounds.Bound(x, y);
    }
    histo.Observe(y);
    /*
    if (std::abs(y) > 1.0e-12) {
      printf("%.11g,%.11g\n", x, y);
    }
    */
  }
  bounds.AddTwoMarginsFrac(0.02, 0.0);

  printf("Axis %d:\n%s\n",
         dim, histo.SimpleANSI(40).c_str());

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
          std::round(sx), std::round(sy) + 2 * PX, CIRCLE, PX, 0xFF000033);
    } else {
      const auto &[sx, sy] = scaler.Scale(x, y);
      image.BlendFilledCircleAA32(
          std::round(sx), std::round(sy), DOT, 0xFFFFFF99);
    }
  }

  image.Save(filename);
}


template<size_t N>
void NDSolutions<N>::Plot1DColor2(int xdim, int cdim1, int cdim2,
                                  int image_width, int image_height,
                                  std::string_view filename) {
  CHECK(xdim >= 0 && xdim < N);
  CHECK(cdim1 >= 0 && cdim1 < N);
  CHECK(cdim2 >= 0 && cdim2 < N);
  MutexLock ml(&m);

  AutoHisto histo(10000);

  Bounds bounds;
  Bounds cbounds;
  // Make sure x axis is included.
  bounds.BoundY(0.0);
  // XXX specific to football
  bounds.BoundX(1.0);
  for (const auto &row : data) {
    double x = row[xdim];
    double y = row[N];
    if (y >= 0.0) {
      bounds.Bound(x, y);
    }
    histo.Observe(y);

    double u = row[cdim1];
    double v = row[cdim2];
    cbounds.Bound(u, v);
    /*
    if (std::abs(y) > 1.0e-12) {
      printf("%.11g,%.11g\n", x, y);
    }
    */
  }
  bounds.AddTwoMarginsFrac(0.02, 0.0);

  printf("Axis %d:\n%s\n",
         xdim, histo.SimpleANSI(40).c_str());

  constexpr float PX = 2.0f;
  constexpr float CIRCLE = 3.0f * PX;
  constexpr float DOT = 2.0f * PX;
  ImageRGBA image(image_width, image_height);
  image.Clear32(0x000000FF);

  Bounds::Scaler scaler =
    bounds.Stretch(image.Width(), image.Height()).FlipY();

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

  Bounds::Scaler cscaler = cbounds.Stretch(1.0, 1.0);

  for (const auto &row : data) {
    const double x = row[xdim];
    const double y = row[N];
    if (y < 0.0) {

      const auto &[sx, sy] = scaler.Scale(x, 0.0);
      image.BlendThickCircleAA32(
          std::round(sx), std::round(sy) + 2 * PX, CIRCLE, PX,
          0xFF000033);
    } else {
      const auto &[cx, cy] = cscaler.Scale(row[cdim1], row[cdim2]);
      uint32_t color = ColorUtil::FloatsTo32(cx, 0.5, cy, 0.05);

      const auto &[sx, sy] = scaler.Scale(x, y);
      image.BlendFilledCircleAA32(
          std::round(sx), std::round(sy), DOT, color);
    }
  }

  image.Save(filename);
}


template<size_t N>
void NDSolutions<N>::Plot2D(int xdim, int ydim,
                            int image_width, int image_height,
                            std::string_view filename) {
  CHECK(xdim >= 0 && xdim < N);
  CHECK(ydim >= 0 && ydim < N);
  CHECK(xdim != ydim);
  MutexLock ml(&m);

  static constexpr ColorUtil::Gradient GREEN_TO_BLUE{
    GradRGB(0.00f, 0x00FF00),
    GradRGB(0.25f, 0x00FFFF),
    GradRGB(0.50f, 0x0000FF),
    GradRGB(0.75f, 0xFF00FF),
    GradRGB(1.00f, 0xFFFFFF)
  };


  AutoHisto histo(100'000'000);

  // Just using Y axis.
  Bounds clearance_bounds;

  Bounds bounds;
  // Make sure x axis is included.
  bounds.BoundY(0.0);
  int64_t unsolved = 0;
  for (size_t idx = 0; idx < data.size(); idx++) {
    const auto &row = data[idx];
    double x = row[xdim];
    double y = row[ydim];
    bounds.Bound(x, y);

    double clearance = row[N];
    if (clearance > 0.0) {
      clearance_bounds.BoundY(clearance);
    } else {
      unsolved++;
    }

    histo.Observe(x);
  }
  bounds.AddTwoMarginsFrac(0.02, 0.0);

  printf("Axis %d:\n%s\n",
         xdim, histo.SimpleANSI(40).c_str());

  double min_clearance = clearance_bounds.MinY();
  double max_clearance = clearance_bounds.MaxY();
  double clearance_span = max_clearance - min_clearance;

  printf("Rows: %lld\n", (int64_t)data.size());
  printf("Unsolved: %lld\n", unsolved);
  printf("xdim %d, ydim %d\n"
         "x range: [%.17g,%.17g]\n"
         "y range: [%.17g,%.17g]\n",
         xdim, ydim,
         bounds.MinX(), bounds.MaxX(),
         bounds.MinY(), bounds.MaxY());

  printf("Min clearance: %.17g\n"
         "Max clearance: %.17g\n"
         "Spanning: %.17g\n",
         min_clearance,
         max_clearance,
         clearance_span);

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

  // major x ticks
  for (double x = 0.0; x < bounds.MaxX(); x += 0.25) {
    double xx = scaler.ScaleX(x);
    image.BlendLine32(std::round(xx), 0, std::round(xx), image.Height() - 1,
                      0x00770099);
  }

  // minor x ticks
  for (int x = (int)bounds.MinX(); x < bounds.MaxX(); x++) {
    double xx = scaler.ScaleX(x);
    image.BlendLine32(std::round(xx), 0, std::round(xx), image.Height() - 1,
                      0x33FF33AA);
  }

  // Parallel vectors, for voronoi.
  std::vector<uint32_t> colors;
  std::vector<std::pair<int, int>> points;
  std::unordered_set<std::pair<int, int>, Hashing<std::pair<int, int>>>
    has_pixel;
  for (const auto &row : data) {
    const double x = row[xdim];
    const double y = row[ydim];
    const auto &[sx, sy] = scaler.Scale(x, y);
    double clearance = row[N];
    uint32_t color = 0xFFFFFFFF;
    // Use voronoi when sparse?
    if (clearance <= 0.0) {
      color = 0xFF0000FF;
    } else {
      double cf = (clearance - min_clearance) / clearance_span;
      color = ColorUtil::LinearGradient32(GREEN_TO_BLUE, cf);
    }

    int isx = (int)std::round(sx);
    int isy = (int)std::round(sy);
    if (!has_pixel.contains(std::make_pair(sx, sy))) {
      colors.push_back(color);
      points.emplace_back(isx, isy);
      has_pixel.insert(std::make_pair(sx, sy));
    }
  }

  ImageRGBA voronoi = IntegerVoronoi::Rasterize32(points,
                                                  image.Width(),
                                                  image.Height());

  for (int y = 0; y < image.Height(); y++) {
    for (int x = 0; x < image.Width(); x++) {
      int idx = voronoi.GetPixel32(x, y);
      CHECK(idx >= 0 && idx < colors.size());
      image.SetPixel32(x, y, colors[idx]);
    }
  }

  image.Save(filename);
}

template<size_t N>
auto NDSolutions<N>::operator[] (size_t row_idx) ->
  std::tuple<std::array<double, N>, double, frame3, frame3> {
  MutexLock ml(&m);
  CHECK(row_idx >= 0 && data.size()) << row_idx << " vs " << data.size();
  const auto &row = data[row_idx];
  return DecodeRow(row);
}

#endif
