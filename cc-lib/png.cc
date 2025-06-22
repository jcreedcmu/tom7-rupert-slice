
#include "png.h"

#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <span>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "base/logging.h"
#include "image.h"
#include "miniz.h"
#include "zip.h"

namespace {
struct Chunk;

struct Buf {
  Buf() {}

  void Reserve(size_t n) {
    bytes.reserve(n);
  }

  void WB(const std::initializer_list<uint8_t> &bs) {
    for (uint8_t b : bs) bytes.push_back(b);
  }

  void WFixedString(std::string_view s) {
    for (int i = 0; i < (int)s.size(); i++) {
      bytes.push_back((uint8_t)s[i]);
    }
  }

  void WPascal(std::string_view s) {
    CHECK(s.size() < 256) << "String too large to be stored "
      "as a Pascal string!";
    bytes.push_back((uint8_t)s.size());
    WFixedString(s);
  }

  void W8(uint8_t b) {
    bytes.push_back(b);
  }

  void W16(uint16_t w) {
    W8(0xFF & (w >> 8));
    W8(0xFF & w);
  }

  void W32(uint32_t w) {
    W8(0xFF & (w >> 24));
    W8(0xFF & (w >> 16));
    W8(0xFF & (w >>  8));
    W8(0xFF & (w >>  0));
  }

  void W32At(size_t idx, uint32_t w) {
    CHECK(idx + 3 < bytes.size());
    bytes[idx + 0] = 0xFF & (w >> 24);
    bytes[idx + 1] = 0xFF & (w >> 16);
    bytes[idx + 2] = 0xFF & (w >>  8);
    bytes[idx + 3] = 0xFF & (w >>  0);
  }

  void WCC(const char (&fourcc)[5]) {
    uint32_t enc =
      (uint32_t(fourcc[0]) << 24) |
      (uint32_t(fourcc[1]) << 16) |
      (uint32_t(fourcc[2]) << 8) |
      (uint32_t(fourcc[3]) << 0);
    W32(enc);
  }

  void WriteSizeTo(size_t idx) {
    size_t s = bytes.size();
    CHECK(s < 0x100000000) << "Need to add support for 64-bit sizes!";
    W32At(idx, s);
  }

  void AddBuf(const Buf &other) {
    for (uint8_t b : other.bytes) {
      bytes.push_back(b);
    }
  }

  inline void AddChunk(Chunk &other);

  size_t Size() const {
    return bytes.size();
  }

  std::vector<uint8_t> bytes;
};

// Best to just use Chunk and AddChunk.
struct Chunk : public Buf {
  Chunk(const char (&fourcc)[5]) {
    // Reserved for size.
    W32(0);
    WCC(fourcc);
  }

  void Finalize() {
    CHECK(!finalized) << "Can only finalize once.";
    CHECK(bytes.size() > 8);
    CHECK(bytes.size() < int64_t{0x100000008});
    // Length only measures the data section.
    W32At(0, (uint32_t)(bytes.size() - 8));
    // CRC measures the header and data.
    uint32_t crc = PNG::CRC(std::span<const uint8_t>(
                                bytes.data() + 4,
                                bytes.size() - 4));
    W32(crc);
    finalized = true;
  }

  // Sets the chunk size (in the destination, not source).
  void AddChunk(const Chunk &other) {
    CHECK(other.Size() >= 4);
    size_t pos = bytes.size();
    AddBuf(other);
    W32At(pos, other.Size());
  }

  bool finalized = false;
};

void Buf::AddChunk(Chunk &other) {
  other.Finalize();
  AddBuf(other);
}

}  // namespace

uint32_t PNG::CRC(std::span<const uint8_t> s) {
  // Apparently the PNG CRC is closely related to the ZIP CRC.
  return (uint32_t)mz_crc32(MZ_CRC32_INIT, s.data(), s.size());
}

std::vector<uint8_t> PNG::EncodeInMemory(const ImageRGBA &img) {
  // First thing we do is count the number of distinct pixels.
  // If we have more than 256, then we cannot use indexed color,
  // so we bail to the miniz implementation.

  std::unordered_map<uint32_t, int64_t> color_count;
  for (int y = 0; y < img.Height(); y++) {
    for (int x = 0; x < img.Width(); x++) {
      uint32_t c = img.GetPixel32(x, y);
      color_count[c]++;
    }
  }

  if (color_count.size() > 256) {
    // XXX detect RGB and use RGBEncodeAsPNG.
    return ZIP::EncodeAsPNG(img.Width(), img.Height(), img.ToBuffer8());
  }

  // Palette ordering mostly doesn't matter. We put opaque colors last
  // (because this allows us to have a shorter transparency chunk) and
  // order the two sections by descending frequency in the image.

  std::vector<std::pair<uint32_t, int64_t>> palette;
  for (const auto &[color, count] : color_count) {
    palette.emplace_back(color, count);
  }

  std::sort(palette.begin(),
            palette.end(),
            [](const auto &a, const auto &b) {
              // transparent pixels first.
              if ((a.first & 0xFF) < 0xFF &&
                  (b.first & 0xFF) == 0xFF) {
                return true;
              } else if ((a.first & 0xFF) == 0xFF &&
                         (b.first & 0xFF) < 0xFF) {
                return false;
              }

              // Otherwise, sort by count descending.
              return a.second > b.second;
            });

  // bpp is 1, 2, 4, or 8 only.
  const int bpp = [&palette]() {
      if (palette.size() <= 2) {
        return 1;
      } else if (palette.size() <= 4) {
        return 2;
      } else if (palette.size() <= 16) {
        return 4;
      } else if (palette.size() <= 256) {
        return 8;
      } else {
        LOG(FATAL) << "Bug: Already checked above.";
        return 0;
      }
    }();

  Buf buf;
  // Magic
  buf.WFixedString("\x89PNG\x0d\x0a\x1a\x0a");

  Chunk ihdr("IHDR");
  ihdr.W32(img.Width());
  ihdr.W32(img.Height());

  ihdr.W8(bpp);
  // Indexed color.
  ihdr.W8(3);
  // Compression method; flate (0) is the only legal option.
  ihdr.W8(0);
  // Filter method.
  ihdr.W8(0);
  // Interlace method.
  ihdr.W8(0);

  buf.AddChunk(ihdr);

  // TODO: Add Palette
  // TODO: Add Transparency
  // TODO: Add Pixel data

  Chunk iend("IEND");
  buf.AddChunk(iend);

  LOG(FATAL) << "unimplemented";
  return {};
}
