
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

  void AddSpan(std::span<const uint8_t> v) {
    bytes.reserve(bytes.size() + v.size());
    for (uint8_t b : v) {
      bytes.push_back(b);
    }
  }

  void AddBuf(const Buf &other) {
    AddSpan(other.bytes);
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

  size_t DataSize() const {
    CHECK(bytes.size() >= 8);
    return bytes.size() - 8;
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

std::vector<uint8_t> PNG::EncodeInMemory(const ImageRGBA &img,
                                         int level) {
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
    return ZIP::EncodeAsPNG(img.Width(), img.Height(), img.ToBuffer8(), level);
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

  // Index of the color in the palette.
  std::unordered_map<uint32_t, uint8_t> color_index;
  for (int i = 0; i < (int)palette.size(); i++) {
    color_index[palette[i].first] = i;
  }

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

  Chunk plte("PLTE");
  for (const auto &[rgba, count_] : palette) {
    uint8_t r = (rgba >> 24) & 0xFF;
    uint8_t g = (rgba >> 16) & 0xFF;
    uint8_t b = (rgba >> 8) & 0xFF;
    plte.W8(r);
    plte.W8(g);
    plte.W8(b);
  }

  buf.AddChunk(plte);

  Chunk trns("tRNS");
  for (int i = 0; i < (int)palette.size(); i++) {
    const auto &[rgba, count_] = palette[i];
    uint8_t a = rgba & 0xFF;
    if (a == 0xFF) {
      // Once we hit a fully opaque color, we don't
      // need to write any more; these are at the end
      // of the palette and PNG will assume a = 0xFF.
      break;
    }
    trns.W8(a);
  }

  // If all opaque, don't even add the chunk.
  if (trns.DataSize() > 0) {
    buf.AddChunk(trns);
  }

  Buf udata;

  for (int y = 0; y < (int)img.Height(); y++) {
    // TODO(twm): Use other filters here, at least
    // heuristically.
    udata.W8(0x00);

    uint8_t current_byte = 0;
    for (int x = 0; x < (int)img.Width(); x++) {
      uint32_t c = img.GetPixel32(x, y);
      auto it = color_index.find(c);
      CHECK(it != color_index.end()) << "Bug: Missing color";

      uint8_t idx = it->second;
      switch (bpp) {
      case 1: {
        [[assume(idx < 2)]];
        current_byte <<= 1;
        current_byte |= idx;
        if ((x & 0b1111) == 0b1111) {
          udata.W8(current_byte);
          current_byte = 0;
        }
        break;
      }
      case 2: {
        [[assume(idx < 4)]];
        current_byte <<= 2;
        current_byte |= idx;
        if ((x & 0b11) == 0b11) {
          udata.W8(current_byte);
          current_byte = 0;
        }
        break;
      }
      case 4: {
        [[assume(idx < 16)]];
        current_byte <<= 4;
        current_byte |= idx;
        if (x & 1) {
          udata.W8(current_byte);
          current_byte = 0;
        }
        break;
      }
      case 8:
        udata.W8(idx);
        break;
      default:
        LOG(FATAL) << "Bad bpp";
      }
    }

    // Flush incomplete pixels.
    switch (bpp) {
    case 1: {
      const int slack = (0b1111 - (img.Width() & 0b1111));
      if (slack > 0) {
        current_byte <<= slack;
        udata.W8(current_byte);
      }
      break;
    }
    case 2: {
      const int slack = (0b11 - (img.Width() & 0b11)) * 2;
      if (slack > 0) {
        current_byte <<= slack;
        udata.W8(current_byte);
      }
      break;
    }
    case 4: {
      const int slack = (0b1 - (img.Width() & 0b1)) * 4;
      if (slack > 0) {
        current_byte <<= slack;
        udata.W8(current_byte);
      }
      break;
    }
    case 8:
      // Always evenly dividing.
      break;
    default:
      LOG(FATAL) << "Bad bpp";
    }
  }

  // Now compress data.
  Chunk idat("IDAT");
  idat.AddSpan(ZIP::ZlibVector(udata.bytes, level));
  udata.bytes.clear();
  buf.AddChunk(idat);

  Chunk iend("IEND");
  buf.AddChunk(iend);

  LOG(FATAL) << "unimplemented";
  return {};
}
