
#include "png.h"

#include <bit>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

#include "base/logging.h"
#include "color-util.h"
#include "crypt/lfsr.h"
#include "image.h"
#include "util.h"

static constexpr bool DEBUG_PNG_TEST = false;

#define CHECK_EQUAL(a, b) do {                  \
    auto aa = (a);                              \
    auto bb = (b);                              \
    CHECK(aa == bb) << #a " vs " #b "\n"        \
                    << "which is\n"             \
                    << aa << " vs " << bb;      \
  } while (0)


inline std::span<const uint8_t> StringSpan(std::string_view s) {
  return std::span<const uint8_t>((const uint8_t*)s.data(),
                                  s.size());
}

static void TestCRC() {
  CHECK_EQUAL(PNG::CRC(StringSpan("table task")), (uint32_t)0x632AECF0);
  CHECK_EQUAL(PNG::CRC(StringSpan("IEND")), (uint32_t)0xAE426082);
}

static void CheckRoundTrip(const ImageRGBA &img) {
  std::vector<uint8_t> bytes = PNG::EncodeInMemory(img);

  std::unique_ptr<ImageRGBA> decoded(
      ImageRGBA::LoadFromMemory(bytes));
  if (decoded.get() != nullptr && img == *decoded) {
    // OK.
  } else {
    img.Save("orig.png");
    Util::WriteFileBytes("encoded.png", bytes);
    CHECK(decoded.get() != nullptr) << "Couldn't decode! "
      "Wrote orig.png, encoded.png.";
    decoded->Save("decoded.png");

    int diffs = 0;
    if (decoded->Width() == img.Width() &&
        decoded->Height() == img.Height()) {
      for (int y = 0; y < (int)img.Height(); y++) {
        for (int x = 0; x < (int)img.Width(); x++) {
          uint32_t c = img.GetPixel32(x, y);
          uint32_t e = decoded->GetPixel32(x, y);

          if (c != e && diffs < 10) {
            printf("[%d,%d] Wanted %08x but got %08x\n",
                   x, y, c, e);
            diffs++;
          }
        }
      }
    } else {
      fprintf(stderr, "Images are different sizes!\n");
    }

    LOG(FATAL) <<
      "Round trip failed. Wrote orig.png, encoded.png, "
      "and decoded.png.\n";
  }
}

static void TestEncode256() {
  // 256-color with transparency.
  std::vector<uint32_t> palette;
  for (int i = 0; i < 256; i++) {
    uint32_t c = ColorUtil::HSVAToRGBA32(i / 255.0f, 0.8, 0.8, 1.0);
    c = (c & 0xFFFFFF00) | 128 | ((i & 0b11) << 5);
    palette.push_back(c);
  }

  ImageRGBA img(128, 100);
  for (int y = 0; y < img.Height(); y++) {
    for (int x = 0; x < img.Width(); x++) {
      img.SetPixel32(x, y, palette[(y * 18 + x) & 0xFF]);
    }
  }

  img.FillRect32(8, 8, 12, 15, palette[7]);
  img.FillRect32(9, 48, 12, 15, palette[80]);
  img.FillRect32(50, 20, 16, 16, palette[190]);

  CheckRoundTrip(img);
}

static void TestEncode2() {
  // 2 color
  ImageRGBA img(99, 47);
  for (int y = 0; y < img.Height(); y++) {
    for (int x = 0; x < img.Width(); x++) {
      uint32_t v = y * img.Width() + x;
      if (std::popcount<uint32_t>(v) >= 4) {
        img.SetPixel32(x, y, 0x123456AA);
      } else {
        img.SetPixel32(x, y, 0xFFFF33FF);
      }
    }
  }

  img.FillRect32(8, 8, 12, 15, 0x123456AA);
  img.FillRect32(50, 20, 16, 16, 0xFFFF33FF);

  if (DEBUG_PNG_TEST)
    img.Save("test2.png");

  CheckRoundTrip(img);
}

static void TestEncode4() {
  // 4 color
  ImageRGBA img(35, 11);
  for (int y = 0; y < img.Height(); y++) {
    for (int x = 0; x < img.Width(); x++) {
      uint32_t u = (std::popcount<uint32_t>(x) >= 3) ? 0xAA44CC77 : 0x22CC2A77;
      uint32_t v = (std::popcount<uint32_t>(y) < 3) ? 0x12345688 : 0x65432122;
      img.SetPixel32(x, y, u | v);
    }
  }

  if (DEBUG_PNG_TEST)
    img.Save("test4.png");

  CheckRoundTrip(img);
}

static void TestEncode16() {
  // 16-color
  ImageRGBA img(7, 45);

  std::vector<uint32_t> palette;
  uint32_t cc = 0x00C0FFEE;
  for (int i = 0; i < 16; i++) {
    palette.push_back(cc);
    cc = LFSRNext32(cc);
  }

  for (int y = 0; y < img.Height(); y++) {
    for (int x = 0; x < img.Width(); x++) {
      img.SetPixel32(x, y, palette[(x * 31337 ^ y) & 15]);
    }
  }

  if (DEBUG_PNG_TEST)
    img.Save("test16.png");

  CheckRoundTrip(img);
}


int main(int argc, char **argv) {
  TestCRC();

  TestEncode256();
  TestEncode16();
  TestEncode4();
  TestEncode2();

  printf("OK");
  return 0;
}
