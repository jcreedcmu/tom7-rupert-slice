
#include <cmath>
#include <cstdint>
#include <stdio.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "ansi.h"
#include "base/logging.h"
#include "hexdump.h"
#include "stb_truetype.h"
#include "util.h"
#include "base/stringprintf.h"

static std::array<const char *, 8> shades = {
  " ",
  "⋅",
  "·",
  "⏹",
  "░",
  "▒",
  "▓",
  "█",
};

static void ShowBitmap(const stbtt_fontinfo *font, int size,
                       int ch) {
  int w = 0, h = 0;
  unsigned char *bitmap = stbtt_GetCodepointBitmap(
      font, 0,
      stbtt_ScaleForPixelHeight(font, size), ch, &w, &h, 0, 0);

  for (int j = 0; j < h; ++j) {
    for (int i = 0; i < w; ++i) {
      printf("%s", shades[7 & (bitmap[j*w+i]>>5)]);
    }
    putchar('\n');
  }
  stbtt_FreeBitmap(bitmap, nullptr);
}

static void ShowBitmapForGlyph(const stbtt_fontinfo *font, int size,
                               int glyph) {
  int w = 0, h = 0;
  unsigned char *bitmap = stbtt_GetGlyphBitmap(
      font, 0,
      stbtt_ScaleForPixelHeight(font, size), glyph, &w, &h, 0, 0);

  for (int j = 0; j < h; ++j) {
    for (int i = 0; i < w; ++i) {
      printf("%s", shades[7 & (bitmap[j*w+i]>>5)]);
    }
    putchar('\n');
  }
  stbtt_FreeBitmap(bitmap, nullptr);
}

static void TestA() {
  std::vector<uint8_t> contents =
    Util::ReadFileBytes("c:/windows/fonts/arialbd.ttf");
  stbtt_fontinfo font;
  stbtt_InitFont(&font, contents.data(), contents.size(), 0);
  ShowBitmap(&font, 20, 'a');
}


// Output:
//
//     .ii.
//    @@@@@@.
//   V@Mio@@o
//   :i.  V@V
//     :oM@@M
//   :@@@MM@M
//   @@o  o@M
//  :@@.  M@M
//   @@@o@@@@
//   :M@@V:@@.


static void ASCIIDemo() {
  // from stbtt
  static unsigned char screen[20][79];

  stbtt_fontinfo font;
  int i,j,ascent,baseline,ch=0;
  // leave a little padding in case the character extends left
  float scale, xpos=2;
  // intentionally misspelled to show 'lj' brokenness
  const char *text = "Heljo World!";

  std::vector<uint8_t> contents =
    Util::ReadFileBytes("c:/windows/fonts/arialbd.ttf");

  stbtt_InitFont(&font, contents.data(), contents.size(), 0);

  scale = stbtt_ScaleForPixelHeight(&font, 15);
  stbtt_GetFontVMetrics(&font, &ascent,0,0);
  baseline = (int) (ascent*scale);

  while (text[ch]) {
    int advance, lsb, x0, y0, x1, y1;
    float x_shift = xpos - (float)floor(xpos);
    stbtt_GetCodepointHMetrics(&font, text[ch], &advance, &lsb);
    stbtt_GetCodepointBitmapBoxSubpixel(&font, text[ch], scale, scale, x_shift,
                                        0, &x0, &y0, &x1, &y1);
    stbtt_MakeCodepointBitmapSubpixel(
        &font, &screen[baseline + y0][(int)xpos + x0], x1 - x0, y1 - y0, 79,
        scale, scale, x_shift, 0, text[ch]);
    // note that this stomps the old data, so where character boxes overlap
    // (e.g. 'lj') it's wrong because this API is really for baking character
    // bitmaps into textures. if you want to render a sequence of characters,
    // you really need to render each bitmap to a temp buffer, then "alpha
    // blend" that into the working buffer
    xpos += (advance * scale);
    if (text[ch + 1])
      xpos += scale * stbtt_GetCodepointKernAdvance(
          &font, text[ch], text[ch + 1]);
    ++ch;
  }

  for (j = 0; j < 20; ++j) {
    for (i = 0; i < 78; ++i)
      putchar(" .:ioVM@"[screen[j][i] >> 5]);
    putchar('\n');
  }
}

static void TestCMap() {
  std::vector<uint8_t> contents =
    Util::ReadFileBytes("fonts/DFXPasement9px.ttf");

  stbtt_fontinfo font;
  stbtt_InitFont(&font, contents.data(), contents.size(), 0);

  std::unordered_map<uint16_t, std::vector<uint32_t>> codepoint_from_glyph =
    stbtt_GetGlyphs(&font);

  CHECK(!codepoint_from_glyph.empty());
  printf("There are %d glyphs with codepoints.\n",
         (int)codepoint_from_glyph.size());

  std::unordered_set<uint32_t> codepoints_found;

  for (const auto &[glyph, cps] : codepoint_from_glyph) {
    for (uint32_t codepoint : cps) {
      CHECK(codepoint > 0);
      codepoints_found.insert(codepoint);
      CHECK(stbtt_FindGlyphIndex(&font, codepoint) == glyph);
    }
  }

  CHECK(codepoints_found.contains('*'));
  // Some non-ascii characters I know are in the font.
  CHECK(codepoints_found.contains(0x2665));
  CHECK(codepoints_found.contains(0x221E));

  printf("glyph/codepoint mapping OK!\n");
}

[[maybe_unused]]
static void DebugFont() {
  std::vector<uint8_t> contents = Util::ReadFileBytes("aj.ttf");

  stbtt_fontinfo font;
  stbtt_InitFont(&font, contents.data(), contents.size(), 0);

  std::unordered_map<uint16_t, std::vector<uint32_t>> codepoint_from_glyph =
    stbtt_GetGlyphs(&font);

  CHECK(!codepoint_from_glyph.empty()) << "No glyphs. Musta failed";
  printf("There are %d glyphs with codepoints.\n",
         (int)codepoint_from_glyph.size());

  std::unordered_set<uint32_t> codepoints_found;

  int correct = 0;
  for (const auto &[glyph, codepoints] : codepoint_from_glyph) {
    for (const uint32_t codepoint : codepoints) {
      CHECK(codepoint > 0);
      codepoints_found.insert(codepoint);
      int stb_glyph = stbtt_FindGlyphIndex(&font, codepoint);
      if (stb_glyph != glyph) {
        printf("\nstb glyph:\n");
        ShowBitmapForGlyph(&font, 40, stb_glyph);
        printf("\n\nmy glyph:\n");
        ShowBitmapForGlyph(&font, 40, glyph);
        printf("\n");
        printf("For U+%04x stb got %04x, but I got %04x. Correct so far: %d\n",
               codepoint, stb_glyph, glyph, correct);
        LOG(FATAL) << "Failed";
      }
      correct++;
}
  }

  CHECK(codepoints_found.contains('*'));

  printf("glyph/codepoint mapping OK!\n");
}

int main(int argc, char **argv) {
  ANSI::Init();

  TestCMap();
  // DebugFont();

  printf("Need to have arialbd.ttf in windows directory, and need to "
         "inspect the output:\n");
  TestA();
  ASCIIDemo();

  return 0;
}
