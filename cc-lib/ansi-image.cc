
#include "ansi-image.h"

#include <string>
#include <cstdint>

#include "base/stringprintf.h"
#include "ansi.h"

std::string ANSIImage::HalfChar(const ImageRGBA &img_in,
                                uint32_t background_color) {

  // Normalize it so that we can ignore alpha and assume an even height.
  ImageRGBA img(img_in.Width(), img_in.Height() + (img_in.Height() & 1));
  background_color |= 0x000000FF;
  img.Clear32(background_color);
  img.BlendImage(0, 0, img_in);


  if (img.Width() < 1 || img.Height() < 2) return "";

  std::string out;

  // We just keep outputting U+2580, the block drawing character with
  // the upper half of the character set. So the foreground color is
  // the pixel from the first row, and the background is the pixel from
  // the second.
  for (int y = 0; y < img.Height(); y += 2) {
    // To save a bit of space in the output, we only change
    // colors when we need to. Set to something that won't match.
    uint32_t last_fg = ~img.GetPixel32(0, y);
    uint32_t last_bg = ~img.GetPixel32(0, y + 1);

    for (int x = 0; x < img.Width(); x++) {
      uint32_t fg = img.GetPixel32(x, y);
      uint32_t bg = img.GetPixel32(x, y + 1);

      if (bg != last_bg) {
        out += ANSI::BackgroundRGB32(bg);
      }
      if (fg != last_fg) {
        out += ANSI::ForegroundRGB32(fg);
      }

      // U+2580
      out += "▀";
    }

    out += ANSI_RESET "\n";
  }

  return out;
}
