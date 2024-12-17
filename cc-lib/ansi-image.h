
// Routines for working with both ImageRGBA and ANSI codes.

#ifndef _CC_LIB_ANSI_IMAGE_H
#define _CC_LIB_ANSI_IMAGE_H

#include <string>
#include <cstdint>

#include "image.h"

struct ANSIImage {

  // Generate an ANSI version of the image. This uses RGB color codes
  // and two pixels per output character, so the output is exactly
  // img.Width() * ceil(image.Height() / 2) characters when rendered.
  //
  // Composites the image onto the background color. This only matters
  // if it has alpha or its height is not even.
  static std::string HalfChar(const ImageRGBA &img,
                              uint32_t background_color = 0x000000FF);

  // TODO: There are box drawing characters for all 2^4 ways of
  // subdividing into 4 pixels. We can only get two colors per character,
  // though, and the pixel aspect ratio would be wrong. But for cases
  // like drawing a line, this would likely be preferable.

  // TODO:
  //
  // Same, but abuse alpha channel for ascii characters.
  //
  // Parse and render ANSI to an ImageRGBA.
  //  - This needs a font, but we will have PNG decoding through
  //    ImageRGBA so it would be easy to embed fixedersys (tens of
  //    kilobytes, but likely smaller if we make the images one bit)?

};

#endif
