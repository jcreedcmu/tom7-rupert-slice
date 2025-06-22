
#ifndef _CC_LIB_PNG_H
#define _CC_LIB_PNG_H

#include <span>
#include <vector>

#include "image.h"

// My own PNG encoder. Note there is also ImageRGBA::Save (via
// stb_image_write) and ZIP::EncodeAsPNG (via miniz). The chief
// advantage of this one is that it supports palettized images, which
// can be much smaller when the input contains only a few distinct
// colors.
struct PNG {

  // Also capable of indexed color encoding. Chooses the minimum palette
  // size. This is exact, even including fully transparent pixels with
  // different colors. You can get smaller files by quantizing the colors
  // ahead of time; the biggest savings happens when you have <= 2^1, 2^2,
  // 2^4, or 2^8 distinct RGBA values in the image.
  static std::vector<uint8_t> EncodeInMemory(const ImageRGBA &img);


  // Details.

  // The PNG CRC-32 function.
  static uint32_t CRC(std::span<const uint8_t> s);

};

#endif
