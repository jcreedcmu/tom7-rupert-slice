
//////////////////////////////////////////////////////////////////////
// I stopped using SDL_Image in favor of stb_image, which doesn't need
// to link nightmare factories like libpng, zlib, libjpeg, etc. etc.
// You can still probably use LoadImageFile to do what you want.
//////////////////////////////////////////////////////////////////////

#ifndef _CC_LIB_SDL_SDLUTIL_H
#define _CC_LIB_SDL_SDLUTIL_H

#include "SDL_video.h"

#include <string>
#include <cstdint>
#include <string_view>

struct ImageRGBA;

struct sdlutil {
  static SDL_Surface *makescreen(int w, int h);
  static void slock(SDL_Surface *surf);
  static void sulock(SDL_Surface *surf);

  /* eat and ignore all the events in the queue for a few ticks */
  static void eatevents(int ticks, uint32_t mask = 0xFFFFFFFF);

  /* for direct pixel access: lock first, unlock after */
  static void drawpixel(SDL_Surface *surf, int x, int y,
                        uint8_t r, uint8_t g, uint8_t b);
  static void drawclippixel(SDL_Surface *surf, int x, int y,
                            uint8_t r, uint8_t g, uint8_t b);
  static uint32_t getpixel(SDL_Surface *, int x, int y);
  static void   setpixel(SDL_Surface *, int x, int y,
                         uint32_t color);

  // Assumes 32-bit surface, no bounds checking, inlined.
  inline static void SetPixel32(SDL_Surface *, int x, int y, uint32_t color);
  // With bounds checking.
  inline static void ClipPixel32(SDL_Surface *, int x, int y, uint32_t color);

  // Load supported files using stb_image.
  static SDL_Surface *LoadImageFile(const std::string &filename);
  // Save using stb_image_write. Might only work for 32-bit RGBA.
  // Note that the output image is only lightly compressed; pngcrush
  // or a similar tool can usually improve it quite a bit.
  static bool SavePNG(const std::string &filename, SDL_Surface *surf);

  // Set the program's icon from an image file, which should be 32x32.
  // This should be called before makescreen.
  static void SetIcon(std::string_view filename);

  // Convert ImageRGBA into a new SDL surface.
  static SDL_Surface *FromRGBA(const ImageRGBA &rgba);

  // Copies the RGBA image to the surface. The source rectangle must
  // be in the bounds of the image, but it is clipped to the destination.
  // Assumes the destination has a 32-bit pixel format.
  static void CopyRGBARect(const ImageRGBA &img,
                           int srcx, int srcy, int srcw, int srch,
                           int dstx, int dsty,
                           SDL_Surface *dst);

  // px is the number of destination pixels (in each dimension) per
  // input pixel. So px=2 yields 2x2 pixels.
  // Assumes the destination has a 32-bit pixel format.
  static void CopyRGBARectNX(const ImageRGBA &img, int px,
                             int srcx, int srcy, int srcw, int srch,
                             int dstx, int dsty,
                             SDL_Surface *dst);

  // Copy the RGBA image to the screen as quickly as I know how.
  static void CopyRGBAToScreen(const ImageRGBA &img,
                               SDL_Surface *screen);

  // Clone the surface. The copy will need to be freed with SDL_FreeSurface.
  static SDL_Surface *duplicate(SDL_Surface *surf);

  // Unchecked--don't draw outside the surface!
  static void drawline(SDL_Surface *, int x1, int y1, int x2, int y2,
                       uint8_t R, uint8_t G, uint8_t B);

  static void drawclipline(SDL_Surface *, int x1, int y1, int x2, int y2,
                           uint8_t R, uint8_t G, uint8_t B);
  // Sets pixels; doesn't blend alpha.
  static void DrawClipLine32(SDL_Surface *, int x1, int y1, int x2, int y2,
                             uint32_t color);

  // Efficiently clip the line segment (x0,y0)-(x1,y1) to the rectangle
  // (cx0,cy0)-(cx1,cy1). Modifies the endpoints to place them on the
  // rectangle if they exceed its bounds. Returns false if the segment
  // is completely outside the rectangle, in which case the endpoints
  // can take on any values.
  static bool clipsegment(float cx0, float cy0, float cx1, float cy1,
                          float &x0, float &y0, float &x1, float &y1);

  // With clipping.
  static void drawbox(SDL_Surface *, int x1, int y1, int w, int h,
                      uint8_t r, uint8_t g, uint8_t b);
  static void DrawBox32(SDL_Surface *s, int x, int y, int w, int h,
                        uint32_t rgba);

  // Note: uint32 must be in the correct byte order for the surface.
  static void clearsurface(SDL_Surface *, uint32_t);
  // Correctly maps color to the surface's byte order.
  static void ClearSurface(SDL_Surface *, uint8_t r, uint8_t g, uint8_t b,
                           uint8_t a = 255);

  /* make a n pixel border around a surface. */
  static void outline(SDL_Surface *, int n, int r, int g, int b, int a);

  /* Blit the entire source surface to the destination at the given
     position. */
  static void blitall(SDL_Surface *src, SDL_Surface *dst, int x, int y);

  /* Just like SDL_Fillrect, but no need to create SDL_Rects */
  static void fillrect(SDL_Surface *, uint32_t color,
                       int x, int y, int w, int h);

  static void FillRectRGB(SDL_Surface *,
                          int x, int y, int w, int h,
                          uint8_t r, uint8_t g, uint8_t b);

  // Draw a circle (outline only) centered at the point with the given
  // integer radius. Sets pixels (no alpha blending). Clipped. Only
  // supports 32-bit surfaces.
  static void DrawCircle32(SDL_Surface *,
                           int x0, int y0, int radius, uint32_t color);

  // Draw a filled triangle of the given color (no alpha blending). Clipped.
  // Only supports 32-bit surfaces.
  static void FillTriangle32(SDL_Surface *,
                             int x1, int y1, int x2, int y2, int x3, int y3,
                             uint32_t color);

  /* create a rectangle of the specified size, filled with a color
     at a certain alpha value. This can then be blit to the screen. */
  static SDL_Surface *makealpharect(int w, int h, int r,
                                    int g, int b, int a);

  /* make an alpha rectangle with a vertical gradient between the two
     specified colors.  bias should be between 0 and 1; a higher bias
     means the gradient will have more of the bottom color.
  */
  static SDL_Surface *makealpharectgrad(int w, int h,
                                        int r1, int b1, int g1, int a1,
                                        int r2, int b2, int g2, int a2,
                                        float bias = 0.5f);

  static SDL_Surface *makesurface(int w, int h, bool alpha = true);

  /* make a new surface with the same contents as the old, or 'color' where
     undefined */
  // XXX version that doesn't need pre-mapped color
  static SDL_Surface *resize_canvas(SDL_Surface *s, int w, int h,
                                    uint32_t color);

  // In an unspecified format (for debugging). Includes surface
  // flags and pixel format information.
  static std::string SurfaceInfo(SDL_Surface *surf);

  /* shrink a 32bpp surface to 50% of its original size,
     returning the new surface. */
  static SDL_Surface *shrink50(SDL_Surface *src);

  /* Grow to 2x its original size, returning a new surface. */
  static SDL_Surface *grow2x(SDL_Surface *src);
  /* For 3x, 4x, etc. */
  static SDL_Surface *GrowX(SDL_Surface *src, int px);

  /* create a mipmap array (of nmips successively half-sized images)
     in the array surfs. The first entry of surfs should be filled
     with a surface, so for nmips==1 this does nothing. */
  static bool make_mipmaps(SDL_Surface **surfs, int nmips);

  /* divide alpha channel by 2 */
  static SDL_Surface *alphadim(SDL_Surface *src);

  /* flip a surface horizontally */
  static SDL_Surface *fliphoriz(SDL_Surface *src);

  /* mix two 32bit colors, doing the right thing for alpha.
     Deprecated -- it's almost always wrong because we don't know
     which channel is alpha. Use Mix2.
   */
  static uint32_t mix2(uint32_t, uint32_t);
  static uint32_t Mix2(const SDL_Surface *surface_for, uint32_t j, uint32_t k);

  /* mix four 32bit colors, byte-order agnostic */
  /* Deprecated -- it's almost always wrong because we don't know
     which channel is alpha. Use Mix4. */
  static uint32_t mix4(uint32_t, uint32_t, uint32_t, uint32_t);
  static uint32_t Mix4(const SDL_Surface *surface_for,
                     uint32_t j, uint32_t k, uint32_t l, uint32_t m);

  /* frac should be between 0 and 1, and we get frac of the first
     color and 1-frac of the second. Note that this does not
     treat alpha specially. */
  static uint32_t mixfrac(uint32_t, uint32_t, float frac);

  /* convert from hue-saturation-value-alpha space to RGBA (compatible
     with the supplied surface) */
  static uint32_t hsv(SDL_Surface *surf, float h, float s, float v, float a);

  // This is like format.format in SDL 2.0.
  // Only works for 32 BPP pixel formats.
  // Note that this is independent of the host byte order (little- or
  // big-endian), so if you have uint8 pixel data, you also need to
  // tend to that.
  enum class ByteOrder {
    // This is for formats with alpha.
    ARGB, RGBA, ABGR, BGRA,
    // Some screen formats use 32 BPP but do not represent the alpha
    // channel. In this case the unused byte is "O".
    ORGB,
  };
  static ByteOrder GetByteOrder(SDL_Surface *surf);
};

// Inline functions follow.

inline void sdlutil::SetPixel32(SDL_Surface *surf, int x, int y,
                                uint32_t color) {
  uint32_t *bufp = (uint32_t *)surf->pixels;
  bufp[y * (surf->pitch >> 2) + x] = color;
}

inline void sdlutil::ClipPixel32(SDL_Surface *surf, int x, int y,
                                 uint32_t color) {
  if (x < 0 || y < 0) return;
  if (x >= surf->w) return;
  if (y >= surf->h) return;
  SetPixel32(surf, x, y, color);
}


#endif
