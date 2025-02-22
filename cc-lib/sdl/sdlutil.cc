
#include "sdlutil.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <stdlib.h>
#include <string>
#include <string_view>
#include <vector>
#include <utility>
#include <cstdint>

// XXX probably just prefer the ImageRGBA interface?
#include "SDL_endian.h"
#include "SDL_events.h"
#include "SDL_timer.h"
#include "stb_image.h"
#include "stb_image_write.h"

#include "image.h"

#include "base/logging.h"
#include "base/stringprintf.h"

#include "SDL_stdinc.h"
#include "SDL_video.h"

using namespace std;

/* by default, use display format. but
   when in console mode (for instance)
   there is no display!! */
#ifndef USE_DISPLAY_FORMAT
# define USE_DISPLAY_FORMAT 1
#endif

/*
   XXX uses of these are probably always wrong. In addition
   to the host byte order (though you're probably compiling
   for x86 or x86-64, so this is probably not an issue), different
   OSes use different channel order. Stamp this out.
   SDL_GetRGBA and SDL_MapRGBA are the correct way to do this,
   though they may not be as fast.

   (Note that these are used in the call to SDL_CreateRGBSurface;
   why?)

   PERF: Instead inline SDL_GetRGBA and SDL_MapRGBA here. They
   are defined in sdl/src/video/SDL_pixels.c, and use shifts
   and masks from the surface; they won't be as fast as constexpr
   masks but should inline decently fast without the function
   call overhead! (And then benchmark!)

 */
#if 0 /* SDL_BYTEORDER == SDL_BIG_ENDIAN */
  static constexpr uint32_t rmask = 0xff000000;
  static constexpr uint32_t gmask = 0x00ff0000;
  static constexpr uint32_t bmask = 0x0000ff00;
  static constexpr uint32_t amask = 0x000000ff;
  static constexpr uint32_t rshift = 24;
  static constexpr uint32_t gshift = 16;
  static constexpr uint32_t bshift = 8;
  static constexpr uint32_t ashift = 0;
#else
  static constexpr uint32_t rmask = 0x000000ff;
  static constexpr uint32_t gmask = 0x0000ff00;
  static constexpr uint32_t bmask = 0x00ff0000;
  static constexpr uint32_t amask = 0xff000000;
  static constexpr uint32_t ashift = 24;
  static constexpr uint32_t bshift = 16;
  static constexpr uint32_t gshift = 8;
  static constexpr uint32_t rshift = 0;
#endif

// Local copy of util's "line" code, to avoid dependency on that and
// facilitate inlining in this code. (That old weird interface also
// does a pointless heap allocation. Inlined as a regular object here
// was almost twice as fast in a line-heavy benchmark on 13 Feb 2016!)
namespace {
struct Line {
  int x0, y0, x1, y1;
  int dx, dy;
  int stepx, stepy;
  int frac;

  Line(int x0, int y0, int x1, int y1) :
    x0(x0), y0(y0), x1(x1), y1(y1) {

    dy = y1 - y0;
    dx = x1 - x0;

    if (dy < 0) {
      dy = -dy;
      stepy = -1;
    } else {
      stepy = 1;
    }

    if (dx < 0) {
      dx = -dx;
      stepx = -1;
    } else {
      stepx = 1;
    }

    dy <<= 1;
    dx <<= 1;

    if (dx > dy) {
      frac = dy - (dx >> 1);
    } else {
      frac = dx - (dy >> 1);
    }
  }

  bool Next(int &cx, int &cy) {
    if (dx > dy) {
      if (x0 == x1) return false;
      else {
        if (frac >= 0) {
          y0 += stepy;
          frac -= dx;
        }
        x0 += stepx;
        frac += dy;
        cx = x0;
        cy = y0;
        return true;
      }
    } else {
      if (y0 == y1) return false;
      else {
        if (frac >= 0) {
          x0 += stepx;
          frac -= dy;
        }
        y0 += stepy;
        frac += dx;
        cx = x0;
        cy = y0;
        return true;
      }
    }
  }
};
}  // namespace

SDL_Surface *sdlutil::resize_canvas(SDL_Surface *s,
                                    int w, int h, uint32_t color) {
  SDL_Surface *m = makesurface(w, h);
  if (!m) return nullptr;

  for (int y = 0; y < h; y++)
    for (int x = 0; x < w; x++) {
      uint32_t c = color;
      if (y < s->h && x < s->w)
        c = getpixel(s, x, y);
      setpixel(m, x, y, c);
    }

  return m;
}

// static
sdlutil::ByteOrder sdlutil::GetByteOrder(SDL_Surface *surf) {
  // PERF: Actually the R channel determines it completely.
  // But maybe nice to have the additional sanity check on A
  // until we know this works.
  if (surf->format->Aloss == 8) {
    if (surf->format->Rshift == 16 &&
        surf->format->Gshift == 8 &&
        surf->format->Bshift == 0) {
      return ByteOrder::ORGB;
    }

    // TODO: Probably there are other screen formats in practice.

  } else {
    switch (surf->format->Ashift) {
    case 0:
      // Could be RGBA or BGRA
      switch (surf->format->Rshift) {
      case 24: return ByteOrder::RGBA;
      case 8: return ByteOrder::BGRA;
      }
      break;
    case 24:
      // Could be ARGB or ABGR
      switch (surf->format->Rshift) {
      case 16: return ByteOrder::ARGB;
      case 0: return ByteOrder::ABGR;
      }
      break;
    }
  }

  LOG(FATAL) << "GetByteOrder: Surface not 32BPP or something else is wrong:\n"
             << SurfaceInfo(surf);
}

// Assumes rgba vector is the same width/height as the surface.
static void CopyRGBAVec(const vector<uint8_t> &rgba,
                        SDL_Surface *surface) {
  using ByteOrder = sdlutil::ByteOrder;
  uint32_t *p = (uint32_t *)surface->pixels;
  int width = surface->w;
  int height = surface->h;
  switch (sdlutil::GetByteOrder(surface)) {
  case ByteOrder::ORGB:
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        const int pidx = (y * width + x);
        const int idx = pidx * 4;
        const uint32_t r = rgba[idx + 0], g = rgba[idx + 1],
          b = rgba[idx + 2], a = 0;
        p[pidx] = (a << 24) | (r << 16) | (g << 8) | b;
      }
    }
    break;
  case ByteOrder::ARGB:
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        const int pidx = (y * width + x);
        const int idx = pidx * 4;
        const uint32_t r = rgba[idx + 0], g = rgba[idx + 1],
          b = rgba[idx + 2], a = rgba[idx + 3];
        p[pidx] = (a << 24) | (r << 16) | (g << 8) | b;
      }
    }
    break;
  case ByteOrder::RGBA:
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        const int pidx = (y * width + x);
        const int idx = pidx * 4;
        const uint32_t r = rgba[idx + 0], g = rgba[idx + 1],
          b = rgba[idx + 2], a = rgba[idx + 3];
        p[pidx] = (r << 24) | (g << 16) | (b << 8) | a;
      }
    }
    break;
  case ByteOrder::ABGR:
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        const int pidx = (y * width + x);
        const int idx = pidx * 4;
        const uint32_t r = rgba[idx + 0], g = rgba[idx + 1],
          b = rgba[idx + 2], a = rgba[idx + 3];
        p[pidx] = (a << 24) | (b << 16) | (g << 8) | r;
      }
    }
    break;
  case ByteOrder::BGRA:
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        const int pidx = (y * width + x);
        const int idx = pidx * 4;
        const uint32_t r = rgba[idx + 0], g = rgba[idx + 1],
          b = rgba[idx + 2], a = rgba[idx + 3];
        p[pidx] = (b << 24) | (g << 16) | (r << 8) | a;
      }
    }
    break;
  }
}

// Return false if there is nothing to do.
static inline bool ClipTo(int width, int height,
                          int &srcx, int &srcy,
                          int &w, int &h,
                          int &dstx, int &dsty) {

  if (dstx < 0) { w += dstx; srcx += dstx; dstx = 0; }
  if (dsty < 0) { h += dsty; srcy += dsty; dsty = 0; }
  if (w <= 0) return false;
  if (h <= 0) return false;

  const int yover = (dsty + h) - height;
  if (yover > 0) h -= yover;
  const int xover = (dstx + w) - width;
  if (xover > 0) w -= xover;

  if (w <= 0 || h <= 0) return false;

  return true;
}

// Copies the full rgba image to (x, y) on the dst. Clips.
void sdlutil::CopyRGBARect(const ImageRGBA &img,
                           int srcx, int srcy,
                           int w, int h,
                           int dstx, int dsty,
                           SDL_Surface *surface) {
  using ByteOrder = sdlutil::ByteOrder;
  uint32_t *p = (uint32_t *)surface->pixels;

  // First, clip.
  const int width = surface->w;
  const int height = surface->h;
  if (!ClipTo(width, height, srcx, srcy, w, h, dstx, dsty))
    return;

  // Now the rectangle given by (dstx,dsty,w,h) will fall within
  // the destination surface's size.

#define PACK4(a, b, c, d) \
  (((uint32_t)(a) << 24) | ((uint32_t)(b) << 16) | \
   ((uint32_t)(c) << 8) | ((uint32_t)(d)))

#define LOOP(A, B, C, D) do {                                         \
    for (int y = 0; y < h; y++) {                                     \
      for (int x = 0; x < w; x++) {                                   \
        const auto &[r, g, b, a] = img.GetPixel(srcx + x, srcy + y);  \
        const uint32_t d = PACK4(A, B, C, D);                         \
        const int pidx = ((dsty + y) * width + (dstx + x));           \
        p[pidx] = d;                                                  \
      }                                                               \
    }                                                                 \
  } while (false)

  switch (sdlutil::GetByteOrder(surface)) {
  case ByteOrder::ORGB:
    LOOP(0, r, g, b);
    break;
  case ByteOrder::ARGB:
    LOOP(a, r, g, b);
    break;
  case ByteOrder::RGBA:
    LOOP(r, g, b, a);
    break;
  case ByteOrder::ABGR:
    LOOP(a, b, g, r);
    break;
  case ByteOrder::BGRA:
    LOOP(b, g, r, a);
    break;
  }

#undef PACK4
#undef LOOP

}

// Copies the full rgba image to (x, y) on the dst. Clips.
void sdlutil::CopyRGBARectNX(const ImageRGBA &img,
                             int px,
                             int srcx, int srcy,
                             int w, int h,
                             int dstx, int dsty,
                             SDL_Surface *surface) {
  using ByteOrder = sdlutil::ByteOrder;
  uint32_t *p = (uint32_t *)surface->pixels;

  CHECK((dstx % px) == 0 && (dsty % px) == 0) << "I should handle "
    "this case by just drawing the last row/column of pixels in a "
    "second pass.";

  // First, clip.
  const int width = surface->w;
  const int height = surface->h;

  int dstw = w * px, dsth = h * px;
  if (!ClipTo(width, height, srcx, srcy, dstw, dsth, dstx, dsty))
    return;

  CHECK((dstw % px) == 0 && (dsth % px) == 0) << "This is implied "
    "by dstx,dsty being aligned.";

  w = dstw / px;
  h = dsth / px;

  // Now the rectangle given by (dstx,dsty,dstw,dsth) will fall within
  // the destination surface's size.

#define PACK4(a, b, c, d) \
  (((uint32_t)(a) << 24) | ((uint32_t)(b) << 16) | \
   ((uint32_t)(c) << 8) | ((uint32_t)(d)))

#define LOOP(A, B, C, D) do {                                         \
    for (int y = 0; y < h; y++) {                                     \
      for (int x = 0; x < w; x++) {                                   \
        const auto &[r, g, b, a] = img.GetPixel(srcx + x, srcy + y);  \
        const uint32_t d = PACK4(A, B, C, D);                         \
        for (int yy = 0; yy < px; yy++) {                             \
          for (int xx = 0; xx < px; xx++) {                           \
            const int pidx = ((dsty + y * px + yy) * width +          \
                              (dstx + x * px + xx));                  \
            p[pidx] = d;                                              \
          }                                                           \
        }                                                             \
      }                                                               \
    }                                                                 \
  } while (false)

  switch (sdlutil::GetByteOrder(surface)) {
  case ByteOrder::ORGB:
    LOOP(0, r, g, b);
    break;
  case ByteOrder::ARGB:
    LOOP(a, r, g, b);
    break;
  case ByteOrder::RGBA:
    LOOP(r, g, b, a);
    break;
  case ByteOrder::ABGR:
    LOOP(a, b, g, r);
    break;
  case ByteOrder::BGRA:
    LOOP(b, g, r, a);
    break;
  }

#undef PACK4
#undef LOOP

}

// Copy the RGBA image to the screen as quickly as I know how.
void sdlutil::CopyRGBAToScreen(const ImageRGBA &img, SDL_Surface *screen) {
  CopyRGBARect(img,
               0, 0, screen->w, screen->h,
               0, 0, screen);
}



SDL_Surface *sdlutil::FromRGBA(const ImageRGBA &rgba) {
  SDL_Surface *surf = makesurface(rgba.Width(), rgba.Height(), true);
  if (surf == nullptr) {
    return nullptr;
  }

  using ByteOrder = sdlutil::ByteOrder;
  // uint8_t *p = (uint8_t *)surface->pixels;
  uint32_t *p = (uint32_t *)surf->pixels;
  int width = surf->w;
  int height = surf->h;
  switch (sdlutil::GetByteOrder(surf)) {
  case ByteOrder::ORGB:
    LOG(FATAL) << "This case is not supported. The destination "
      "surface is expected to have an alpha channel.";
    break;
  case ByteOrder::ARGB:
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        const int pidx = (y * width + x);
        const auto [r, g, b, a] = rgba.GetPixel(x, y);
        p[pidx] = (a << 24) | (r << 16) | (g << 8) | b;
      }
    }
    break;
  case ByteOrder::RGBA:
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        const int pidx = (y * width + x);
        const auto [r, g, b, a] = rgba.GetPixel(x, y);
        p[pidx] = (r << 24) | (g << 16) | (b << 8) | a;
      }
    }
    break;
  case ByteOrder::ABGR:
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        const int pidx = (y * width + x);
        const auto [r, g, b, a] = rgba.GetPixel(x, y);
        p[pidx] = (a << 24) | (b << 16) | (g << 8) | r;
      }
    }
    break;
  case ByteOrder::BGRA:
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        const int pidx = (y * width + x);
        const auto [r, g, b, a] = rgba.GetPixel(x, y);
        p[pidx] = (b << 24) | (g << 16) | (r << 8) | a;
      }
    }
    break;
  }
  return surf;
}

// Note: Avoid "LoadImage" since windows.h may #define the
// symbol to LoadImageA etc. :(
SDL_Surface *sdlutil::LoadImageFile(const string &filename) {
  std::unique_ptr<ImageRGBA> img(ImageRGBA::Load(filename));
  if (img.get() == nullptr) return nullptr;
  return FromRGBA(*img);
}

bool sdlutil::SavePNG(const string &filename, SDL_Surface *surf) {
  // TODO: Can just use the ImageRGBA interface.
  // This could be implemented for other formats, of course.
  if (surf->format->BytesPerPixel != 4) return false;
  vector<uint8_t> rgba;
  rgba.reserve(surf->w * surf->h * 4);
  for (int i = 0; i < surf->w * surf->h; i++) {
    uint8_t r, g, b, a;
    SDL_GetRGBA(((uint32_t*)surf->pixels)[i], surf->format, &r, &g, &b, &a);
    rgba.push_back(r);
    rgba.push_back(g);
    rgba.push_back(b);
    rgba.push_back(a);
  }
  return !!stbi_write_png(filename.c_str(),
                          surf->w, surf->h, 4, rgba.data(), 4 * surf->w);
}

SDL_Surface *sdlutil::duplicate(SDL_Surface *surf) {
  return SDL_ConvertSurface(surf, surf->format, surf->flags);
}

void sdlutil::eatevents(int ticks, uint32_t mask) {
  int now = SDL_GetTicks();
  const int destiny = now + ticks;

  SDL_Event e;
  while (now < destiny) {
    SDL_PumpEvents();
    while (1 == SDL_PeepEvents(&e, 1, SDL_GETEVENT, mask)) { }

    now = SDL_GetTicks();
  }
}

/* recursive; nmips can't reasonably be very large */
bool sdlutil::make_mipmaps(SDL_Surface **s, int nmips) {
  if (nmips <= 1) return true;

  s[1] = shrink50(s[0]);
  if (!s[1]) return false;
  return make_mipmaps(&s[1], nmips - 1);
}

void sdlutil::clearsurface(SDL_Surface *s, uint32_t color) {
  SDL_FillRect(s, 0, color);
}

void sdlutil::ClearSurface(SDL_Surface *s,
                           uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
  uint32_t color = SDL_MapRGBA(s->format, r, g, b, a);
  SDL_FillRect(s, 0, color);
}

uint32_t sdlutil::mix2(uint32_t ia, uint32_t ib) {
  uint32_t ar = (ia >> rshift) & 0xFF;
  uint32_t br = (ib >> rshift) & 0xFF;

  uint32_t ag = (ia >> gshift) & 0xFF;
  uint32_t bg = (ib >> gshift) & 0xFF;

  uint32_t ab = (ia >> bshift) & 0xFF;
  uint32_t bb = (ib >> bshift) & 0xFF;

  uint32_t aa = (ia >> ashift) & 0xFF;
  uint32_t ba = (ib >> ashift) & 0xFF;

  /* if these are all fractions 0..1,
     color output is
       color1 * alpha1 + color2 * alpha2
       ---------------------------------
              alpha1 + alpha2               */

  uint32_t r = 0;
  uint32_t g = 0;
  uint32_t b = 0;

  if ((aa + ba) > 0) {
    /* really want
        (ar / 255) * (aa / 255) +
        (br / 255) * (ba / 255)
        -------------------------
        (aa / 255) + (ba / 255)

        to get r as a fraction. we then
        multiply by 255 to get the output
        as a byte.

        but this is:

        ar * (aa / 255) +
        br * (ba / 255)
        -----------------------
        (aa / 255) + (ba / 255)

        which can be simplified to

        ar * aa + br * ba
        -----------------
             aa + ba

        .. and doing the division last keeps
        us from quantization errors.
    */
    r = (ar * aa + br * ba) / (aa + ba);
    g = (ag * aa + bg * ba) / (aa + ba);
    b = (ab * aa + bb * ba) / (aa + ba);
  }

  /* alpha output is just the average */
  uint32_t a = (aa + ba) >> 1;

  return (r << rshift) | (g << gshift) | (b << bshift) | (a << ashift);
}

uint32_t sdlutil::mix4(uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
  /* XXX PERF

     would probably get better results (less quant error) and
     better performance to write this directly, using the calculation
     in mix2. */
  return mix2(mix2(a, b), mix2(c, d));
}

uint32_t sdlutil::Mix4(const SDL_Surface *surf,
                     uint32_t ai, uint32_t bi, uint32_t ci, uint32_t di) {

  uint8_t ar8, ag8, ab8, aa8;
  uint8_t br8, bg8, bb8, ba8;
  uint8_t cr8, cg8, cb8, ca8;
  uint8_t dr8, dg8, db8, da8;

  SDL_GetRGBA(ai, surf->format, &ar8, &ag8, &ab8, &aa8);
  SDL_GetRGBA(bi, surf->format, &br8, &bg8, &bb8, &ba8);
  SDL_GetRGBA(ci, surf->format, &cr8, &cg8, &cb8, &ca8);
  SDL_GetRGBA(di, surf->format, &dr8, &dg8, &db8, &da8);

  const uint32_t ar = ar8, ag = ag8, ab = ab8, aa = aa8;
  const uint32_t br = br8, bg = bg8, bb = bb8, ba = ba8;
  const uint32_t cr = cr8, cg = cg8, cb = cb8, ca = ca8;
  const uint32_t dr = dr8, dg = dg8, db = db8, da = da8;

  // This is the same as idea as Mix2; see that for the math.

  uint32_t r = 0;
  uint32_t g = 0;
  uint32_t b = 0;

  const uint32_t denom = aa + ba + ca + da;
  if (denom > 0) {
    r = (ar * aa + br * ba + cr * ca + dr * da) / denom;
    g = (ag * aa + bg * ba + cg * ca + dg * da) / denom;
    b = (ab * aa + bb * ba + cb * ca + db * da) / denom;
  }

  /* alpha output is just the average */
  const uint32_t a = denom >> 2;

  return SDL_MapRGBA(surf->format, r, g, b, a);
}

uint32_t sdlutil::Mix2(const SDL_Surface *surf,
                     uint32_t ia, uint32_t ib) {
  uint8_t ar8, ag8, ab8, aa8;
  uint8_t br8, bg8, bb8, ba8;
  SDL_GetRGBA(ia, surf->format, &ar8, &ag8, &ab8, &aa8);
  SDL_GetRGBA(ib, surf->format, &br8, &bg8, &bb8, &ba8);

  uint32_t ar = ar8, ag = ag8, ab = ab8, aa = aa8;
  uint32_t br = br8, bg = bg8, bb = bb8, ba = ba8;

  /* if these are all fractions 0..1,
     color output is
       color1 * alpha1 + color2 * alpha2
       ---------------------------------
              alpha1 + alpha2               */

  uint32_t r = 0;
  uint32_t g = 0;
  uint32_t b = 0;

  if ((aa + ba) > 0) {
    /* really want
        (ar / 255) * (aa / 255) +
        (br / 255) * (ba / 255)
        -------------------------
        (aa / 255) + (ba / 255)

        to get r as a fraction. we then
        multiply by 255 to get the output
        as a byte.

        but this is:

        ar * (aa / 255) +
        br * (ba / 255)
        -----------------------
        (aa / 255) + (ba / 255)

        which can be simplified to

        ar * aa + br * ba
        -----------------
             aa + ba

        .. and doing the division last keeps
        us from quantization errors.
    */
    r = (ar * aa + br * ba) / (aa + ba);
    g = (ag * aa + bg * ba) / (aa + ba);
    b = (ab * aa + bb * ba) / (aa + ba);
  }

  /* alpha output is just the average */
  const uint32_t a = (aa + ba) >> 1;

  return SDL_MapRGBA(surf->format, r, g, b, a);
}


uint32_t sdlutil::mixfrac(uint32_t a, uint32_t b, float f) {
  uint32_t factor  = (uint32_t) (f * 0x100);
  uint32_t ofactor = 0x100 - factor;

  uint32_t o24 = (((a >> 24) & 0xFF) * factor +
                ((b >> 24) & 0xFF) * ofactor) >> 8;

  uint32_t o16 = (((a >> 16) & 0xFF) * factor +
                ((b >> 16) & 0xFF) * ofactor) >> 8;

  uint32_t o8 = (((a >> 8) & 0xFF) * factor +
                ((b >> 8) & 0xFF) * ofactor) >> 8;


  uint32_t o = ((a & 0xFF) * factor +
              (b & 0xFF) * ofactor) >> 8;

  return (o24 << 24) | (o16 << 16) | (o8 << 8) | o;
}


uint32_t sdlutil::hsv(SDL_Surface *sur, float h, float s, float v, float a) {
  int r, g, b;
  if (s == 0.0f) {
    r = (int)(v * 255);
    g = (int)(v * 255);
    b = (int)(v * 255);
  } else {
    float hue = h * 6.0f;
    int fh = (int)hue;
    float var_1 = v * (1 - s);
    float var_2 = v * (1 - s * (hue - fh));
    float var_3 = v * (1 - s * (1 - (hue - fh)));

    float red, green, blue;

    switch ((int)hue) {
    case 0:  red = v     ; green = var_3 ; blue = var_1; break;
    case 1:  red = var_2 ; green = v     ; blue = var_1; break;
    case 2:  red = var_1 ; green = v     ; blue = var_3; break;
    case 3:  red = var_1 ; green = var_2 ; blue = v    ; break;
    case 4:  red = var_3 ; green = var_1 ; blue = v    ; break;
    default: red = v     ; green = var_1 ; blue = var_2; break;
    }

    r = (int)(red * 255);
    g = (int)(green * 255);
    b = (int)(blue * 255);
  }

  return SDL_MapRGBA(sur->format, r, g, b, (int)(a * 255));
}

SDL_Surface *sdlutil::alphadim(SDL_Surface *src) {
  /* must be 32 bpp */
  if (src->format->BytesPerPixel != 4) return nullptr;

  int ww = src->w, hh = src->h;

  SDL_Surface *ret = makesurface(ww, hh);

  if (!ret) return nullptr;

  slock(ret);
  slock(src);

  for (int y = 0; y < hh; y++) {
    for (int x = 0; x < ww; x++) {
      uint32_t color = *((uint32_t *)src->pixels + y * src->pitch / 4 + x);

      /* divide alpha channel by 2 */
      uint8_t r, g, b, a;
      SDL_GetRGBA(color, src->format, &r, &g, &b, &a);
      a >>= 1;

      color = SDL_MapRGBA(src->format, r, g, b, a);

      *((uint32_t *)ret->pixels + y * ret->pitch / 4 + x) = color;
    }
  }

  sulock(src);
  sulock(ret);

  return ret;
}

SDL_Surface *sdlutil::shrink50(SDL_Surface *src) {
  /* must be 32 bpp */
  if (src->format->BytesPerPixel != 4) return nullptr;

  int ww = src->w, hh = src->h;

  /* we need there to be at least two pixels for each destination
     pixel, so if the src dimensions are not even, then we ignore
     that last pixel. */
  if (ww & 1) ww -= 1;
  if (hh & 1) hh -= 1;

  if (ww <= 0) return nullptr;
  if (hh <= 0) return nullptr;

  SDL_Surface *ret = makesurface(ww / 2, hh / 2);

  if (!ret) return nullptr;

  slock(ret);
  slock(src);

  for (int y = 0; y < ret->h; y++) {
    for (int x = 0; x < ret->w; x++) {
      /* get and mix four src pixels for each dst pixel */
      uint32_t c1 = *((uint32_t *)src->pixels + (y*2)*src->pitch/4 + (x*2));
      uint32_t c2 = *((uint32_t *)src->pixels + (1+y*2)*src->pitch/4 + (1+x*2));
      uint32_t c3 = *((uint32_t *)src->pixels + (1+y*2)*src->pitch/4 + (x*2));
      uint32_t c4 = *((uint32_t *)src->pixels + (y*2)*src->pitch/4 + (1+x*2));

      uint32_t color = Mix4(src, c1, c2, c3, c4);

      *((uint32_t *)ret->pixels + y*ret->pitch/4 + x) = color;
    }
  }

  sulock(src);
  sulock(ret);

  return ret;
}

SDL_Surface *sdlutil::grow2x(SDL_Surface *src) {
  /* must be 32 bpp */
  if (src->format->BytesPerPixel != 4) return nullptr;

  int ww = src->w, hh = src->h;


  SDL_Surface *ret = makesurface(ww << 1, hh << 1);
  if (!ret) return nullptr;

  slock(ret);
  slock(src);

  uint8_t *p = (uint8_t*)src->pixels;
  uint8_t *pdest = (uint8_t*)ret->pixels;

  int ww2 = ww << 1;
  for (int y = 0; y < hh; y++) {
    for (int x = 0; x < ww; x++) {
      uint32_t rgba = *(uint32_t*)(p + 4 * (y * ww + x));

      // Write four pixels.
      int y2 = y << 1;
      int x2 = x << 1;
      *(uint32_t*)(pdest + 4 * (y2 * ww2 + x2)) = rgba;
      *(uint32_t*)(pdest + 4 * (y2 * ww2 + x2 + 1)) = rgba;
      *(uint32_t*)(pdest + 4 * ((y2 + 1) * ww2 + x2)) = rgba;
      *(uint32_t*)(pdest + 4 * ((y2 + 1) * ww2 + x2 + 1)) = rgba;
    }
  }

  sulock(src);
  sulock(ret);

  return ret;
}

SDL_Surface *sdlutil::GrowX(SDL_Surface *src, int px) {
  /* must be 32 bpp */
  if (src->format->BytesPerPixel != 4) {
    return nullptr;
  }

  int ww = src->w, hh = src->h;
  SDL_Surface *ret = makesurface(ww * px, hh * px);
  if (!ret) {
    return nullptr;
  }

  slock(ret);
  slock(src);

  uint8_t *p = (uint8_t*)src->pixels;
  uint8_t *pdest = (uint8_t*)ret->pixels;

  int ww2 = ww * px;
  for (int y = 0; y < hh; y++) {
    for (int x = 0; x < ww; x++) {
      const uint32_t rgba = *(uint32_t*)(p + 4 * (y * ww + x));

      // Write px * px pixels.
      for (int yu = 0; yu < px; yu++) {
        const int y2 = y * px + yu;
        for (int xu = 0; xu < px; xu++) {
          const int x2 = x * px + xu;
          *(uint32_t*)(pdest + 4 * (y2 * ww2 + x2)) = rgba;
        }
      }
    }
  }

  sulock(src);
  sulock(ret);

  return ret;
}

void sdlutil::SetIcon(std::string_view filename) {
  std::unique_ptr<ImageRGBA> img(ImageRGBA::Load(filename));
  if (img.get() != nullptr) {
    SDL_Surface *icon = SDL_CreateRGBSurface(
        0, 32, 32, 32,
        // Something we know is supported by ByteOrder: ARGB.
        0xFF0000, 0xFF00, 0xFF, 0xFF000000);
    if (icon == nullptr) return;

    CopyRGBARect(*img, 0, 0, 32, 32, 0, 0, icon);

    // TODO: Could derive mask from alpha?
    SDL_WM_SetIcon(icon, nullptr);

    // TODO: Unclear from documentation whether SetIcon is taking
    // ownership of the icon? Leaking a single 32x32 image is not
    // a big deal, though.
    // SDL_FreeSurface(ss);
  }
}


/* try to make a hardware surface, and, failing that,
   make a software surface */
SDL_Surface *sdlutil::makesurface(int w, int h, bool alpha) {
  /* PERF need to investigate relative performance
     of sw/hw surfaces */
  SDL_Surface *ss = 0;
#if 0
    SDL_CreateRGBSurface(SDL_HWSURFACE |
                         (alpha?SDL_SRCALPHA:0),
                         w, h, 32,
                         rmask, gmask, bmask,
                         amask);
#endif

  if (!ss) ss = SDL_CreateRGBSurface(SDL_SWSURFACE |
                                     (alpha?SDL_SRCALPHA:0),
                                     w, h, 32,
                                     rmask, gmask, bmask,
                                     amask);

  if (ss && !alpha) SDL_SetAlpha(ss, 0, 255);

  /* then convert to the display format. */
# if USE_DISPLAY_FORMAT
  if (ss) {
    SDL_Surface *rr;
    if (alpha) rr = SDL_DisplayFormatAlpha(ss);
    else rr = SDL_DisplayFormat(ss);
    SDL_FreeSurface(ss);
    return rr;
  } else {
    return nullptr;
  }
# else
  return ss;
# endif
}

SDL_Surface *sdlutil::makealpharect(int w, int h, int r, int g, int b, int a) {
  SDL_Surface *ret = makesurface(w, h);

  if (!ret) return nullptr;

  uint32_t color = SDL_MapRGBA(ret->format, r, g, b, a);

  SDL_FillRect(ret, 0, color);

  return ret;
}

SDL_Surface *sdlutil::makealpharectgrad(int w, int h,
                                        int r1, int b1, int g1, int a1,
                                        int r2, int b2, int g2, int a2,
                                        float bias) {
  SDL_Surface *ret = makesurface(w, h);

  if (!ret) return nullptr;

  /* draws each line as a separate rectangle. */
  for (int i = 0; i < h; i++) {
    /* no bias yet */
    float frac = 1.0f - ((float)i / (float)h);
    int r = (int) ((r1 * frac) + (r2 * (1.0 - frac)));
    int g = (int) ((g1 * frac) + (g2 * (1.0 - frac)));
    int b = (int) ((b1 * frac) + (b2 * (1.0 - frac)));
    int a = (int) ((a1 * frac) + (a2 * (1.0 - frac)));
    uint32_t color = SDL_MapRGBA(ret->format, r, g, b, a);
    SDL_Rect rect;
    rect.x = 0;
    rect.y = i;
    rect.w = w;
    rect.h = 1;
    SDL_FillRect(ret, &rect, color);
  }

  return ret;
}

void sdlutil::fillrect(SDL_Surface *s, uint32_t color,
                       int x, int y, int w, int h) {
  SDL_Rect dst;
  dst.x = x;
  dst.y = y;
  dst.w = w;
  dst.h = h;
  SDL_FillRect(s, &dst, color);
}

void sdlutil::FillRectRGB(SDL_Surface *s,
                          int x, int y, int w, int h,
                          uint8_t r, uint8_t g, uint8_t b) {
  SDL_Rect dst;
  dst.x = x;
  dst.y = y;
  dst.w = w;
  dst.h = h;
  SDL_FillRect(s, &dst, SDL_MapRGB(s->format, r, g, b));
}

void sdlutil::blitall(SDL_Surface *src, SDL_Surface *dst, int x, int y) {
  SDL_Rect rec;
  rec.x = x;
  rec.y = y;
  SDL_BlitSurface(src, 0, dst, &rec);
}

void sdlutil::outline(SDL_Surface *s, int n, int r, int g, int b, int a) {
  uint32_t color = SDL_MapRGBA(s->format, r, g, b, a);

  SDL_Rect dst;
  dst.x = 0;
  dst.y = 0;
  dst.h = n;
  dst.w = s->w;
  SDL_FillRect(s, &dst, color);
  dst.w = n;
  dst.h = s->h;
  SDL_FillRect(s, &dst, color);
  dst.x = s->w - n;
  SDL_FillRect(s, &dst, color);
  dst.y = s->h - n;
  dst.x = 0;
  dst.w = s->w;
  dst.h = n;
  SDL_FillRect(s, &dst, color);
}

void sdlutil::DrawCircle32(SDL_Surface *surf,
                           int x0, int y0, int radius, uint32_t color) {
  // Only 32-bit!
  if (surf->format->BytesPerPixel != 4) return;

  uint32_t *bufp = (uint32_t *)surf->pixels;
  const int stride = surf->pitch >> 2;
  const int width = surf->w, height = surf->h;
  auto SetPixel = [color, bufp, stride, width, height](int x, int y) {
      if (x < 0 || y < 0 || x >= width || y >= height)
        return;
      bufp[y * stride + x] = color;
    };

  SetPixel(x0, y0 + radius);
  SetPixel(x0, y0 - radius);
  SetPixel(x0 + radius, y0);
  SetPixel(x0 - radius, y0);

  int f = 1 - radius;
  int ddF_x = 1;
  int ddF_y = -2 * radius;
  int x = 0;
  int y = radius;

  while (x < y) {
    // fun loop (x, y, f, ddF_x, ddF_y) =
    if (f >= 0) {
      y--;
      f += 2 + ddF_y;
      ddF_y += 2;
    }

    x++;
    ddF_x += 2;
    f += ddF_x;

    SetPixel(x0 + x, y0 + y);
    SetPixel(x0 - x, y0 + y);
    SetPixel(x0 + x, y0 - y);
    SetPixel(x0 - x, y0 - y);
    SetPixel(x0 + y, y0 + x);
    SetPixel(x0 - y, y0 + x);
    SetPixel(x0 + y, y0 - x);
    SetPixel(x0 - y, y0 - x);
  }
}

SDL_Surface *sdlutil::makescreen(int w, int h) {
  /* Can't use HWSURFACE here, because not handling this SDL_BlitSurface
     case mentioned in the documentation:

     "If either of the surfaces were in video memory, and the blit returns -2,
     the video memory was lost, so it should be reloaded with artwork
     and re-blitted."

     Plus, on Windows, the only time you get HWSURFACE is with FULLSCREEN.

     -- Adam
  */

  /* SDL_ANYFORMAT
     "Normally, if a video surface of the requested bits-per-pixel (bpp)
     is not available, SDL will emulate one with a shadow surface.
     Passing SDL_ANYFORMAT prevents this and causes SDL to use the
     video surface, regardless of its pixel depth."

     Probably should not pass this.

     -- Adam
  */


  /* SDL_DOUBLEBUF only valid with SDL_HWSURFACE! */
  SDL_Surface *ret = SDL_SetVideoMode(w, h, 32,
                                      SDL_SWSURFACE |
                                      SDL_RESIZABLE);
  return ret;
}

/* XXX apparently you are not supposed
   to lock the surface for blits, only
   direct pixel access. should check
   this out if mysterious crashes on some
   systems ... */

void sdlutil::slock(SDL_Surface *screen) {
  if (SDL_MUSTLOCK(screen)) {
    SDL_LockSurface(screen);
  }
}

void sdlutil::sulock(SDL_Surface *screen) {
  if (SDL_MUSTLOCK(screen)) {
    SDL_UnlockSurface(screen);
  }
}


// XXX There may be strict aliasing problems with this and the
// next function.
/* lock before calling */
/* getpixel function only:
   Copyright (c) 2004 Bill Kendrick, Tom Murphy
   from the GPL "Tux Paint" */
uint32_t sdlutil::getpixel(SDL_Surface *surface, int x, int y) {
  uint32_t pixel = 0;

  /* Check that x/y values are within the bounds of this surface... */
  if (x >= 0 && y >= 0 && x < surface -> w && y < surface -> h) {

    /* Determine bytes-per-pixel for the surface in question: */

    int bpp = surface->format->BytesPerPixel;

    /* Set a pointer to the exact location in memory of the pixel
       in question: */

    uint8_t *p =
      (uint8_t *) (((uint8_t *)surface->pixels) +  /* Start at top of RAM */
                 (y * surface->pitch) +  /* Go down Y lines */
                 (x * bpp));             /* Go in X pixels */

    /* Return the correctly-sized piece of data containing the
       pixel's value (an 8-bit palette value, or a 16-, 24- or 32-bit
       RGB value) */

    if (bpp == 1)         /* 8-bit display */
      pixel = *p;
    else if (bpp == 2)    /* 16-bit display */
      pixel = *(Uint16 *)p;
    else if (bpp == 3) {    /* 24-bit display */
      /* Depending on the byte-order, it could be stored RGB or BGR! */

      if (SDL_BYTEORDER == SDL_BIG_ENDIAN)
        pixel = p[0] << 16 | p[1] << 8 | p[2];
      else
        pixel = p[0] | p[1] << 8 | p[2] << 16;
    } else if (bpp == 4) {    /* 32-bit display */
      pixel = *(uint32_t *)p;
    }
  }
  return pixel;
}

/* based on getpixel above */
void sdlutil::setpixel(SDL_Surface *surface, int x, int y, uint32_t px) {
  /* Check that x/y values are within the bounds of this surface... */

  if (x >= 0 && y >= 0 && x < surface -> w && y < surface -> h) {
    /* Determine bytes-per-pixel for the surface in question: */

    int bpp = surface->format->BytesPerPixel;

    /* Set a pointer to the exact location in memory of the pixel
       in question: */

    uint8_t *p =
      (uint8_t *) (((uint8_t *)surface->pixels) +  /* Start at top of RAM */
                 (y * surface->pitch) +  /* Go down Y lines */
                 (x * bpp));             /* Go in X pixels */

    /* Return the correctly-sized piece of data containing the
       pixel's value (an 8-bit palette value, or a 16-, 24- or 32-bit
       RGB value) */

    if (bpp == 1)         /* 8-bit display */
      *p = px;
    else if (bpp == 2)    /* 16-bit display */
      *(Uint16 *)p = px;
    else if (bpp == 3) {    /* 24-bit display */
      /* Depending on the byte-order, it could be stored RGB or BGR! */

      /* XX never tested */
      if (SDL_BYTEORDER == SDL_BIG_ENDIAN) {
        p += 2;
        for (int i = 0; i < 3; i++) {
          *p = px & 255;
          px >>= 8;
          p--;
        }
      }
      else {
        for (int i = 0; i < 3; i++) {
          *p = px & 255;
          px >>= 8;
          p++;
        }
      }
    } else if (bpp == 4) {    /* 32-bit display */
      *(uint32_t *)p = px;
    }
  }
}

void sdlutil::drawline(SDL_Surface *screen, int x0, int y0,
                       int x1, int y1,
                       uint8_t R, uint8_t G, uint8_t B) {
  Line l{x0, y0, x1, y1};

  /* direct pixel access */
  slock(screen);

  if (4 == screen->format->BytesPerPixel) {
    const uint32_t color = SDL_MapRGB(screen->format, R, G, B);
    // This is the most common case, so unroll.
    const uint32_t stride = screen->pitch/4;

    uint32_t *bufp = (uint32_t *)screen->pixels;

    bufp[y0 * stride + x0] = color;
    int x, y;
    while (l.Next(x, y))
      bufp[y * stride + x] = color;

  } else {
    // General case.
    int x, y;
    drawpixel(screen, x0, y0, R, G, B);
    while (l.Next(x, y)) {
      drawpixel(screen, x, y, R, G, B);
    }
  }

  sulock(screen);
}

void sdlutil::DrawClipLine32(SDL_Surface *screen, int x0, int y0,
                             int x1, int y1,
                             uint32_t color) {
  // Only 32-bit!
  if (screen->format->BytesPerPixel != 4) return;

  // However, if a line completely misses the screen, we can
  // just draw nothing.
  if (x0 < 0 && x1 < 0)
    return;
  if (x0 >= screen->w && x1 >= screen->w)
    return;
  if (y0 < 0 && y1 < 0)
    return;
  if (y0 >= screen->h && y1 >= screen->h)
    return;

  Line l{x0, y0, x1, y1};

  uint32_t *bufp = (uint32_t *)screen->pixels;
  int stride = screen->pitch >> 2;
  auto SetPixel = [color, bufp, stride](int x, int y) {
      bufp[y * stride + x] = color;
    };

  /* direct pixel access */
  slock(screen);
  int x, y;
  if (x0 >= 0 && y0 >= 0 && x0 < screen->w && y0 < screen->h)
    SetPixel(x0, y0);
  while (l.Next(x, y)) {
    if (x >= 0 && y >= 0 && x < screen->w && y < screen->h) {
      SetPixel(x, y);
    }
  }
  sulock(screen);
}

void sdlutil::drawclipline(SDL_Surface *screen, int x0, int y0,
                           int x1, int y1,
                           uint8_t R, uint8_t G, uint8_t B) {
  /* PERF could maprgb once */

  /* PERF clipping can be much more efficient, but it is a
     bit tricky to do in integer coordinates, we can not often
     represent a clipped line exactly with new integer points.
  */

  // However, if a line completely misses the screen, we can
  // just draw nothing.
  if (x0 < 0 && x1 < 0)
    return;
  if (x0 >= screen->w && x1 >= screen->w)
    return;
  if (y0 < 0 && y1 < 0)
    return;
  if (y0 >= screen->h && y1 >= screen->h)
    return;

  Line l{x0, y0, x1, y1};

  /* direct pixel access */
  slock(screen);
  int x, y;
  if (x0 >= 0 && y0 >= 0 && x0 < screen->w && y0 < screen->h)
    drawpixel(screen, x0, y0, R, G, B);
  while (l.Next(x, y)) {
    if (x >= 0 && y >= 0 && x < screen->w && y < screen->h)
      drawpixel(screen, x, y, R, G, B);
  }
  sulock(screen);
}

bool sdlutil::clipsegment(float cx0, float cy0, float cx1, float cy1,
                          float &x0, float &y0, float &x1, float &y1) {
  // Cohen--Sutherland clipping.
  enum Code : uint32_t {
    LEFT = 1 << 0,
    RIGHT = 1 << 1,
    BOTTOM = 1 << 2,
    TOP = 1 << 3,
  };

  auto GetCode = [cx0, cy0, cx1, cy1](float x, float y) {
    uint32_t code = 0;
    if (x < cx0) code |= LEFT;
    else if (x > cx1) code |= RIGHT;
    if (y < cy0) code |= TOP;
    else if (y > cy1) code |= BOTTOM;
    return code;
  };

  // Line completely misses screen.
  uint32_t code0 = GetCode(x0, y0), code1 = GetCode(x1, y1);
  for (;;) {
    if (0 == (code0 | code1)) {
      // Both endpoints inside now.
      return true;
    }

    // Completely missing the clip rectangle. Draw nothing.
    if (code0 & code1)
      return false;

    auto Update = [cx0, cy0, cx1, cy1, x0, y0, x1, y1, &GetCode]
      (float &x, float &y, uint32_t &code) {
      // (Beware that x aliases x0 or x1, etc.
      if (code & TOP) {
        x = x0 + (x1 - x0) * (cy1 - y0) / (y1 - y0);
        y = cy1;
      } else if (code & BOTTOM) {
        x = x0 + (x1 - x0) * (cy0 - y0) / (y1 - y0);
        y = cy0;
      } else if (code & LEFT) {
        y = y0 + (y1 - y0) * (cx0 - x0) / (x1 - x0);
        x = cx0;
      } else if (code & RIGHT) {
        y = y0 + (y1 - y0) * (cx1 - x0) / (x1 - x0);
        x = cx1;
      }
      // PERF can probably be rolled into branches above.
      code = GetCode(x, y);
    };

    if (code0) Update(x0, y0, code0);
    else Update(x1, y1, code1);
  }
}

void sdlutil::drawbox(SDL_Surface *s, int x, int y, int w, int h,
                      uint8_t r, uint8_t g, uint8_t b) {
  // PERF can unroll a lot of this. Also, straight lines can be
  // drawn much more easily than with Bresenham, and can be clipped
  // much more easily as well.
  // Top
  drawclipline(s, x, y, x + w - 1, y, r, g, b);
  // Left
  drawclipline(s, x, y, x, y + h - 1, r, g, b);
  // Right
  drawclipline(s, x + w - 1, y, x + w - 1, y + h - 1, r, g, b);
  // Bottom
  drawclipline(s, x, y + h - 1, x + w - 1, y + h - 1, r, g, b);
}

void sdlutil::DrawBox32(SDL_Surface *s, int x, int y, int w, int h,
                        uint32_t rgba) {
  // PERF: Same as above.
  // Top
  DrawClipLine32(s, x, y, x + w - 1, y, rgba);
  // Left
  DrawClipLine32(s, x, y, x, y + h - 1, rgba);
  // Right
  DrawClipLine32(s, x + w - 1, y, x + w - 1, y + h - 1, rgba);
  // Bottom
  DrawClipLine32(s, x, y + h - 1, x + w - 1, y + h - 1, rgba);
}


/* XXX change to use function pointer? */
/* lock before calling */
void sdlutil::drawpixel(SDL_Surface *screen, int x, int y,
                        uint8_t R, uint8_t G, uint8_t B) {
  uint32_t color = SDL_MapRGB(screen->format, R, G, B);
  switch (screen->format->BytesPerPixel) {
  case 1: // Assuming 8-bpp
    {
      uint8_t *bufp;
      bufp = (uint8_t *)screen->pixels + y*screen->pitch + x;
      *bufp = color;
    }
    break;
  case 2: // Probably 15-bpp or 16-bpp
    {
      Uint16 *bufp;
      bufp = (Uint16 *)screen->pixels + y*screen->pitch/2 + x;
      *bufp = color;
    }
    break;
  case 3: // Slow 24-bpp mode, usually not used
    {
      uint8_t *bufp;
      bufp = (uint8_t *)screen->pixels + y*screen->pitch + x * 3;
      if (SDL_BYTEORDER == SDL_LIL_ENDIAN) {
        bufp[0] = color;
        bufp[1] = color >> 8;
        bufp[2] = color >> 16;
      } else {
        bufp[2] = color;
        bufp[1] = color >> 8;
        bufp[0] = color >> 16;
      }
    }
    break;
  case 4: // Probably 32-bpp
    {
      uint32_t *bufp;
      bufp = (uint32_t *)screen->pixels + y*screen->pitch/4 + x;
      *bufp = color;
    }
    break;
  }
}

void sdlutil::drawclippixel(SDL_Surface *screen, int x, int y,
                            uint8_t r, uint8_t g, uint8_t b) {
  if (x < 0 || y < 0 || x >= screen->w || y >= screen->h)
    return;
  drawpixel(screen, x, y, r, g, b);
}

std::string sdlutil::SurfaceInfo(SDL_Surface *surf) {

#if 0
  Uint8  BitsPerPixel;
  Uint8  BytesPerPixel;
  Uint8  Rloss;
  Uint8  Gloss;
  Uint8  Bloss;
  Uint8  Aloss;
  Uint8  Rshift;
  Uint8  Gshift;
  Uint8  Bshift;
  Uint8  Ashift;
  Uint32 Rmask;
  Uint32 Gmask;
  Uint32 Bmask;
  Uint32 Amask;

  /** RGB color key information */
  Uint32 colorkey;
  /** Alpha value information (per-surface alpha) */
  Uint8  alpha;
#endif

  std::string info =
    StringPrintf("Surface sized %d x %d\n",
                 surf->w, surf->h);

  StringAppendF(
      &info,
      "  bits/bytes per pixel: %d/%d\n"
      "  RGBA loss: %d %d %d %d\n"
      "  RGBA shifts: %d %d %d %d\n"
      "  RGBA masks: %x %x %x %x\n",
      surf->format->BitsPerPixel,
      surf->format->BytesPerPixel,
      surf->format->Rloss,
      surf->format->Gloss,
      surf->format->Bloss,
      surf->format->Aloss,
      surf->format->Rshift,
      surf->format->Gshift,
      surf->format->Bshift,
      surf->format->Ashift,
      surf->format->Rmask,
      surf->format->Gmask,
      surf->format->Bmask,
      surf->format->Amask);

  const int f = surf->flags;

  StringAppendF(&info, "Flags:\n");
#define INFO(flag, str) \
  StringAppendF(&info, "  %s " str "\n", (f & (flag))?"X":" ");
  INFO(SDL_HWSURFACE,   "In Video Memory");
  INFO(SDL_ASYNCBLIT,   "Asynch blits if possible");
  INFO(SDL_ANYFORMAT,   "Any pixel format");
  INFO(SDL_HWPALETTE,   "Exclusive palette");
  INFO(SDL_DOUBLEBUF,   "Double buffered");
  INFO(SDL_FULLSCREEN,  "Full screen");
  INFO(SDL_OPENGL,      "OpenGL context");
  INFO(SDL_OPENGLBLIT,  "Supports OpenGL blitting");
  INFO(SDL_RESIZABLE,   "Can resize");
  INFO(SDL_HWACCEL,     "Blit is hardware accelerated");
  INFO(SDL_SRCCOLORKEY, "Colorkey blitting");
  INFO(SDL_RLEACCEL,    "RLE accelerated blitting");
  INFO(SDL_SRCALPHA,    "Blit uses source alpha blending");
  INFO(SDL_PREALLOC,    "Uses preallocated memory");
#undef INFO

  return info;
}

SDL_Surface *sdlutil::fliphoriz(SDL_Surface *src) {
  SDL_Surface *dst = makesurface(src->w, src->h);
  if (!dst) return nullptr;

  slock(src);
  slock(dst);

  for (int y = 0; y < src->h; y++) {
    for (int x = 0; x < src->w; x++) {
      uint32_t px = getpixel(src, x, y);
      setpixel(dst, (src->w - x) - 1, y, px);
    }
  }

  slock(dst);
  sulock(src);
  return dst;
}

// Fills a flat triangle (vertices 2 and 3 have the same y coordinate, given once as y23)
// with the given color. Works by drawing the edges from v1 to v2 and v1 to v3 using
// Bresenham's algorithm, then drawing a horizontal line between the generated x coordinates
// when they advance. ASSUMES bpp = 4.
static void BresenhamTriangle32(SDL_Surface *surf,
                                int x1, int y1, int x2, int y23, int x3, uint32_t color) {
  uint32_t *bufp = (uint32_t *)surf->pixels;
  const int height = surf->h;
  const int width = surf->w;
  const int stride = surf->pitch >> 2;
  auto DrawHorizLine = [color, bufp, height, width, stride](int x1, int y, int x2) {
      if (y < 0 || y >= height)
        return;

      if (x2 < x1) std::swap(x1, x2);

      x1 = std::max(x1, 0);
      x2 = std::min(x2, width - 1);

      uint32_t *p = &bufp[y * stride + x1];
      for (int i = 0; i < (x2 - x1); i++) {
        *p = color;
        p++;
      }
    };

  const int y2 = y23, y3 = y23;
  int tmpx1 = x1, tmpy1 = y1;
  int tmpx2 = x1, tmpy2 = y1;

  bool changed1 = false, changed2 = false;

  int dx1 = std::abs(x2 - x1);
  int dy1 = std::abs(y2 - y1);

  int dx2 = std::abs(x3 - x1);
  int dy2 = std::abs(y3 - y1);

  // Like copysign for ints. -1, 0, or 1
  auto Sign = [](int i) { return (i > 0) - (i < 0); };

  int signx1 = Sign(x2 - x1);
  int signx2 = Sign(x3 - x1);

  int signy1 = Sign(y2 - y1);
  int signy2 = Sign(y3 - y1);

  // Canonicalize the order. PERF: Maybe caller can ensure this; it's bad
  // that we branch in the loop below.
  if (dy1 > dx1) {
    std::swap(dy1, dx1);
    changed1 = true;
  }

  if (dy2 > dx2) {
    std::swap(dx2, dy2);
    changed2 = true;
  }

  int e1 = 2 * dy1 - dx1;
  int e2 = 2 * dy2 - dx2;

  // PERF: Can exit loop early if y value is offscreen.
  for (int i = 0; i <= dx1; i++) {
    DrawHorizLine(tmpx1, tmpy1, tmpx2);

    // This is like the merge step in merge sort now; generate points
    // from each line until it catches up with the other one.
    while (e1 >= 0) {
      if (changed1)
        tmpx1 += signx1;
      else
        tmpy1 += signy1;
      e1 -= 2 * dx1;
    }

    if (changed1)
      tmpy1 += signy1;
    else
      tmpx1 += signx1;

    e1 += 2 * dy1;

    while (tmpy2 != tmpy1) {
      while (e2 >= 0) {
        if (changed2)
          tmpx2 += signx2;
        else
          tmpy2 += signy2;
        e2 -= 2 * dx2;
      }

      if (changed2)
        tmpy2 += signy2;
      else
        tmpx2 += signx2;

      e2 += 2 * dy2;
    }
  }
}

void sdlutil::FillTriangle32(SDL_Surface *surf,
                             int x1, int y1, int x2, int y2, int x3, int y3,
                             uint32_t color) {
  // Only 32-bit!
  if (surf->format->BytesPerPixel != 4) return;

# define SWAP(a, b) do { std::swap(x ## a, x ## b); std::swap(y ## a, y ## b); } while (0)
  // Strategy is to draw two triangles with a horizontal edge, using
  // BresenhamTriangle32. To figure out what case we're in, sort the
  // three vertices such that y1 <= y2 <= y3.

  // PERF: This is "sort all but the last, sort all but the first, and
  // then sort all but the last" which is decent here (max 3
  // comparisons), although in some branches we can know that we don't
  // need to execute other ones.
  if (y1 > y2) {
    SWAP(1, 2);
  }
  if (y2 > y3) {
    SWAP(2, 3);
  }
  if (y1 > y2) {
    SWAP(1, 2);
  }
# undef SWAP

  // Degenerate cases where one edge is already flat.
  if (y2 == y3) {
    BresenhamTriangle32(surf, x1, y1, x2, y2, x3, color);
  } else if (y1 == y2) {
    BresenhamTriangle32(surf, x3, y3, x1, y1, x2, color);
  } else {
    // Split into two triangles; need to calculate the projection of v2 onto the
    // line from v1 to v3.
    // Note some loss of precision here, and use of floating point. :/
    const int x = x1 + ((float)(y2 - y1) / (float)(y3 - y1)) * (x3 - x1);
    BresenhamTriangle32(surf, x1, y1, x2, y2, x, color);
    BresenhamTriangle32(surf, x3, y3, x2, y2, x, ~color);
  }
}
