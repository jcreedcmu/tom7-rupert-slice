
#ifndef _CC_LIB_FONTS_TTF_H
#define _CC_LIB_FONTS_TTF_H

#include <array>
#include <cmath>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>
#include <string_view>

#include "stb_truetype.h"
#include "utf8.h"
#include "image.h"

// "Simplified" interface to TrueType fonts (based on stb_truetype),
// with utilities, including an extremely basic export to FontForge
// .SFD files.

// TODO: Maybe better if we decoupled the abstract representation
// from the loading of TTFs, so that we don't need to depend on
// stb_truetype just to export a generated SFD.
//
// TODO: Move more implementations to .cc
// TODO: Only expose normalized coordinates, to reduce confusion.
// TODO: Use int codepoints, not char, in interface.
struct TTF {
  const stbtt_fontinfo *FontInfo() const { return &font; }

  explicit TTF(std::string_view filename);

  // Normalizes an input coordinate (which is usually int16) to be
  // *nominally* in the unit rectangle.
  std::pair<float, float> Norm(float x, float y) const {
    // x coordinate is easy; just scale by the same factor.
    x = norm * x;
    // y is flipped (want +y downward) and offset (want ascent to be 0.0).

    // flip around baseline
    y = -y;
    // baseline (0) becomes ascent
    y += native_ascent;
    y = norm * y;
    return {x, y};
  }

  // amount to advance from one line of text to the next. This would be +1.0 by
  // definition except that we also take into account the "line gap".
  float NormLineHeight() const {
    // Note: Lots of fonts have an incorrect descent (i.e., positive
    // when it should be negative). Maybe it's worth just
    // heuristically taking +abs(native_descent)?
    int native = (native_ascent - native_descent) + native_linegap;
    // ( = native / (native_ascent - native_descent)
    return native * norm;
  }

  // Not cached, so this does a lot more allocation than you probably want.
  ImageA GetChar(int codepoint, int size);

  // Pass DrawPixel(int x, int y, uint8 v) which should do the pixel blending.
  // y position is the top of the font.
  // Prefer BlitStringFloat.
  template<class DP>
  void BlitString(int x, int y, int size_px,
                  std::string_view text, const DP &DrawPixel,
                  bool subpixel = true) const;

  // This is generally preferable to the above, which I should probably
  // deprecate. Always uses subpixel rendering.
  //
  // y position is baseline.
  template<class DP>
  void BlitStringFloat(float x, float y, float size_px,
                       std::string_view text, const DP &DrawPixel,
                       bool kern);


  // Uses SCREEN COORDINATES.
  // Measure the nominal width and height of the string using the same
  // method as above. (This does not mean that all pixels lie within
  // the rectangle.)
  std::pair<int, int>
  MeasureString(std::string_view text, int size_px,
                bool subpixel = true) const;

  enum class PathType {
    LINE,
    // Quadratic bezier (one control point).
    BEZIER,
  };

  // Normalized path.
  struct Path {
    PathType type = PathType::LINE;
    float x = 0.0f, y = 0.0f;
    // For Bezier curves.
    float cx = 0.0f, cy = 0.0f;

    Path(float x, float y) : type(PathType::LINE), x(x), y(y) {}
    Path(float x, float y, float cx, float cy) :
      type(PathType::BEZIER), x(x), y(y), cx(cx), cy(cy) {}
  };

  // There can be clockwise and counterclockwise (thinking about non-
  // self-intersecting) contours. Clockwise increases the winding count
  // by 1; counter-clockwise decreases it.
  //
  // TTF uses the nonzero winding rule: As a ray passes from a point
  // (pixel) to infinity (any direction works), the sum of contours it
  // crosses gives the winding number. If zero, it is "white" (outside
  // the shape). If nonzero, it is "black".
  //
  // Because negative is also nonzero, a shape without holes like an
  // 'm' can be drawn with either winding order (XXX check), although
  // clockwise is normative.
  //
  // This is always closed; the last point in the path is the start
  // point.
  struct Contour {
    // Empty paths are degenerate and should be avoided, but we
    // return 0,0 as the start point for these to avoid undefined
    // behavior.
    float StartX() const {
      if (paths.empty()) return 0.0f;
      else return paths.back().x;
    }
    float StartY() const {
      if (paths.empty()) return 0.0f;
      else return paths.back().y;
    }
    std::vector<Path> paths;
  };

  // Get the contours for a codepoint (e.g. ascii character). Can be
  // empty (space character).
  std::vector<Contour> GetContours(int codepoint) const;

  // For a vector of contours describing a shape, convert any lines
  // into equivalent Bezier curves (with the control point at the
  // midpoint of the line).
  static std::vector<Contour> MakeOnlyBezier(
      const std::vector<Contour> &contours);

  // A contour is just a loop so it can equivalently start at any of
  // its points. This picks the point closest to the given origin.
  // Doesn't change winding orders.
  //
  // Requires that the last path end exactly on the start point. This will
  // be the case for Contours that come from GetContours.
  static std::vector<Contour> NormalizeOrder(
      const std::vector<Contour> &contours,
      float origin_x, float origin_y);

  // Reverse the order of the path (e.g. from clockwise to
  // counter-clockwise).
  static Contour ReverseContour(const Contour &c);

  // Return true if the contour has a clockwise winding order.
  // Only really correct for simple polygons, but a self-intersecting
  // one will still "kind of work". Note that this assumes screen y
  // (smaller y is higher on the screen); the winding order is backwards
  // if math y.
  //
  // Bezier control points do not affect the answer.
  static bool IsClockwise(const Contour &c);

  // Extremely simple "font" representation as a database of characters.
  // Uses the normalized float cordinates so that the font nominally
  // falls in the box (0,0)-(1,1) with y-0 at the top (screen coordinate
  // style), the baseline somewhere in [0,1].
  //
  // The typical use here is for exporting to SFD, which FontForge can
  // make into a TTF (etc.) file. No support for anything other than
  // line and quad contours.
  struct Char {
    std::vector<Contour> contours;
    float width = 1.0f;
  };
  struct Font {
    // Key is the unicode codepoint.
    std::map<int, Char> chars;
    float baseline = 0.75f;
    float linegap = 0.0f;
    // Additional scale factor with baseline origin. Typical use is to
    // remove unwanted space above the character. Affects widths but
    // not baseline, ascent, or descent.
    float extra_scale = 1.0f;

    // Hint about whether anti-aliasing is appropriate.
    bool antialias = true;

    // If positive, a hint about the height of the pixel grid. This
    // improves coordinates for low-resolution bitmap fonts, and
    // should be zero for normal vector fonts or high res (e.g. 100px)
    // bitmaps.
    //
    // For low-res bitmaps, this should be the total number of pixels
    // from top to bottom including the descent. This is used to
    // generate a units-per-em value that pixel coordinates will
    // evenly divide.
    int bitmap_grid_height = 0;

    // Freeform line of text, which seems to be the standard
    // place to put something like a URL.
    std::string copyright = "Generated by cc-lib ttf.cc";

    // 4-character vendor code. 'Frog' is reserved for Tom 7 plz (used
    // by my "Divide By Zero Fonts" since the 1990s!).
    std::array<char, 4> vendor = {'F', 'r', 'o', 'g'};

    // Not to be confused with SDF! This is the text file format for
    // FontForge. Extremely simple subset with many fields hackily
    // hard-coded. Name is the font name (spaces are stripped to
    // produce the internal name; avoid weird characters) and
    // copyright is a free-form line of text which seems to be the
    // standard place to put something like a URL, too.
    std::string ToSFD(const std::string &name) const;
  };

  // In-place update control point coordinates (only -- metrics are
  // not affected) by calling f(x,y) for each in the font.
  template<class F>
  static void MapCoords(F f, Font *font);
  template<class F>
  static void MapCoords(F f, Char *ch);


  // c2 may be 0 for no kerning.
  float NormKernAdvance(char c1, char c2) {
    int advance = 0;
    stbtt_GetCodepointHMetrics(&font, c1, &advance, nullptr);
    if (c2 != 0) {
      advance += stbtt_GetCodepointKernAdvance(&font, c1, c2);
    }
    return Norm(advance, 1.0f).first;
  }

  // Returns (minx, miny, maxx, maxy) in normalized coordinates.
  std::tuple<float, float, float, float>
  BoundingBox() const {
    int x0, y0, x1, y1;
    stbtt_GetFontBoundingBox(&font, &x0, &y0, &x1, &y1);

    printf("x0: %d, y0: %d, x1: %d, y1: %d\n",
           x0, y0, x1, y1);

    // Note maxness of y coordinate is swapped since we use
    // a flipped coordinate system.
    const auto &[xmin, ymax] = Norm(x0, y0);
    const auto &[xmax, ymin] = Norm(x1, y1);

    printf("x0: %.2f, y0: %.2f, x1: %.2f, y1: %.2f\n",
           xmin, ymin, xmax, ymax);

    return std::make_tuple(xmin, ymin, xmax, ymax);
  }

  // Get SDF for the character. This is tuned for ML applications, not
  // graphics.
  //
  //         sdf_size
  //  +---------------------+
  //  |      |              |
  //  |     pad top         |
  //  |      |              | s
  //  |     +---------+     | d
  //  |     |         :     | f
  //  |     |         : h   | _
  //  |-pad-|         : t   | s
  //  | left|         :     | i
  //  |     +---------+     | z
  //  |   origin   |        | e
  //  |           pad bot   |
  //  |            |        |
  //  +---------------------+
  //
  // The output image will be sdf_size * sdf_size if successful. The
  // character is rendered at the nominal pixel height ht, which
  // is sdf_size - pad * 2.
  // The character is registered with its origin at (pad, pad + ht);
  // this means that two SDFs drawn on top of one another have the
  // character in the correct relative position. Padding allows the
  // character to extend outside the box (this is normal: a
  // character like w is wider than its nominal height; a character
  // like j typically descends below the origin and to its left; even
  // o typically drops slightly below the baseline and left edge.)
  // However, if pixels that are inside the character (sdf value >=
  // onedge_value) fall outside the bitmap, nullopt is returned.
  // Internally we render the SDF with lots of padding so that the
  // entire output should be filled with the actual distance function,
  // but of course it must be cropped at the output's edges.
  //
  // onedge_value and falloff_per_pixel are the same as in the
  // stb_truetype lib. A value ~200 seems good for onedge_value
  // because exterior distances are typically much higher than
  // interior (especially with padding).
  // Unclear what a good falloff_per_pixel is, but it probably makes
  // sense to use the full dynamic range. The longest edge is from the
  // center to a corner (sqrt(2) * sdf_size / 2), which would be a
  // falloff of (sqrt(2) * sdf_size / 2) / 200 = 0.0035 * sdf_size.
  // Alternatively, we could take the center box (ht * ht) to be
  // the nominal character's edge (or at least the closest we "normally"
  // get to the sdf edge), and expect to get to zero along the
  // shortest edge; this distance is "pad". So we'd have 200 / pad.
  // So, suggestion is 0.0035 * sdf_size <= falloff <= 200 / pad
  // for an onedge_value of 200.
  //
  // Padding can actually be specified separately for top, bottom, and left.
  // (Pad right would not actually do anything.)
  // left = bottom and 1:3 ratio of top:bottom seems to be generally efficient.
  std::optional<ImageA> GetSDF(int codepoint,
                               int sdf_size,
                               int pad_top, int pad_bot, int pad_left,
                               uint8_t onedge_value,
                               float falloff_per_pixel) const;

  float Baseline() const { return baseline; }

private:

  // font uses "y positive up" coordinates.
  // ascent is the coordinate above the baseline (typically positive) and
  // descent is the coordinate below the baseline (typically negative).
  int native_ascent = 0, native_descent = 0;
  int native_linegap = 0;
  float norm = 0.0;
  float baseline = 0.0f;
  // By definition: ascent = 0.0, descent = 0.0

  struct NativePath {
    PathType type = PathType::LINE;
    int x = 0, y = 0;
    // For Bezier curves.
    int cx = 0, cy = 0;

    NativePath(int x, int y) : type(PathType::LINE), x(x), y(y) {}
    NativePath(int x, int y, int cx, int cy) :
      type(PathType::BEZIER), x(x), y(y), cx(cx), cy(cy) {}
  };

  // Note: Source integer coordinates are int16.
  struct NativeContour {
    // int startx = 0, starty = 0;
    int StartX() const {
      if (paths.empty()) return 0;
      else return paths.back().x;
    }
    int StartY() const {
      if (paths.empty()) return 0;
      else return paths.back().y;
    }
    std::vector<NativePath> paths;
  };

  std::vector<NativeContour> GetNativeContours(int codepoint) const;

  std::vector<uint8_t> ttf_bytes;
  stbtt_fontinfo font;

  TTF(const TTF &other) = delete;
  TTF &operator =(const TTF &other) = delete;
};


// Template implementations follow.

template<class DP>
void TTF::BlitString(int x, int y, int size_px,
                     std::string_view text, const DP &DrawPixel,
                     bool subpixel) const {
  const float scale = stbtt_ScaleForPixelHeight(&font, size_px);

  const int baseline = [&]() {
      int ascent = 0;
      stbtt_GetFontVMetrics(&font, &ascent, 0, 0);
      return (int) (ascent * scale);
    }();

  const int ypos = y + baseline;
  // Should stay integral if subpixel is false.
  float xpos = x;
  for (int idx = 0; idx < (int)text.size(); idx++) {

    int advance = 0, left_side_bearing = 0;
    stbtt_GetCodepointHMetrics(
        &font, text[idx], &advance, &left_side_bearing);

    int bitmap_w = 0, bitmap_h = 0;
    int xoff = 0, yoff = 0;
    uint8_t *bitmap = nullptr;
    if (subpixel) {
      const float x_shift = xpos - (float) floor(xpos);
      constexpr float y_shift = 0.0f;
      bitmap = stbtt_GetCodepointBitmapSubpixel(&font, scale, scale,
                                                x_shift, y_shift,
                                                text[idx],
                                                &bitmap_w, &bitmap_h,
                                                &xoff, &yoff);
    } else {
      bitmap = stbtt_GetCodepointBitmap(&font, scale, scale,
                                        text[idx],
                                        &bitmap_w, &bitmap_h,
                                        &xoff, &yoff);
    }
    if (bitmap != nullptr) {
      for (int yy = 0; yy < bitmap_h; yy++) {
        for (int xx = 0; xx < bitmap_w; xx++) {
          DrawPixel(xpos + xx + xoff, ypos + yy + yoff,
                    bitmap[yy * bitmap_w + xx]);
        }
      }
      stbtt_FreeBitmap(bitmap, nullptr);
    }

    xpos += advance * scale;
    if (text[idx + 1] != '\0') {
      xpos += scale *
        stbtt_GetCodepointKernAdvance(&font, text[idx], text[idx + 1]);
    }

    if (!subpixel) {
      // Or floor?
      xpos = roundf(xpos);
    }
  }
}

template<class DP>
void TTF::BlitStringFloat(float x, float y, float size_px,
                          std::string_view text, const DP &DrawPixel,
                          bool kern) {
  const float scale = stbtt_ScaleForPixelHeight(&font, size_px);

  // y position stays the same throughout.
  const float ypos = y;
  const int y_int = floor(ypos);
  const float y_shift = ypos - y_int;

  // Should stay integral if subpixel is false.
  float xpos = x;

  std::vector<uint32_t> codepoints = UTF8::Codepoints(text);
  for (int idx = 0; idx < (int)codepoints.size(); idx++) {

    int advance = 0, left_side_bearing = 0;
    stbtt_GetCodepointHMetrics(
        &font, codepoints[idx], &advance, &left_side_bearing);

    int bitmap_w = 0, bitmap_h = 0;
    // We always render the glyph's bitmap as though at 0,0.
    int xoff = 0, yoff = 0;
    uint8_t *bitmap = nullptr;

    const int x_int = floor(xpos);
    const float x_shift = xpos - x_int;
    bitmap = stbtt_GetCodepointBitmapSubpixel(&font, scale, scale,
                                              x_shift, y_shift,
                                              codepoints[idx],
                                              &bitmap_w, &bitmap_h,
                                              &xoff, &yoff);

    if (bitmap != nullptr) {
      for (int yy = 0; yy < bitmap_h; yy++) {
        for (int xx = 0; xx < bitmap_w; xx++) {
          DrawPixel(xoff + x_int + xx, yoff + y_int + yy,
                    bitmap[yy * bitmap_w + xx]);
        }
      }
      stbtt_FreeBitmap(bitmap, nullptr);
    }

    if (false) {
      for (int yy = 0; yy < bitmap_h; yy++) {
        for (int xx = 0; xx < bitmap_w; xx++) {
          DrawPixel(x_int + xx, y_int - bitmap_h + yy, 0x44);
        }
      }
    }

    xpos += advance * scale;
    if (kern && idx + 1 < (int)codepoints.size()) {
      xpos += scale *
        stbtt_GetCodepointKernAdvance(
            &font, codepoints[idx], codepoints[idx + 1]);
    }
  }
}


template<class F>
void TTF::MapCoords(F f, TTF::Char *ch) {
  for (Contour &cc : ch->contours) {
    for (Path &p : cc.paths) {
      std::tie(p.x, p.y) = f(p.x, p.y);
      if (p.type == PathType::BEZIER) {
        std::tie(p.cx, p.cy) = f(p.cx, p.cy);
      }
    }
  }
}

template<class F>
void TTF::MapCoords(F f, TTF::Font *font) {
  for (auto &[c, ch] : font->chars) {
    MapCoords(f, &ch);
  }
}


#endif
