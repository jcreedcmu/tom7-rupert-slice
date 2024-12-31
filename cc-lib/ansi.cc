
#include "ansi.h"

#include <algorithm>
#include <cstdlib>
#include <string>
#include <cmath>
#include <cstdint>
#include <tuple>
#include <vector>
#include <utility>
#include <string_view>
#include <span>

#ifdef __MINGW32__
#include <windows.h>
#undef ARRAYSIZE
#endif

#include "utf8.h"
// TODO: Would be good to avoid this dependency too.
// We could use std::format in c++20, for example.
#include "base/stringprintf.h"

static constexpr bool VERBOSE = false;

using std::string;

// Normal non-bright colors, as 00RRGGBB.
static std::array<uint32_t, 8> DARK_RGB = {
  0x000000,  // Black
  0xd42c3a,  // Red
  0x1ca800,  // Green
  0xc0a000,  // Yellow
  0x005dff,  // Blue
  0xb148c6,  // Magenta
  0x00a89a,  // Cyan
  0xBFBFBF,  // White
};

static std::array<uint32_t, 8> BRIGHT_RGB = {
  0x606060,  // Black
  0xff7676,  // Red
  0x99f299,  // Green
  0xf2f200,  // Yellow
  0x7d97ff,  // Blue
  0xff70ff,  // Magenta
  0x00f0f0,  // Cyan
  0xffffff,  // White
};

void ANSI::Init() {
  #ifdef __MINGW32__
  // Turn on ANSI support in Windows 10+. (Otherwise, use ANSICON etc.)
  // https://docs.microsoft.com/en-us/windows/console/setconsolemode
  //
  // TODO: This works but seems to subsequently break control keys
  // and stuff like that in cygwin bash?
  HANDLE hStdOut = GetStdHandle(STD_OUTPUT_HANDLE);
  // mingw headers may not know about this new flag
  static constexpr int kVirtualTerminalProcessing = 0x0004;
  DWORD old_mode = 0;
  GetConsoleMode(hStdOut, &old_mode);
  SetConsoleMode(hStdOut, old_mode | kVirtualTerminalProcessing);

  // XXX Not related to ANSI; enables utf-8 output.
  // Maybe we should separate this out? Someone might want to use
  // high-ascii output?
  SetConsoleOutputCP( CP_UTF8 );
  #endif
}

// TODO: It would be better if we had a way of stripping the
// terminal codes if they are not supported?
void CPrintf(const char* format, ...) {
  // Do formatting.
  va_list ap;
  va_start(ap, format);
  std::string result;
  StringAppendV(&result, format, ap);
  va_end(ap);

  #ifdef __MINGW32__
  DWORD n = 0;
  WriteConsole(GetStdHandle(STD_OUTPUT_HANDLE),
               result.c_str(),
               result.size(),
               &n,
               nullptr);
  #else
  printf("%s", result.c_str());
  #endif
}

// and 0xRRGGBBAA
inline static constexpr std::tuple<uint8, uint8, uint8, uint8>
Unpack32(uint32 color) {
  return {(uint8)((color >> 24) & 255),
          (uint8)((color >> 16) & 255),
          (uint8)((color >> 8) & 255),
          (uint8)(color & 255)};
}

inline static constexpr uint32 Pack32(uint8 r, uint8 g, uint8 b, uint8 a) {
  return
    ((uint32)r << 24) | ((uint32)g << 16) | ((uint32)b << 8) | (uint32)a;
}

std::string ANSI::ForegroundRGB(uint8_t r, uint8_t g, uint8_t b) {
  char escape[24] = {};
  snprintf(escape, 24, "\x1B[38;2;%d;%d;%dm", r, g, b);
  return escape;
}

std::string ANSI::BackgroundRGB(uint8_t r, uint8_t g, uint8_t b) {
  // Max size is 12.
  char escape[24] = {};
  snprintf(escape, 24, "\x1B[48;2;%d;%d;%dm", r, g, b);
  return escape;
}

std::string ANSI::ForegroundRGB32(uint32_t rgba) {
  const auto [r, g, b, a_] = Unpack32(rgba);
  return ForegroundRGB(r, g, b);
}

std::string ANSI::BackgroundRGB32(uint32_t rgba) {
  const auto [r, g, b, a_] = Unpack32(rgba);
  return BackgroundRGB(r, g, b);
}

std::string ANSI::Time(double seconds) {
  char result[64] = {};
  if (seconds < 0.001) {
    snprintf(result, 64, AYELLOW("%.3f") "μs", seconds * 1000000.0);
  } else if (seconds < 1.0) {
    snprintf(result, 64, AYELLOW("%.2f") "ms", seconds * 1000.0);
  } else if (seconds < 60.0) {
    snprintf(result, 64, AYELLOW("%.3f") "s", seconds);
  } else if (seconds < 60.0 * 60.0) {
    int sec = std::round(seconds);
    int omin = sec / 60;
    int osec = sec % 60;
    snprintf(result, 64, AYELLOW("%d") "m" AYELLOW("%02d") "s",
             omin, osec);
  } else {
    int sec = std::round(seconds);
    int ohour = sec / 3600;
    sec -= ohour * 3600;
    int omin = sec / 60;
    int osec = sec % 60;
    if (ohour <= 24) {
      snprintf(result, 64, AYELLOW("%d") "h"
               AYELLOW("%d") "m"
               AYELLOW("%02d") "s",
               ohour, omin, osec);
    } else {
      int odays = ohour / 24;
      ohour %= 24;
      snprintf(result, 64, AYELLOW("%d") "d"
               AYELLOW("%d") "h"
               AYELLOW("%d") "m",
               odays, ohour, omin);
    }
  }
  return (string)result;
}

std::string ANSI::StripCodes(const std::string &s) {
  std::string out;
  out.reserve(s.size());
  bool in_escape = false;
  for (int i = 0; i < (int)s.size(); i++) {
    if (in_escape) {
      if (s[i] >= '@' && s[i] <= '~') {
        in_escape = false;
      }
    } else {
      // OK to access one past the end of the string.
      if (s[i] == '\x1B' && s[i + 1] == '[') {
        in_escape = true;
        i++;
      } else {
        out += s[i];
      }
    }
  }
  return out;
}

int ANSI::StringWidth(const std::string &s) {
  // PERF could do this without copying.
  return StripCodes(s).size();
}


std::string ANSI::ProgressBar(uint64_t numer, uint64_t denom,
                              const std::string &operation,
                              double seconds,
                              ProgressBarOptions options) {
  double frac = numer / (double)denom;

  double spe = numer > 0 ? seconds / numer : 1.0;
  double remaining_sec = (denom - numer) * spe;
  string eta = Time(remaining_sec);
  int eta_len = StringWidth(eta);

  int bar_width = options.full_width - 2 - 1 - eta_len;
  // Number of characters that get background color.
  int filled_width = std::clamp((int)std::round(bar_width * frac),
                                0, bar_width);
  std::string bar_text;
  if (options.include_frac) {
    StringAppendF(&bar_text, "%llu / %llu", numer, denom);
  }
  if (options.include_percent) {
    if (!bar_text.empty()) bar_text += "  ";
    StringAppendF(&bar_text, "(%.1f%%)", frac * 100.0);
  }

  if (!bar_text.empty()) bar_text.push_back(' ');
  bar_text += operation;

  // Strip codes from bar_text and generate the rasterized foreground
  // color. We discard the background color, since we use it to draw
  // the progress bar.
  std::string bar_text_plain;
  std::vector<uint32_t> fgraster, bgraster_;
  std::tie(bar_text_plain, fgraster, bgraster_) =
    Decompose(bar_text, options.fg);

  // Generate the background raster.
  std::vector<uint32_t> bgraster(bar_width, 0);
  for (int i = 0; i < bar_width; i++) {
    bgraster[i] = i < filled_width ? options.bar_filled : options.bar_empty;
  }

  // could do "..."
  if ((int)bar_text_plain.size() > bar_width) {
    bar_text_plain = UTF8::Truncate(bar_text_plain, bar_width);
  }
  bar_text_plain.reserve(bar_width);
  while ((int)bar_text_plain.size() < bar_width)
    bar_text_plain.push_back(' ');

  string colored_bar = Composite(bar_text_plain, fgraster, bgraster);
  string out = AWHITE("[") + colored_bar + AWHITE("]") " " + eta;

  return out;
}

// From image.cc; see explanation there.
static inline uint32_t CompositeRGBA(uint32_t fg, uint32_t bg) {
  using word = uint16_t;
  const auto &[r, g, b, a] = Unpack32(fg);
  const auto &[old_r, old_g, old_b, old_a_] = Unpack32(bg);
  // so a + oma = 255.
  const word oma = 0xFF - a;
  const word rr = (((word)r * (word)a) + (old_r * oma)) / 0xFF;
  const word gg = (((word)g * (word)a) + (old_g * oma)) / 0xFF;
  const word bb = (((word)b * (word)a) + (old_b * oma)) / 0xFF;
  // Note that the components cannot be > 0xFF.
  if (rr > 0xFF) __builtin_unreachable();
  if (gg > 0xFF) __builtin_unreachable();
  if (bb > 0xFF) __builtin_unreachable();

  return Pack32(rr, gg, bb, 0xFF);
}

std::vector<uint32_t> ANSI::Rasterize(
    const std::vector<std::pair<uint32_t, int>> &spans,
    int width) {
  std::vector<uint32_t> flat;
  flat.reserve(width);
  uint32_t last = 0;
  for (const auto &[c, w] : spans) {
    for (int i = 0; i < w; i++) flat.push_back(c);
    last = c;
  }

  if ((int)flat.size() > width) {
    flat.resize(width);
  } else {
    while ((int)flat.size() < width) flat.push_back(last);
  }
  return flat;
}

std::string ANSI::Composite(
    const std::string &text_raw,
    const std::vector<uint32_t> &fgcolors,
    const std::vector<uint32_t> &bgcolors) {

  const int width = std::max(fgcolors.size(), bgcolors.size());

  std::string text = StripCodes(text_raw);

  std::vector<uint32_t> codepoints = UTF8::Codepoints(text);

  if (width <= 0) return "";
  if ((int)codepoints.size() > width) codepoints.resize(width);
  while ((int)codepoints.size() < width) codepoints.push_back(' ');

  auto GetExtend = [](const std::vector<uint32_t> &v, int idx) -> uint32_t {
      if (idx < (int)v.size()) return v[idx];
      if (v.empty()) return 0x000000FF;
      return v.back();
    };

  // Now generate output string. Here we generate a color code whenever
  // the foreground or background actually changes.

  // This string will generally be longer than width because of the
  // ansi codes.
  std::string out;
  uint32_t last_fg = 0, last_bg = 0;
  for (int i = 0; i < width; i++) {
    uint32_t bgcolor = GetExtend(bgcolors, i);
    uint32_t fgcolor = CompositeRGBA(GetExtend(fgcolors, i), bgcolor);

    if (i == 0 || bgcolor != last_bg) {
      const auto &[r, g, b, a_] = Unpack32(bgcolor);
      StringAppendF(&out, "\x1B[48;2;%d;%d;%dm", r, g, b);
      last_bg = bgcolor;
    }

    if (i == 0 || fgcolor != last_fg) {
      const auto &[r, g, b, a_] = Unpack32(fgcolor);
      StringAppendF(&out, "\x1B[38;2;%d;%d;%dm", r, g, b);
      last_fg = fgcolor;
    }

    // And always add the text codepoint.
    out += UTF8::Encode(codepoints[i]);
  }

  return out + ANSI_RESET;
}

static bool FindAndRemove(int x, std::vector<int> *left) {
  return std::erase_if(*left, [x](int y) { return x == y; }) != 0;
}

std::tuple<std::string,
           std::vector<uint32_t>,
           std::vector<uint32_t>>
ANSI::Decompose(const std::string &text_with_codes,
                uint32_t default_fg,
                uint32_t default_bg) {
  if (VERBOSE) printf("[%s]\n", text_with_codes.c_str());

  std::vector<uint32_t> codepoints = UTF8::Codepoints(text_with_codes);

  const int length = (int)codepoints.size();
  std::string out;
  std::vector<uint32_t> fgs, bgs;
  // n.b. Might not be this long because of color codes.
  out.reserve(length);
  fgs.reserve(length);
  bgs.reserve(length);

  uint32_t fg = default_fg;
  uint32_t bg = default_bg;

  std::span<uint32_t> s(codepoints.data(), codepoints.size());

  while (!s.empty()) {
    // Parse control codes. We just accept the grammar
    // "\x1b["
    //   followed by ([0-9]*;)*
    //   followed by [a-z]
    if (s.size() >= 2 && s[0] == '\x1b' && s[1] == '[') {
      // Skip the escape sequence.
      s = s.subspan(2);

      const auto &[params, cmd] = [&s]() ->
        std::pair<std::vector<int>, uint32_t> {
          std::vector<int> params;
          int value = 0;

          while (!s.empty()) {
            uint32_t c = *s.begin();
            // Always consume; we want to consume the
            // whole code including the command.
            s = s.subspan(1);
            if (c >= '0' && c <= '9') {
              value *= 10;
              value += (c - '0');
            } else if (c == ';') {
              params.push_back(value);
              value = 0;
            } else if (c >= 0x40 && c <= 0x7E) {
              params.push_back(value);
              value = 0;
              return std::make_pair(std::move(params), c);
            } else {
              // "intermediate bytes" unsupported / ignored
            }
          }

          return std::make_pair(std::vector<int>(), 'x');
      }();

      // The only command we support is 'm'.
      if (cmd == 'm') {
        // ANSI_RED = \x1B[1;31;40m

        // 1 means bold. 30-37 are foreground, 40-47 are background.
        // 0 means reset.
        std::vector<int> pleft = params;
        bool is_reset = FindAndRemove(0, &pleft);
        // bool is_bright = FindAndRemove(1, &pleft);
        bool is_bright = false;

        if (is_reset) {
          fg = default_fg;
          bg = default_bg;
        }

        for (int pidx = 0; pidx < (int)pleft.size(); pidx++) {
          const int p = pleft[pidx];
          if (p == 1) {
            is_bright = true;
          } else if (p >= 30 && p <= 37) {
            if (is_bright) {
              fg = (BRIGHT_RGB[p - 30] << 8) | 0xFF;
            } else {
              fg = (DARK_RGB[p - 30] << 8) | 0xFF;
            }
          } else if (p >= 40 && p <= 47) {
            // Bright only affects foreground, not background.
            bg = (DARK_RGB[p - 40] << 8) | 0xFF;
          } else if (p >= 90 && p <= 97) {
            // unusual: direct access to bright foreground.
            fg = (BRIGHT_RGB[p - 90] << 8) | 0xFF;
          } else if (p >= 100 && p <= 107) {
            // unusual: direct access to bright background.
            bg = (BRIGHT_RGB[p - 100] << 8) | 0xFF;
          } else if (p == 39) {
            fg = default_fg;
          } else if (p == 49) {
            bg = default_bg;
          } else if (p == 38 || p == 48) {
            // Set RGB colors directly.

            if (VERBOSE) {
              printf("Pidx: %d\n", pidx);
              for (int i = 0; i < (int)pleft.size(); i++) {
                printf("Pleft[%d] = %d\n", i, pleft[i]);
              }
            }

            if (pidx + 4 >= (int)pleft.size() || pleft[pidx + 1] != 2) {
              if (VERBOSE) printf("Expected 2;R;G;B.\n");
              break;
            }

            const uint8_t r = pleft[pidx + 2];
            const uint8_t g = pleft[pidx + 3];
            const uint8_t b = pleft[pidx + 4];
            const uint32_t rgb = ((uint32_t)r << 24) |
              ((uint32_t)g << 16) |
              ((uint32_t)b << 8) |
              0xFF;

            if (VERBOSE) {
              printf("GOT: %04x\n", rgb);
            }

            if (p == 38) fg = rgb;
            else bg = rgb;

            pidx += 5;

          } else {
            if (VERBOSE) printf("Unknown param %d\n", p);
          }


        }
      } else {
        if (VERBOSE) printf("Unknown command %c\n", cmd);
      }


    } else {
      // Encode back as UTF-8.
      out += UTF8::Encode(s[0]);
      fgs.push_back(fg);
      bgs.push_back(bg);
      s = s.subspan(1);
    }
  }

  return std::make_tuple(std::move(out), std::move(fgs), std::move(bgs));
}
