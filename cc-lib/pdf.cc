// PDF output, based on Andre Renaud's public domain PDFGen:
//   https://github.com/AndreRenaud/PDFGen/tree/master
//
// I made a large number of local edits. Some notable ones:
//   - "fixed" some printf format parameter warnings
//   - Ported to C++.
//   - Fix bug with text justification: Denominator should be
//      len - 1, not len - 2.
//   - Removed many things I didn't want
//   - Added compression
//   - Added QR codes
//   - Added support for embedded TrueType fonts

// PERF: Output will have a lot of 200.00000000 stuff; use smarter
// float to text routine.

/* TODO
Validating file "minimal.pdf" for conformance level pdf1.3
The key Flags is required but missing.
The key FontBBox is required but missing.
The key ItalicAngle is required but missing.
The key Ascent is required but missing.
The key Descent is required but missing.
The key CapHeight is required but missing.
The key StemV is required but missing.
The value of the key Flags is 0 but must be either symbolic or non-symbolic.
The document does not conform to the requested standard.
The document doesn't conform to the PDF reference (missing required entries, wrong value types, etc.).
The document contains fonts without embedded font programs or encoding information (CMAPs).
The document does not conform to the PDF 1.3 standard.
Done.
*/

#include "pdf.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <ctype.h>
#include <errno.h>
#include <format>
#include <initializer_list>
#include <inttypes.h>
#include <math.h>
#include <numbers>
#include <optional>
#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <string_view>
#include <sys/stat.h>
#include <time.h>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "base/logging.h"
#include "base/stringprintf.h"
#include "hashing.h"
#include "image.h"
#include "map-util.h"
#include "qr-code.h"
#include "stb_truetype.h"
#include "utf8.h"
#include "zip.h"
#include "util.h"

// XXX just for debugging output
#include "ansi.h"

#define PDF_RGB_R(c) (((c) >> 16) & 0xff)
#define PDF_RGB_G(c) (((c) >> 8) & 0xff)
#define PDF_RGB_B(c) (((c) >> 0) & 0xff)


#define PDF_RGB_R_FLOAT(c) (float)(PDF_RGB_R(c) / 255.0f)
#define PDF_RGB_G_FLOAT(c) (float)(PDF_RGB_G(c) / 255.0f)
#define PDF_RGB_B_FLOAT(c) (float)(PDF_RGB_B(c) / 255.0f)
#define PDF_COLOR_FLOAT(a) (float)( (a) / 255.0f )
#define PDF_IS_TRANSPARENT(c) (((c) >> 24) == 0xff)

#if defined(_MSC_VER)
#define inline __inline
#define snprintf _snprintf
#define fileno _fileno
#define fstat _fstat
#ifdef stat
#undef stat
#endif
#define stat _stat
#define SKIP_ATTRIBUTE
#endif

// Enable for copious debugging information.
static constexpr bool VERBOSE = false;

// Special signatures for PNG chunks
static const char png_chunk_header[] = "IHDR";
static const char png_chunk_palette[] = "PLTE";
static const char png_chunk_data[] = "IDAT";
static const char png_chunk_end[] = "IEND";

static const uint16_t *find_font_widths(PDF::BuiltInFont font);

using FontEncoding = PDF::FontEncoding;

static std::string Float(float f) {
  std::string s = std::format("{:.6f}", f);
  for (;;) {
    if (s.empty()) return "0";
    // Trailing zeroes do nothing
    if (s.back() == '0') {
      s.resize(s.size() - 1);
    } else if (s.back() == '.') {
      // And if we had an integer, we don't need the period either.
      s.resize(s.size() - 1);
      // But now we are done.
      if (s.empty()) return "0";
      return s;
    } else {
      // Other digits? Done.
      return s;
    }
  }
}

// Return the Latin-1 (ISO 8859-1) encoded character if it exists
// (always one byte, and in fact always the same as the input), or zero
// otherwise.
static uint8_t UnicodeToLatin1(uint32_t codepoint) {
  // Printable ASCII.
  if (codepoint >= 0x20 && codepoint < 0x7e) return (uint8_t)codepoint;
  if (codepoint >= 0xA0 && codepoint <= 0xFF) return (uint8_t)codepoint;
  return 0;
}

namespace {

// Prefer this one, which is already converted into native byte order.
struct PNGChunk {
  uint32_t length;
  char type[4];
};

// As defined by https://www.w3.org/TR/2003/REC-PNG-20031110/#6Colour-values
enum PNGColorType : uint8_t {
  // Greyscale
  PNG_COLOR_GREYSCALE = 0,
  // Truecolor
  PNG_COLOR_RGB = 2,
  // Indexed-color
  PNG_COLOR_INDEXED = 3,
  // Greyscale with alpha
  PNG_COLOR_GREYSCALE_A = 4,
  // Truecolor with alpha
  PNG_COLOR_RGBA = 6,

  PNG_COLOR_INVALID = 255
};

// Header of a PNG file.
// Prefer this one, already converted into native byte order.
struct PNGHeader {
  // Dimensions in pixels.
  uint32_t width;
  uint32_t height;
  uint8_t bit_depth;
  PNGColorType color_type;
  uint8_t deflate;
  uint8_t filtering;
  uint8_t interlace;
};

// Data should point at the beginning of the PNG file.
// Assumes that it is at least the minimum size. Doesn't
// check anything about the format.
static constexpr size_t PNG_FILE_MIN_SIZE = 67;

// This is not exact. But we just need it to rule out invalid headers.
static constexpr size_t JPG_FILE_MIN_SIZE = 100;
struct JPGHeader {
  uint32_t width = 0;
  uint32_t height = 0;
  int ncolors = 0;
};

// Simple data container to store a single 24 Bit RGB value, used for
// processing PNG images
struct rgb_value {
  uint8_t red;
  uint8_t blue;
  uint8_t green;
};

}  // namespace

// Read 4 bytes in big-endian; no checks.
inline static uint32_t Read32(const uint8_t *data) {
  return ((uint32_t)data[0] << 24) |
    ((uint32_t)data[1] << 16) |
    ((uint32_t)data[2] << 8) |
    (uint32_t)data[3];
}

inline static PNGHeader ReadPngHeader(const uint8_t *data) {
  const uint8_t *header = data + 16;

  PNGHeader ret;
  ret.width = Read32(header + 0);
  ret.height = Read32(header + 4);
  ret.bit_depth = header[8];
  ret.color_type = (PNGColorType)header[9];
  ret.deflate = header[10];
  ret.filtering = header[11];
  ret.interlace = header[12];
  return ret;
}

// data should point at the next chunk.
inline static PNGChunk ReadPngChunk(const uint8_t *data) {
  PNGChunk ret;
  ret.length = ((uint32_t)data[0] << 24) |
    ((uint32_t)data[1] << 16) |
    ((uint32_t)data[2] << 8) |
    (uint32_t)data[3];
  ret.type[0] = data[4];
  ret.type[1] = data[5];
  ret.type[2] = data[6];
  ret.type[3] = data[7];
  return ret;
}


const char *PDF::ObjTypeName(ObjType t) {
  switch (t) {
  case OBJ_none: return "none";
  case OBJ_info: return "info";
  case OBJ_stream: return "stream";
  case OBJ_builtin_font: return "builtin_font";
  case OBJ_font8: return "font8";
  case OBJ_font0: return "font0";
  case OBJ_fontcid: return "fontcid";
  case OBJ_page: return "page";
  case OBJ_bookmark: return "bookmark";
  case OBJ_outline: return "outline";
  case OBJ_catalog: return "catalog";
  case OBJ_pages: return "pages";
  case OBJ_image: return "image";
  case OBJ_link: return "link";
  case OBJ_widths: return "widths";
  default: break;
  }
  return "???";
}

int PDF::SetErr(int errval, std::string s) {
  this->errval = errval;
  this->error_message = std::move(s);
  return errval;
}

std::string PDF::GetErr() const {
  return error_message;
}

void PDF::ClearErr() {
  error_message[0] = '\0';
  errval = 0;
}

int PDF::GetErrCode() const {
  return errval;
}

PDF::Object *PDF::GetObject(int index) {
  return objects[index];
}

void PDF::AppendObject(Object *obj) {
  int index = (int)objects.size();
  objects.push_back(obj);

  CHECK(index >= 0);

  obj->index = index;

  if (last_objects[obj->type]) {
    obj->prev = last_objects[obj->type];
    last_objects[obj->type]->next = obj;
  }
  last_objects[obj->type] = obj;

  if (!first_objects[obj->type])
    first_objects[obj->type] = obj;
}

void PDF::DestroyObject(Object *object) {
  delete object;
}

PDF::Object *PDF::AddObjectInternal(Object *obj) {
  if (VERBOSE) {
    printf("Add object internal (%p, type %s)\n", obj, ObjTypeName(obj->type));
  }
  CHECK(obj != nullptr);
  AppendObject(obj);
  return obj;
}

void PDF::SetInfo(const PDF::Info &info) {
  InfoObj *obj = (InfoObj*)FindFirstObject(OBJ_info);
  CHECK(obj != nullptr) << "This is created by the "
    "constructor!";

  obj->info = info;
}

const PDF::Info &PDF::GetInfo() const {
  const InfoObj *obj = (const InfoObj*)FindFirstObject(OBJ_info);
  CHECK(obj != nullptr) << "This is created by the "
    "constructor!";
  return obj->info;
}

PDF::PDF(float width, float height, Options options) :
  document_width(width), document_height(height), options(options) {

  /* We don't want to use ID 0 */
  (void)AddObject(new NoneObj);

  // Defaults for the info object.
  InfoObj *obj = AddObject(new InfoObj);
  CHECK(obj != nullptr);

  obj->info.creator = "pdf.cc";
  obj->info.producer = "pdf.cc";
  obj->info.title = "Untitled";
  obj->info.date = Util::FormatTime("%Y%m%d%H%M%SZ", time(nullptr));

  CHECK(AddObject(new PagesObj) != nullptr);
  CHECK(AddObject(new CatalogObj) != nullptr);
  SetFont(TIMES_ROMAN);
}

float PDF::Width() const { return document_width; }
float PDF::Height() const { return document_height; }

PDF::~PDF() {
  for (Object *obj : objects)
    DestroyObject(obj);
  objects.clear();

  for (const auto &[_, f] : embedded_fonts)
    delete f;
  embedded_fonts.clear();

  for (const auto &[_, f] : builtin_fonts)
    delete f;
  builtin_fonts.clear();
}

PDF::Object *PDF::FindFirstObject(int type) {
  return first_objects[type];
}

PDF::Object *PDF::FindLastObject(int type) {
  return last_objects[type];
}

const PDF::Object *PDF::FindFirstObject(int type) const {
  return first_objects[type];
}

const PDF::Object *PDF::FindLastObject(int type) const {
  return last_objects[type];
}

// Or returns nullptr if the font has not been loaded.
const PDF::Font *PDF::GetFontByName(const std::string &font_name) const {
  auto it = embedded_fonts.find(font_name);
  if (it != embedded_fonts.end())
    return it->second;

  return nullptr;
}

const PDF::Font *PDF::GetBuiltInFont(BuiltInFont f) {
  auto it = builtin_fonts.find(f);
  if (it != builtin_fonts.end())
    return it->second;

  // Create a new font object, then.
  BuiltInFontObj *fobj = AddObject(new BuiltInFontObj);
  CHECK(fobj);
  Font *font = new Font(fobj);
  fobj->font = font;
  font->builtin_font.emplace(f);
  font->font_index = next_font_index;
  next_font_index++;

  builtin_fonts[f] = font;

  const uint16_t *widths_table = find_font_widths(f);
  CHECK(widths_table != nullptr) << "Missing widths table "
    "for built-in font?";

  WidthsObj *wobj = AddObject(new WidthsObj);
  wobj->encoding = FontEncoding::WIN_ANSI;
  wobj->firstchar = 0;
  wobj->lastchar = 255;
  wobj->widths8.resize(256);
  for (int cid = 0; cid < 256; cid++) {
    int w = widths_table[cid];
    wobj->widths8[cid] = w;
    font->widths[cid] = w;
  }

  fobj->widths_obj = wobj;

  return font;
}

bool PDF::SetFont(const std::string &font_name) {
  // See if we've used this font before.
  if (const Font *font = GetFontByName(font_name)) {
    current_font = font;
    return true;
  }

  SetErr(-ENOENT,
         std::format(
             "The font '{}' has not been loaded. "
             "For built-in fonts, use the version of SetFont "
             "that takes an enum.", font_name));
  return false;
}

void PDF::SetFont(BuiltInFont f) {
  current_font = GetBuiltInFont(f);
}

void PDF::SetFont(const Font *font) {
  current_font = font;
}

PDF::Page *PDF::AppendNewPage() {
  Page *page = AddObject(new Page);

  if (!page)
    return nullptr;

  page->width = Width();
  page->height = Height();

  return page;
}

PDF::Page *PDF::GetPage(int page_number) {
  if (page_number <= 0) {
    SetErr(-EINVAL, "page number must be >= 1");
    return nullptr;
  }

  for (Object *obj = FindFirstObject(OBJ_page); obj;
       obj = obj->next, page_number--) {
    if (page_number == 1) {
      return (Page*)obj;
    }
  }

  SetErr(-EINVAL, "no such page");
  return nullptr;
}

void PDF::Page::SetSize(float w, float h) {
  width = w;
  height = h;
}

// Recursively scan for the number of children
int PDF::pdf_get_bookmark_count(const Object *obj) {
  int count = 0;
  if (obj->type == OBJ_bookmark) {
    const BookmarkObj *bobj = (const BookmarkObj*)obj;
    int nchildren = (int)bobj->children.size();
    count += nchildren;
    for (int i = 0; i < nchildren; i++) {
      count += pdf_get_bookmark_count(bobj->children[i]);
    }
  }
  return count;
}

const char *PDF::BuiltInFontName(BuiltInFont f) {
  switch (f) {
  case PDF::HELVETICA: return "Helvetica";
  case PDF::HELVETICA_BOLD: return "Helvetica-Bold";
  case PDF::HELVETICA_OBLIQUE: return "Helvetica-Oblique";
  case PDF::HELVETICA_BOLD_OBLIQUE: return "Helvetica-BoldOblique";
  case PDF::COURIER: return "Courier";
  case PDF::COURIER_BOLD: return "Courier-Bold";
  case PDF::COURIER_OBLIQUE: return "Courier-Oblique";
  case PDF::COURIER_BOLD_OBLIQUE: return "Courier-BoldOblique";
  case PDF::TIMES_ROMAN: return "Times-Roman";
  case PDF::TIMES_BOLD: return "Times-Bold";
  case PDF::TIMES_ITALIC: return "Times-Italic";
  case PDF::TIMES_BOLD_ITALIC: return "Times-BoldItalic";
  case PDF::SYMBOL: return "Symbol";
  case PDF::ZAPF_DINGBATS: return "ZapfDingbats";
  default:
    LOG(FATAL) << "Unknown builtin font";
    return "";
  }
}

// Creates and returns a font descriptor Object (and the
// embedded content stream) for the given font. This must be
// used via an indirect reference (which is why we can't just
// embed it) but is otherwise a boring object.
PDF::Object *PDF::TrueTypeFontDescriptor(
    const stbtt_fontinfo *ttf,
    const std::vector<uint8_t> &ttf_bytes,
    int font_index) {
  CHECK(ttf != nullptr);

  int x0 = 0, y0 = 0, x1 = 0, y1 = 0;
  stbtt_GetFontBoundingBox(ttf, &x0, &y0, &x1, &y1);

  // PERF: Make it possible to just pass a byte vector
  std::string bytes_string;
  bytes_string.resize(ttf_bytes.size());
  for (int i = 0; i < (int)ttf_bytes.size(); i++)
    bytes_string[i] = ttf_bytes[i];

  StreamObj *filestream = AddStreamObject(
      // Ideally we only output Length1 if we do compress.
      {{"/Length1", std::format("{}", ttf_bytes.size())}},
      bytes_string);

  // Now the desriptor itself.
  StreamObj *desc = AddObject(new StreamObj);

  // More in PDF 1.3 spec on p330, 5.7 Font Descriptors
  AppendFormat(&desc->stream,
               "<<\n"
               "  /Type /FontDescriptor\n"
               // FontName; we just use FontName<id>
               "  /FontName /FontName{}\n"
               // Refers to the embedded file in its own stream.
               "  /FontFile2 {} 0 R\n",
               font_index,
               filestream->index);

  AppendFormat(&desc->stream,
               "  /FontBBox [{} {} {} {}]\n",
               x0, y0, x1, y1);

  AppendFormat(&desc->stream,
               ">>\n");

  return desc;
}

// returns the number of bytes parsed. non-positive means error.
static int utf8_to_utf32(const char *utf8, int len, uint32_t *utf32) {
  uint8_t mask = 0;

  if (len <= 0 || !utf8 || !utf32)
    return -EINVAL;

  uint32_t ch = *(const uint8_t *)utf8;
  if ((ch & 0x80) == 0) {
    len = 1;
    mask = 0x7f;
  } else if ((ch & 0xe0) == 0xc0 && len >= 2) {
    len = 2;
    mask = 0x1f;
  } else if ((ch & 0xf0) == 0xe0 && len >= 3) {
    len = 3;
    mask = 0xf;
  } else if ((ch & 0xf8) == 0xf0 && len >= 4) {
    len = 4;
    mask = 0x7;
  } else {
    return -EINVAL;
  }

  ch = 0;
  for (int i = 0; i < len; i++) {
    int shift = (len - i - 1) * 6;
    if (!*utf8)
      return -EINVAL;
    if (i == 0)
      ch |= ((uint32_t)(*utf8++) & mask) << shift;
    else
      ch |= ((uint32_t)(*utf8++) & 0x3f) << shift;
  }

  *utf32 = ch;

  return len;
}

static std::optional<uint32_t> UTF8NextCodepoint(std::string_view *s) {
  uint32_t cp = 0;
  int len = utf8_to_utf32(s->data(), s->size(), &cp);
  if (len <= 0) return std::nullopt;
  s->remove_prefix(len);
  return {cp};
}

// Codepoints aside from 0-128 that are mapped.
static constexpr std::initializer_list<uint16_t> MAPPED_CODEPOINTS = {
  0x152, 0x153, 0x160, 0x161, 0x178, 0x17d, 0x17e, 0x192, 0x2c6, 0x2dc,
  0x2013, 0x2014, 0x2018, 0x2019, 0x201a, 0x201c, 0x201d, 0x201e, 0x2020,
  0x2021, 0x2022, 0x2026, 0x2030, 0x2039, 0x203a, 0x20ac, 0x2122,
  // Added by Tom 7
  0x00BF, 0x00E9, 0x00F4, 0x00F1, 0x00D7,
};

static constexpr std::optional<int> MapCodepointWinAnsi(int codepoint) {
  // Note this does not match the code below, which I think is a bug.
  if (codepoint < 128) return {codepoint};
  switch (codepoint) {
    // TODO: Include a more complete mapping.
    // We support *some* minimal UTF-8 characters.
    // See Appendix D of
    // opensource.adobe.com/dc-acrobat-sdk-docs/pdfstandards/
    //    pdfreference1.7old.pdf
  case 0x152: // Latin Capital Ligature OE
    return {0214};
  case 0x153: // Latin Small Ligature oe
    return {0234};
  case 0x160: // Latin Capital Letter S with caron
    return {0212};
  case 0x161: // Latin Small Letter S with caron
    return {0232};
  case 0x178: // Latin Capital Letter y with diaeresis
    return {0237};
  case 0x17d: // Latin Capital Letter Z with caron
    return {0216};
  case 0x17e: // Latin Small Letter Z with caron
    return {0236};
  case 0x192: // Latin Small Letter F with hook
    return {0203};
  case 0x2c6: // Modifier Letter Circumflex Accent
    return {0210};
  case 0x2dc: // Small Tilde
    return {0230};
  case 0x2013: // Endash
    return {0226};
  case 0x2014: // Emdash
    return {0227};
  case 0x2018: // Left Single Quote
    return {0221};
  case 0x2019: // Right Single Quote
    return {0222};
  case 0x201a: // Single low-9 Quotation Mark
    return {0202};
  case 0x201c: // Left Double Quote
    return {0223};
  case 0x201d: // Right Double Quote
    return {0224};
  case 0x201e: // Double low-9 Quotation Mark
    return {0204};
  case 0x2020: // Dagger
    return {0206};
  case 0x2021: // Double Dagger
    return {0207};
  case 0x2022: // Bullet
    return {0225};
  case 0x2026: // Horizontal Ellipsis
    return {0205};
  case 0x2030: // Per Mille Sign
    return {0211};
  case 0x2039: // Single Left-pointing Angle Quotation Mark
    return {0213};
  case 0x203a: // Single Right-pointing Angle Quotation Mark
    return {0233};
  case 0x20ac: // Euro
    return {0200};
  case 0x2122: // Trade Mark Sign
    return {0231};
  case 0x00BF: // Rotated question mark
    return {0277};
  case 0x00E9: // e acute
    return {0351};
  case 0x00F4: // o circumflex
    return {0364};
  case 0x00F1: // n tilde
    return {0361};
  case 0x00D7: // multiplication sign
    return {0327};
  default:
    break;
  }

  return std::nullopt;
}

static int UTF8ToWinAnsiEncoding(const char *utf8, int len, uint8_t *res) {
  *res = 0;

  uint32_t code = 0;
  int code_len = utf8_to_utf32(utf8, len, &code);
  CHECK(code_len >= 0) << "Invalid UTF-8 encoding";

  // XXX Bug? Does this encoding really map U+0080 to U+00FF?
  if (code > 255) {
    const auto co = MapCodepointWinAnsi(code);
    CHECK(co.has_value()) <<
      std::format("Unsupported UTF-8 character: 0x{:x} 0o{:o} {}",
                   code, code, utf8);
    *res = co.value();
  } else {
    *res = code;
  }

  return code_len;
}

// Encode a string in "PDF Document Encoding", which is the encoding
// used for text strings (like the title) within PDF itself. Input
// text is UTF-8, but only Latin-1 is supported. Unsupported
// codepoints are dropped.
static std::string PDFDocEncodeString(std::string_view utf8_text) {
  std::string ret = "(";
  ret.reserve(utf8_text.size());
  while (!utf8_text.empty()) {

    std::optional<uint32_t> cpo = UTF8NextCodepoint(&utf8_text);
    if (!cpo.has_value()) break;
    uint8_t latin1 = UnicodeToLatin1(cpo.value());
    switch (latin1) {
    case 0:
      // could not be converted; just skip
      continue;
    case '(':
    case ')':
    case '\\':
      // Escape these.
      ret.push_back('\\');
      ret.push_back(latin1);
      break;
    case '\n':
    case '\r':
    case '\t':
    case '\b':
    case '\f':
      // Skip over these characters.
      break;
    default:
      if (latin1 >= 32 && latin1 <= 128) {
        ret.push_back(latin1);
      } else {
        AppendFormat(&ret, "\\{:03o}", latin1);
      }
    }
  }
  ret.push_back(')');
  return ret;
}

int PDF::SaveObject(FILE *fp, int index) {
  Object *object = GetObject(index);
  if (!object)
    return -ENOENT;

  if (VERBOSE) {
    printf("Save object %p %d type %d=%s\n",
           fp, index, object->type,
           ObjTypeName(object->type));
  }

  // This is a special placeholder object in slot 0.
  // It does not get written out.
  if (object->type == OBJ_none)
    return -ENOENT;

  object->offset = ftell(fp);

  fprintf(fp, "%d 0 obj\n", index);

  switch (object->type) {
  case OBJ_stream: {
    const StreamObj *sobj = (const StreamObj*)object;
    fwrite(sobj->stream.data(),
           sobj->stream.size(),
           1, fp);
    break;
  }

  case OBJ_image: {
    const ImageObj *iobj = (const ImageObj *)object;
    fwrite(iobj->stream.data(),
           iobj->stream.size(),
           1, fp);
    break;
  }

  case OBJ_info: {
    const InfoObj *iobj = (const InfoObj*)object;
    const PDF::Info *info = &iobj->info;

    fprintf(fp, "<<\n");
    if (!info->creator.empty())
      fprintf(fp, "  /Creator %s\n",
              PDFDocEncodeString(info->creator).c_str());
    if (!info->producer.empty())
      fprintf(fp, "  /Producer %s\n",
              PDFDocEncodeString(info->producer).c_str());
    if (!info->title.empty())
      fprintf(fp, "  /Title %s\n",
              PDFDocEncodeString(info->title).c_str());
    if (!info->author.empty())
      fprintf(fp, "  /Author %s\n",
              PDFDocEncodeString(info->author).c_str());
    if (!info->subject.empty())
      fprintf(fp, "  /Subject %s\n",
              PDFDocEncodeString(info->subject).c_str());
    if (!info->keywords.empty())
      fprintf(fp, "  /Keywords %s\n",
              PDFDocEncodeString(info->keywords).c_str());

    if (info->date[0])
      fprintf(fp, "  /CreationDate %s\n",
              PDFDocEncodeString(
                  std::string("D:") + info->date).c_str());
    fprintf(fp, ">>\n");
    break;
  }

  case OBJ_page: {
    const Object *pages = FindFirstObject(OBJ_pages);
    bool printed_xobjects = false;

    Page *pobj = (Page*)object;

    // Flush any delayed buffered commands, which may add
    // a new child.
    FlushDrawCommands(pobj);

    fprintf(fp,
            "<<\n"
            "  /Type /Page\n"
            "  /Parent %d 0 R\n",
            pages->index);
    fprintf(fp, "  /MediaBox [0 0 %s %s]\n",
            Float(pobj->width).c_str(),
            Float(pobj->height).c_str());
    fprintf(fp, "  /Resources <<\n");
    fprintf(fp, "    /Font <<\n");

    // Annoyingly we have to name all of the fonts again. This
    // is pretty short, but best would be if we only made names
    // for the ones actually used on this page? Also annoying
    // is that there are three different objects that we might
    // be using as fonts.
    auto RegisterFont = [fp](int font_index, int obj_ref) {
        fprintf(fp, "      /F%d %d 0 R\n", font_index, obj_ref);
      };
    for (const Object *obj = FindFirstObject(OBJ_builtin_font);
         obj; obj = obj->next) {
      const BuiltInFontObj *fbobj = (const BuiltInFontObj *)obj;
      RegisterFont(fbobj->font->font_index, fbobj->index);
    }
    for (const Object *obj = FindFirstObject(OBJ_font8);
         obj; obj = obj->next) {
      const Font8Obj *f8obj = (const Font8Obj *)obj;
      RegisterFont(f8obj->font->font_index, f8obj->index);
    }
    for (const Object *obj = FindFirstObject(OBJ_font0);
         obj; obj = obj->next) {
      const Font0Obj *f0obj = (const Font0Obj *)obj;
      RegisterFont(f0obj->font->font_index, f0obj->index);
    }
    // ... but notably not CID font objects, since we do not
    // refer to those directly (they exist only as "descendant"
    // fonts of the type 0 containers).

    fprintf(fp, "    >>\n");
    // TODO: The way alpha is implemented is to always generate 16
    // different graphics states on each page, for 4-bit transparency.
    // (/GS0 ... /GS15). Better would be to register the actual
    // transparency levels used as we draw stuff, and allow arbitrary
    // levels.
    //
    // We trim transparency to just 4 bits.
    fprintf(fp, "    /ExtGState <<\n");
    for (int i = 0; i < 16; i++) {
      fprintf(fp, "      /GS%d <</ca %f>>\n", i,
              (float)(15 - i) / 15);
    }
    fprintf(fp, "    >>\n");

    for (const Object *image = FindFirstObject(OBJ_image);
         image; image = image->next) {
      const ImageObj *iobj = (const ImageObj *)image;
      if (iobj->page == object) {
        if (!printed_xobjects) {
          fprintf(fp, "    /XObject <<");
          printed_xobjects = true;
        }
        fprintf(fp, "      /Image%d %d 0 R ",
                image->index,
                image->index);
      }
    }
    if (printed_xobjects)
      fprintf(fp, "    >>\n");
    fprintf(fp, "  >>\n");

    fprintf(fp, "  /Contents [\n");
    for (const Object *child : pobj->children) {
      fprintf(fp, "%d 0 R\n", child->index);
    }
    fprintf(fp, "]\n");

    if (!pobj->annotations.empty()) {
      fprintf(fp, "  /Annots [\n");
      for (const Object *child : pobj->annotations) {
        fprintf(fp, "%d 0 R\n", child->index);
      }
      fprintf(fp, "]\n");
    }

    fprintf(fp, ">>\n");
    break;
  }

  case OBJ_bookmark: {
    const BookmarkObj *bobj = (const BookmarkObj *)object;

    const Object *parent = bobj->parent;
    if (!parent)
      parent = FindFirstObject(OBJ_outline);
    if (!bobj->page)
      break;
    fprintf(fp,
            "<<\n"
            "  /Dest [%d 0 R /XYZ 0 %s null]\n"
            "  /Parent %d 0 R\n"
            "  /Title (%s)\n",
            bobj->page->index,
            Float(this->document_height).c_str(),
            parent->index,
            bobj->name.c_str());
    int nchildren = (int)bobj->children.size();
    if (nchildren > 0) {
      const Object *f = (const Object *)bobj->children[0];
      const Object *l = (const Object *)bobj->children[nchildren - 1];
      fprintf(fp, "  /First %d 0 R\n", f->index);
      fprintf(fp, "  /Last %d 0 R\n", l->index);
      fprintf(fp, "  /Count %d\n", pdf_get_bookmark_count(object));
    }

    {
      // Find the previous bookmark with the same parent
      const BookmarkObj *other = (const BookmarkObj*)object->prev;
      while (other && other->parent != bobj->parent) {
        other = (const BookmarkObj*)other->prev;
      }

      if (other != nullptr) {
        fprintf(fp, "  /Prev %d 0 R\n", other->index);
      }
    }

    {
      // Find the next bookmark with the same parent
      const BookmarkObj *other = (BookmarkObj *)object->next;
      while (other && other->parent != bobj->parent) {
        other = (const BookmarkObj *)other->next;
      }

      if (other != nullptr) {
        fprintf(fp, "  /Next %d 0 R\n", other->index);
      }
    }

    fprintf(fp, ">>\n");
    break;
  }

  case OBJ_outline: {
    const Object *first = FindFirstObject(OBJ_bookmark);
    const Object *last = FindLastObject(OBJ_bookmark);

    if (first && last) {
      int count = 0;
      const BookmarkObj *cur = (const BookmarkObj *)first;
      while (cur) {
        if (!cur->parent) {
          count += pdf_get_bookmark_count(cur) + 1;
        }
        cur = (const BookmarkObj *)cur->next;
      }

      /* Bookmark outline */
      fprintf(fp,
              "<<\n"
              "  /Count %d\n"
              "  /Type /Outlines\n"
              "  /First %d 0 R\n"
              "  /Last %d 0 R\n"
              ">>\n",
              count, first->index, last->index);
    }
    break;
  }

  case OBJ_font0: {
    const Font0Obj *fobj = (const Font0Obj*)object;
    CHECK(fobj->font != nullptr);
    CHECK(fobj->fcobj != nullptr);
    CHECK(!fobj->font->builtin_font.has_value()) << "Built-in fonts "
      "should not use Type 0 wrappers.";
    CHECK(fobj->cmap_obj != nullptr) << "We should have a CMap object "
      "when using Unicode text encoding.";


    FontEncoding encoding = fobj->font->encoding;
    CHECK(encoding == FontEncoding::UNICODE);

    // This is the Type 0 wrapper for a single CIDType2 (embedded
    // truetype) font. We always just have the single descendant
    // font, and this is where we put the CMap.

    fprintf(fp,
            "<<\n"
            "  /Type /Font\n"
            "  /Subtype /Type0\n"
            // This is the name we can use elsewhere to draw text.
            "  /BaseFont /Font%d\n"
            // Identity means we just use 16-bit glyph codes (two
            // bytes each) from the font. The PDF understands what
            // codepoints these are because of the /ToUnicode
            // CMap.
            "  /Encoding /Identity-H\n"
            // The single wrapped CID font.
            "  /DescendantFonts [%d 0 R]\n"
            "  /ToUnicode %d 0 R\n"
            ">>\n",
            // Basefont: Just needs a unique name.
            fobj->index,
            fobj->fcobj->index,
            fobj->cmap_obj->index);

    break;
  }

  case OBJ_fontcid: {
    const FontCIDObj *fobj = (const FontCIDObj*)object;
    CHECK(fobj->font != nullptr);
    CHECK(!fobj->font->builtin_font.has_value()) << "Built-in fonts "
      "should not use this legacy object";

    FontEncoding encoding = fobj->font->encoding;
    CHECK(encoding == FontEncoding::UNICODE);

    CHECK(fobj->ttf != nullptr) << "Missing embedded TTF?";

    // We need a width16 since we'll refer to it with /W.
    // Maybe would be cleaner to just have two different widths object
    // types.
    CHECK(fobj->widths_obj != nullptr);
    CHECK(fobj->widths_obj->encoding == FontEncoding::UNICODE);

    // The CIDFontType2 is wrapped inside the Type0 font. This is
    // where we embed the font data and metrics, but the cmap is in
    // the parent.

    fprintf(fp,
            "<<\n"
            "  /Type /Font\n"
            "  /Subtype /CIDFontType2\n"
            "  /BaseFont /Font%d\n"
            "  /FontDescriptor %d 0 R\n"
            // This is boilerplate saying that we want CID = glyph id.
            "  /CIDSystemInfo <<\n"
            "    /Registry (Adobe)\n"
            "    /Ordering (Identity)\n"
            "    /Supplement 0\n"
            "  >>\n"
            "  /CIDToGIDMap /Identity\n"
            // The default width (not really supported at the moment)
            "  /DW %d\n"
            // The array of widths for this font.
            "  /W %d 0 R\n"
            ">>\n",
            // Basefont: Just needs a unique name.
            fobj->index,
            fobj->ttf->index,
            fobj->widths_obj->default_width,
            fobj->widths_obj->index);

    break;
  }

  case OBJ_font8: {
    // Embedded WinAnsi-encoded truetype font.
    const Font8Obj *fobj = (const Font8Obj*)object;
    CHECK(fobj->widths_obj != nullptr) << "All fonts should have "
      "a widths object.";
    CHECK(fobj->font != nullptr);
    CHECK(!fobj->font->builtin_font.has_value()) << "Built-in fonts "
      "should not use this.";

    FontEncoding encoding = fobj->font->encoding;
    CHECK(encoding == FontEncoding::WIN_ANSI);
    CHECK(fobj->ttf != nullptr) << "Missing embedded TTF?";

    fprintf(fp,
            "<<\n"
            "  /Type /Font\n"
            "  /Subtype /TrueType\n"
            "  /BaseFont /Font%d\n"
            "  /Encoding /WinAnsiEncoding\n"
            "  /FontDescriptor %d\n"
            "  /FirstChar %d\n"
            "  /LastChar %d\n"
            "  /Widths %d 0 R\n"
            ">>\n",
            // Basefont: Just needs a unique name.
            fobj->index,
            fobj->ttf->index,
            // Widths
            fobj->widths_obj->firstchar,
            fobj->widths_obj->lastchar,
            // Array of widths in its own object.
            fobj->widths_obj->index);

    break;
  }

  case OBJ_builtin_font: {
    const BuiltInFontObj *fobj = (const BuiltInFontObj*)object;
    CHECK(fobj->widths_obj != nullptr) << "All fonts should have "
      "a widths object. Even built-in ones!";
    CHECK(fobj->font != nullptr);

    CHECK(fobj->font->builtin_font.has_value()) << "A BuiltInFontObj "
      "should reference a built-in Font!";

    CHECK(fobj->font->encoding == FontEncoding::WIN_ANSI) << "Built-in "
      "fonts are WIN_ANSI by definition.";

    // A built-in font (BaseFont).
    fprintf(fp,
            "<<\n"
            "  /Type /Font\n"
            "  /Subtype /Type1\n"
            "  /BaseFont /%s\n"
            "  /Encoding /WinAnsiEncoding\n"
            "  /FirstChar %d\n"
            "  /LastChar %d\n"
            "  /Widths %d 0 R\n"
            ">>\n",
            BuiltInFontName(fobj->font->builtin_font.value()),
            fobj->widths_obj->firstchar,
            fobj->widths_obj->lastchar,
            // Array of widths in its own object.
            fobj->widths_obj->index);

    break;
  }

  case OBJ_pages: {
    int npages = 0;

    fprintf(fp, "<<\n"
            "  /Type /Pages\n"
            "  /Kids [ ");
    for (const Object *page = FindFirstObject(OBJ_page);
         page; page = page->next) {
      npages++;
      fprintf(fp, "%d 0 R ", page->index);
    }
    fprintf(fp, "]\n");
    fprintf(fp, "  /Count %d\n", npages);
    fprintf(fp, ">>\n");
    break;
  }

  case OBJ_catalog: {
    const Object *outline = FindFirstObject(OBJ_outline);
    const Object *pages = FindFirstObject(OBJ_pages);

    fprintf(fp, "<<\n"
            "  /Type /Catalog\n");
    if (outline != nullptr) {
      fprintf(fp,
              "  /Outlines %d 0 R\n"
              "  /PageMode /UseOutlines\n",
              outline->index);
    }
    fprintf(fp,
            "  /Pages %d 0 R\n"
            ">>\n",
            pages->index);
    break;
  }

  case OBJ_link: {
    const LinkObj *lobj = (LinkObj *)object;
    fprintf(fp,
            "<<\n"
            "  /Type /Annot\n"
            "  /Subtype /Link\n"
            "  /Rect [%s %s %s %s]\n"
            "  /Dest [%u 0 R /XYZ %s %s null]\n"
            "  /Border [0 0 0]\n"
            ">>\n",
            Float(lobj->llx).c_str(),
            Float(lobj->lly).c_str(),
            Float(lobj->urx).c_str(),
            Float(lobj->ury).c_str(),
            lobj->target_page->index,
            Float(lobj->target_x).c_str(),
            Float(lobj->target_y).c_str());
    break;
  }

  case OBJ_widths: {
    const WidthsObj *wobj = (WidthsObj *)object;

    // The format of the object depends on the encoding.
    // It could be a /Widths (win ansi) or /W (unicode).
    // Probably it would be cleaner to have different object
    // types, or to just do this in the font obj code.
    if (wobj->encoding == FontEncoding::WIN_ANSI) {
      fprintf(fp, "[");
      for (int w : wobj->widths8) {
        fprintf(fp, " %d", w);
      }
      fprintf(fp, " ]\n");
    } else {
      CHECK(wobj->encoding == FontEncoding::UNICODE);

      std::vector<std::pair<uint16_t, int>> sorted =
        MapToSortedVec(wobj->widths16);

      // Two entry types:
      // first last width      (which meanas first = width ... last = width)
      // base [w0 w1 w2 w3]    (which means base+0 = w0, base+1 = w1, ...)
      //
      // This is the dumbest encoding to get it working. We should
      // generate something more compact:
      fprintf(fp, "[\n");
      for (int i = 0; i < (int)sorted.size(); i++) {
        const auto &[cid, width] = sorted[i];
        fprintf(fp, "%d %d %d\n", cid, cid, width);
      }
      fprintf(fp, "]\n");

      #if 0
      // WRONG!
      fprintf(fp, "[\n");
      std::optional<int> prev_width = {};
      for (int i = 0; i < (int)sorted.size(); i++) {
        const auto &[cid, width] = sorted[i];
        // Always output the last one exactly, since otherwise
        // we won't imply runs up to it.
        if (i == (int)sorted.size() - 1) {
          fprintf(fp, "%d %d\n", cid, width);
          prev_width = std::nullopt;
        } else if (prev_width.has_value() &&
                   width == prev_width.value()) {
          // Just continue the run in this case.
          continue;
        } else {
          fprintf(fp, "%d %d\n", cid, width);
          prev_width = {width};
        }
      }
      fprintf(fp, "]\n");
      #endif
    }
    break;
  }

  default:
    return SetErr(-EINVAL,
                  std::format("Invalid PDF object type {}",
                              (int)object->type));
  }

  fprintf(fp, "endobj\n");

  return 0;
}

// Slightly modified djb2 hash algorithm to get pseudo-random ID
static uint64_t hash(uint64_t hash, const void *data, size_t len) {
  const uint8_t *d8 = (const uint8_t *)data;
  for (; len; len--) {
    hash = (((hash & 0x03ffffffffffffff) << 5) +
            (hash & 0x7fffffffffffffff)) +
      *d8++;
  }
  return hash;
}

int PDF::SaveFile(FILE *fp) {
  int xref_offset;
  int xref_count = 0;
  uint64_t id1, id2;
  time_t now = time(nullptr);

  // TODO: The original code did some shenanigans to change the locale
  // temporarily, since it claims that some locales will convert the
  // decimal point to a comma. This seems bad (not thread safe, etc.).
  // We should avoid locale-specific formatting here, as the decimal
  // point must be . in the PDF format. std::format will
  // do the right thing (always c locale unless specified).

  fprintf(fp, "%%PDF-1.3\n");
  /* Hibit bytes */
  fprintf(fp, "%c%c%c%c%c\n", 0x25, 0xc7, 0xec, 0x8f, 0xa2);

  /* Dump all the objects & get their file offsets */
  for (int i = 0; i < (int)objects.size(); i++) {
    int err = SaveObject(fp, i);
    if (err >= 0) {
      xref_count++;
    } else if (err == -ENOENT) {
      /* ok */
    } else {
      LOG(FATAL) << "Could not write object: " << error_message;
    }
  }

  /* xref */
  xref_offset = ftell(fp);
  fprintf(fp, "xref\n");
  fprintf(fp, "0 %d\n", xref_count + 1);
  fprintf(fp, "0000000000 65535 f\n");
  for (Object *obj : objects) {
    if (obj->type != OBJ_none) {
      fprintf(fp, "%10.10d 00000 n\n", obj->offset);
    }
  }

  fprintf(fp,
          "trailer\n"
          "<<\n"
          "/Size %d\n",
          xref_count + 1);
  Object *obj = FindFirstObject(OBJ_catalog);
  CHECK(obj != nullptr);
  fprintf(fp, "/Root %d 0 R\n", obj->index);

  const InfoObj *iobj = (InfoObj*)FindFirstObject(OBJ_info);
  fprintf(fp, "/Info %d 0 R\n", iobj->index);
  /* Generate document unique IDs */
  id1 = hash(5381, &iobj->info, sizeof (PDF::Info));
  id1 = hash(id1, &xref_count, sizeof (xref_count));
  id2 = hash(5381, &now, sizeof(now));
  fprintf(fp, "/ID [<%16.16" PRIx64 "> <%16.16" PRIx64 ">]\n", id1, id2);
  fprintf(fp, ">>\n"
          "startxref\n");
  fprintf(fp, "%d\n", xref_offset);
  fprintf(fp, "%%%%EOF\n");

  return 0;
}

bool PDF::Save(const std::string &filename) {
  FILE *fp = fopen(filename.c_str(), "wb");

  if (fp == nullptr) {
    SetErr(-errno,
           std::format("Unable to open '{}': {}", filename,
                       strerror(errno)));
    return false;
  }

  int e = SaveFile(fp);

  if (fp != stdout) {
    if (fclose(fp) != 0 && e >= 0) {
      SetErr(-errno,
             std::format("Unable to close '{}': {}",
                         filename, strerror(errno)));
      return false;
    }
  }

  return true;
}

// In order to reduce the number of objects, and increase the effectiveness
// of compression, we consolidate consecutive drawing commands on a page
// into a single stream. This manages all of that.
void PDF::AppendDrawCommand(Page *page, std::string_view cmd) {
  if (!page)
    page = (Page*)FindLastObject(OBJ_page);

  auto IsWhitespace = [](char c) { return c == ' ' || c == '\n'; };

  CHECK(page != nullptr);
  if (!page->draw_buffer.empty() && !IsWhitespace(page->draw_buffer.back()))
    page->draw_buffer.push_back('\n');

  page->draw_buffer.append(cmd);
}

// When adding an object (e.g. an image) that is not drawing commands,
// flush existing commands to a regular stream object so that they are
// properly ordered.
void PDF::FlushDrawCommands(Page *page) {
  CHECK(page != nullptr);
  if (!page->draw_buffer.empty()) {
    pdf_add_stream_raw(page, std::move(page->draw_buffer));
    page->draw_buffer.clear();
  }
}

// Consider whether you can use AppendDrawCommand.
void PDF::pdf_add_stream(Page *page, std::string str) {
  if (!page)
    page = (Page*)FindLastObject(OBJ_page);
  FlushDrawCommands(page);
  pdf_add_stream_raw(page, std::move(str));
}

PDF::StreamObj *PDF::AddStreamObject(
    // Like {{"/Type", "/CMap"}}
    const std::vector<std::pair<std::string, std::string>> &keys,
    const std::string &contents) {
  StreamObj *sobj = AddObject(new StreamObj);
  CHECK(sobj != nullptr);

  sobj->stream = "<< ";
  for (const auto &[k, v] : keys) {
    CHECK(!k.empty() && k[0] == '/' && !v.empty());
    AppendFormat(&sobj->stream, "{} {} ", k, v);
  }

  // Note: Stream /Length does not include the \n after stream or
  // before endstream.
  if (options.use_compression) {
    std::string flate_bytes =
      ZIP::ZlibString(contents, options.compression_level);
    // /Filter /FlateDecode
    // 012345678901234567890
    if (flate_bytes.size() + 20 < contents.size()) {
      AppendFormat(&sobj->stream,
                   "/Length {} /Filter /FlateDecode >>stream\n",
                   (int)flate_bytes.size());
      sobj->stream.append(flate_bytes);
    } else {
      // If it's not smaller (which is common for very short streams,
      // for example) then don't bother compressing it.
      AppendFormat(&sobj->stream,
                   "/Length {} >>stream\n", (int)contents.size());
      sobj->stream.append(contents);
    }
  } else {
    AppendFormat(&sobj->stream, "/Length {} >>stream\n",
                 (int)contents.size());
    sobj->stream.append(contents);
  }
  AppendFormat(&sobj->stream, "\nendstream\n");

  return sobj;
}

void PDF::pdf_add_stream_raw(Page *page, std::string str) {
  CHECK(page != nullptr) << "You may need to add a page to the document "
    "first.";

  // We don't want any trailing whitespace in the stream.
  // (XXX: Is this OK for non-text streams? -tom7)
  while (!str.empty() &&
         (str.back() == '\r' ||
          str.back() == '\n')) {
    str.resize(str.size() - 1);
  }

  page->children.push_back(AddStreamObject({}, str));
}

namespace {
struct Encoding128a {
  uint32_t code;
  char ch;
};
}

static std::array<Encoding128a, 107> code_128a_encoding = {
    Encoding128a{0x212222, ' '},
    {0x222122, '!'},  {0x222221, '"'},  {0x121223, '#'},
    {0x121322, '$'},  {0x131222, '%'},  {0x122213, '&'},   {0x122312, '\''},
    {0x132212, '('},  {0x221213, ')'},  {0x221312, '*'},   {0x231212, '+'},
    {0x112232, ','},  {0x122132, '-'},  {0x122231, '.'},   {0x113222, '/'},
    {0x123122, '0'},  {0x123221, '1'},  {0x223211, '2'},   {0x221132, '3'},
    {0x221231, '4'},  {0x213212, '5'},  {0x223112, '6'},   {0x312131, '7'},
    {0x311222, '8'},  {0x321122, '9'},  {0x321221, ':'},   {0x312212, ';'},
    {0x322112, '<'},  {0x322211, '='},  {0x212123, '>'},   {0x212321, '?'},
    {0x232121, '@'},  {0x111323, 'A'},  {0x131123, 'B'},   {0x131321, 'C'},
    {0x112313, 'D'},  {0x132113, 'E'},  {0x132311, 'F'},   {0x211313, 'G'},
    {0x231113, 'H'},  {0x231311, 'I'},  {0x112133, 'J'},   {0x112331, 'K'},
    {0x132131, 'L'},  {0x113123, 'M'},  {0x113321, 'N'},   {0x133121, 'O'},
    {0x313121, 'P'},  {0x211331, 'Q'},  {0x231131, 'R'},   {0x213113, 'S'},
    {0x213311, 'T'},  {0x213131, 'U'},  {0x311123, 'V'},   {0x311321, 'W'},
    {0x331121, 'X'},  {0x312113, 'Y'},  {0x312311, 'Z'},   {0x332111, '['},
    {0x314111, '\\'}, {0x221411, ']'},  {0x431111, '^'},   {0x111224, '_'},
    {0x111422, '`'},  {0x121124, 'a'},  {0x121421, 'b'},   {0x141122, 'c'},
    {0x141221, 'd'},  {0x112214, 'e'},  {0x112412, 'f'},   {0x122114, 'g'},
    {0x122411, 'h'},  {0x142112, 'i'},  {0x142211, 'j'},   {0x241211, 'k'},
    {0x221114, 'l'},  {0x413111, 'm'},  {0x241112, 'n'},   {0x134111, 'o'},
    {0x111242, 'p'},  {0x121142, 'q'},  {0x121241, 'r'},   {0x114212, 's'},
    {0x124112, 't'},  {0x124211, 'u'},  {0x411212, 'v'},   {0x421112, 'w'},
    {0x421211, 'x'},  {0x212141, 'y'},  {0x214121, 'z'},   {0x412121, '{'},
    {0x111143, '|'},  {0x111341, '}'},  {0x131141, '~'},   {0x114113, '\0'},
    {0x114311, '\0'}, {0x411113, '\0'}, {0x411311, '\0'},  {0x113141, '\0'},
    {0x114131, '\0'}, {0x311141, '\0'}, {0x411131, '\0'},  {0x211412, '\0'},
    {0x211214, '\0'}, {0x211232, '\0'}, {0x2331112, '\0'}
};

// PERF: Linear lookup! We can just invert the mapping and use the
// char as an index.
static int find_128_encoding(char ch) {
  for (int i = 0; i < (int)code_128a_encoding.size(); i++) {
    if (code_128a_encoding[i].ch == ch)
      return i;
  }
  return -1;
}

float PDF::pdf_barcode_128a_ch(float x, float y, float width, float height,
                               uint32_t color, int index, int code_len,
                               Page *page) {
  const uint32_t code = code_128a_encoding[index].code;
  const float line_width = width / 11.0f;

  for (int i = 0; i < code_len; i++) {
    uint8_t shift = (code_len - 1 - i) * 4;
    uint8_t mask = (code >> shift) & 0xf;

    if (!(i % 2))
      AddFilledRectangle(x, y, line_width * mask,
                         height, 0, color, PDF_TRANSPARENT, page);
    x += line_width * mask;
  }

  return x;
}

bool PDF::AddBarcode128a(float x, float y, float width, float height,
                         const std::string &str, uint32_t color, Page *page) {
  const size_t len = str.size() + 3;
  const float char_width = width / len;

  if (char_width / 11.0f <= 0) {
    SetErr(-EINVAL, "Insufficient width to draw barcode");
    return false;
  }

  for (char c : str) {
    if (find_128_encoding(c) < 0) {
      SetErr(-EINVAL, std::format("Invalid barcode character 0x{:x}", c));
      return false;
    }
  }

  x = pdf_barcode_128a_ch(x, y, char_width, height, color, 104, 6, page);
  int checksum = 104;

  for (int i = 0; i < (int)str.size(); i++) {
    int index = find_128_encoding(str[i]);
    CHECK(index >= 0) << "Bug: Checked above.";
    x = pdf_barcode_128a_ch(x, y, char_width, height, color,
                            index, 6, page);
    checksum += index * (i + 1);
  }

  x = pdf_barcode_128a_ch(x, y, char_width, height, color,
                          checksum % 103, 6, page);

  (void)pdf_barcode_128a_ch(x, y, char_width, height, color,
                            106, 7, page);

  return true;
}

namespace {
// Code 39 character encoding. Each 4-bit value indicates:
// 0 => wide bar
// 1 => narrow bar
// 2 => wide space
struct Encoding39 {
  int code;
  char ch;
};
}  // namespace

static constexpr std::array<Encoding39, 40> code_39_encoding = {
  Encoding39{0x012110, '1'}, {0x102110, '2'}, {0x002111, '3'},
    {0x112010, '4'}, {0x012011, '5'}, {0x102011, '6'},
    {0x112100, '7'}, {0x012101, '8'}, {0x102101, '9'},
    {0x112001, '0'}, {0x011210, 'A'}, {0x101210, 'B'},
    {0x001211, 'C'}, {0x110210, 'D'}, {0x010211, 'E'},
    {0x100211, 'F'}, {0x111200, 'G'}, {0x011201, 'H'},
    {0x101201, 'I'}, {0x110201, 'J'}, {0x011120, 'K'},
    {0x101120, 'L'}, {0x001121, 'M'}, {0x110120, 'N'},
    {0x010121, 'O'}, {0x100121, 'P'}, {0x111020, 'Q'},
    {0x011021, 'R'}, {0x101021, 'S'}, {0x110021, 'T'},
    {0x021110, 'U'}, {0x120110, 'V'}, {0x020111, 'W'},
    {0x121010, 'X'}, {0x021011, 'Y'}, {0x120011, 'Z'},
    {0x121100, '-'}, {0x021101, '.'}, {0x120101, ' '},
    {0x121001, '*'}, // 'stop' character
};

// PERF As above.
static int find_39_encoding(char ch) {
  for (int i = 0; i < (int)code_39_encoding.size(); i++) {
    if (code_39_encoding[i].ch == ch) {
      return code_39_encoding[i].code;
    }
  }
  return -1;
}

bool PDF::pdf_barcode_39_ch(float x, float y, float char_width, float height,
                            uint32_t color, char ch, float *new_x, Page *page) {
  const float nw = char_width / 12.0f;
  const float ww = char_width / 4.0f;
  const int code = find_39_encoding(ch);

  if (code < 0) {
    SetErr(-EINVAL,
           std::format("Invalid Code 39 character {:c} 0x{:x}", ch, ch));
    return false;
  }

  for (int i = 5; i >= 0; i--) {
    const int pattern = (code >> i * 4) & 0xf;
    switch (pattern) {
    case 0:
      // wide
      AddFilledRectangle(x, y, ww - 1, height, 0,
                         color, PDF_TRANSPARENT, page);
      x += ww;
      break;
    case 1:
      // narrow
      AddFilledRectangle(x, y, nw - 1, height, 0,
                         color, PDF_TRANSPARENT, page);
      x += nw;
      break;
    case 2:
      // space
      x += nw;
      break;
    default:
      break;
    }
  }

  *new_x = x;

  return true;
}

bool PDF::AddBarcode39(float x, float y, float width, float height,
                       const std::string &str, uint32_t color, Page *page) {
  const size_t len = (int)str.size();
  const float char_width = width / (len + 2);

  CHECK(pdf_barcode_39_ch(x, y, char_width, height, color, '*',
                          &x, page)) << "Bug: * needs encoding.";

  for (char c : str) {
    bool ok = pdf_barcode_39_ch(x, y, char_width, height, color,
                                c, &x, page);
    // XXX would be better if this rejected the barcode before
    // starting to draw it.
    if (!ok) {
      SetErr(-EINVAL,
             std::format("Character 0x{:02x} cannot be encoded", c));
      return false;
    }
  }

  CHECK(pdf_barcode_39_ch(x, y, char_width, height, color, '*',
                          &x, page)) << "Bug: * needs encoding.";

  return true;
}

// EAN/UPC character encoding. Each 4-bit value indicates width in x units.
// Elements are SBSB (Space, Bar, Space, Bar) for LHS digits.
// Elements are inverted for RHS digits.
static constexpr const int code_eanupc_encoding[] = {
  0x3211, // 0
  0x2221, // 1
  0x2122, // 2
  0x1411, // 3
  0x1132, // 4
  0x1231, // 5
  0x1114, // 6
  0x1312, // 7
  0x1213, // 8
  0x3112, // 9
};

static constexpr const int code_eanupc_aux_encoding[] = {
  0x150, // Normal guard
  0x554, // Centre guard
  0x555, // Special guard
  0x160, // Add-on guard
  0x500, // Add-on delineator
};

static constexpr const int set_ean13_encoding[] = {
  0x00, // 0
  0x34, // 1
  0x2c, // 2
  0x1c, // 3
  0x32, // 4
  0x26, // 5
  0x0e, // 6
  0x2a, // 7
  0x1a, // 8
  0x16, // 9
};

static constexpr const int set_upce_encoding[] = {
  0x07, // 0
  0x0b, // 1
  0x13, // 2
  0x23, // 3
  0x0d, // 4
  0x19, // 5
  0x31, // 6
  0x15, // 7
  0x25, // 8
  0x29, // 9
};

static constexpr float EANUPC_X = PDF_MM_TO_POINT(0.330f);

enum BarcodeType {
  PDF_BARCODE_EAN13 = 0,
  PDF_BARCODE_UPCA = 1,
  PDF_BARCODE_EAN8 = 2,
  PDF_BARCODE_UPCE = 2,
};

// indexed by BarcodeType.
static constexpr const struct {
  unsigned modules;
  float height_bar;
  float height_outer;
  unsigned quiet_left;
  unsigned quiet_right;
} eanupc_dimensions[] = {
  {113, PDF_MM_TO_POINT(22.85), PDF_MM_TO_POINT(25.93), 11, 7}, // EAN-13
  {113, PDF_MM_TO_POINT(22.85), PDF_MM_TO_POINT(25.93), 9, 9},  // UPC-A
  {81, PDF_MM_TO_POINT(18.23), PDF_MM_TO_POINT(21.31), 7, 7},   // EAN-8
  {67, PDF_MM_TO_POINT(22.85), PDF_MM_TO_POINT(25.93), 9, 7},   // UPC-E
};

static void pdf_barcode_eanupc_calc_dims(BarcodeType type, float width, float height,
                                         float *x_off, float *y_off,
                                         float *new_width, float *new_height,
                                         float *x, float *bar_height,
                                         float *bar_ext, float *font_size) {
  float aspect_rect = width / height;
  float aspect_barcode = eanupc_dimensions[type].modules *
    EANUPC_X /
    eanupc_dimensions[type].height_outer;
  if (aspect_rect > aspect_barcode) {
    *new_height = height;
    *new_width = height * aspect_barcode;
  } else if (aspect_rect < aspect_barcode) {
    *new_width = width;
    *new_height = width / aspect_barcode;
  } else {
    *new_width = width;
    *new_height = height;
  }
  float scale = *new_height /
    eanupc_dimensions[type].height_outer;

  *x = *new_width / eanupc_dimensions[type].modules;
  *bar_ext = *x * 5;
  *bar_height = eanupc_dimensions[type].height_bar * scale;
  *font_size = 8.0f * scale;
  *x_off = (width - *new_width) / 2.0f;
  *y_off = (height - *new_height) / 2.0f;
}

bool PDF::pdf_barcode_eanupc_ch(float x, float y, float x_width,
                                float height, uint32_t color, char ch,
                                int set, float *new_x, Page *page) {
  if ('0' > ch || ch > '9') {
    SetErr(-EINVAL,
           std::format("Invalid EAN/UPC character {:c} 0x{:x}", ch, ch));
    return false;
  }

  int code = code_eanupc_encoding[ch - '0'];

  for (int i = 3; i >= 0; i--) {
    int shift = (set == 1 ? 3 - i : i) * 4;
    int bar = (set == 2 && i & 0x1) || (set != 2 && (i & 0x1) == 0);
    float width = (float)((code >> shift) & 0xf);

    switch (ch) {
    case '1':
    case '2':
      if ((set == 0 && bar) || (set != 0 && !bar)) {
        width -= 1.0f / 13.0f;
      } else {
        width += 1.0f / 13.0f;
      }
      break;

    case '7':
    case '8':
      if ((set == 0 && bar) || (set != 0 && !bar)) {
        width += 1.0f / 13.0f;
      } else {
        width -= 1.0f / 13.0f;
      }
      break;
    }

    width *= x_width;
    if (bar) {
      AddFilledRectangle(x, y, width, height, 0,
                         color, PDF_TRANSPARENT, page);
    }
    x += width;
  }

  if (new_x)
    *new_x = x;

  return true;
}

void PDF::pdf_barcode_eanupc_aux(float x, float y,
                                 float x_width, float height,
                                 uint32_t color, GuardPattern guard_type,
                                 float *new_x, Page *page) {
  const int code = code_eanupc_aux_encoding[guard_type];

  for (int i = 5; i >= 0; i--) {
    int value = code >> i * 2 & 0x3;
    if (value) {
      if ((i & 0x1) == 0) {
        AddFilledRectangle(x, y, x_width * value,
                           height, 0, color,
                           PDF_TRANSPARENT, page);
      }
      x += x_width * value;
    }
  }
  if (new_x)
    *new_x = x;
}

namespace {
// Barcode routines need to switch the font to Courier temporarily.
// This object restores the font when it goes out of scope.
struct ScopedRestoreFont {
  ScopedRestoreFont(PDF *pdf) : pdf(pdf),
                                old_font(pdf->GetCurrentFont()) {
  }

  ~ScopedRestoreFont() {
    pdf->SetFont(old_font);
  }

  PDF *pdf = nullptr;
  const PDF::Font *old_font = nullptr;
};
}  // namespace

bool PDF::AddBarcodeEAN13(float x, float y, float width, float height,
                          const std::string &str, uint32_t color, Page *page) {
  if (str.empty()) {
    SetErr(-EINVAL, "Empty EAN13 barcode");
    return false;
  }

  const char *s = str.data();

  const size_t len = str.size();
  int lead = 0;
  if (len == 13) {
    const char ch = s[0];
    if (!isdigit(ch)) {
      SetErr(-EINVAL, std::format("Invalid EAN13 character {:c} 0x{:x}",
                                  ch, ch));
      return false;
    }
    lead = ch - '0';
    s++;

  } else if (len != 12) {
    SetErr(-EINVAL, std::format("Invalid EAN13 string length {}", len));
    return false;
  }

  /* Scale and calculate dimensions */
  float x_off, y_off, new_width, new_height, x_width, bar_height, bar_ext;
  float font;

  pdf_barcode_eanupc_calc_dims(PDF_BARCODE_EAN13, width, height, &x_off,
                               &y_off, &new_width, &new_height, &x_width,
                               &bar_height, &bar_ext, &font);

  x += x_off;
  y += y_off;
  float bar_y = y + new_height - bar_height;

  ScopedRestoreFont restore_font(this);
  // built-in monospace font.
  SetFont(COURIER);

  char text[2];
  text[1] = 0;
  text[0] = lead + '0';
  if (!AddText(text, font, x, y, color, page)) {
    return false;
  }

  x += eanupc_dimensions[0].quiet_left * x_width;

  pdf_barcode_eanupc_aux(x, bar_y - bar_ext, x_width,
                         bar_height + bar_ext, color, GUARD_NORMAL,
                         &x, page);

  for (int i = 0; i != 6; i++) {
    text[0] = *s;
    if (!AddTextWrap(text, font, x, y, 0, color,
                     7 * x_width, PDF_ALIGN_CENTER, nullptr, page)) {
      return false;
    }

    int set = (set_ean13_encoding[lead] & 1 << i) ? 1 : 0;
    if (!pdf_barcode_eanupc_ch(x, bar_y, x_width, bar_height,
                               color, *s, set, &x, page)) {
      return false;
    }
    s++;
  }

  pdf_barcode_eanupc_aux(x, bar_y - bar_ext, x_width,
                         bar_height + bar_ext, color, GUARD_CENTRE,
                         &x, page);

  for (int i = 0; i != 6; i++) {
    text[0] = *s;
    if (!AddTextWrap(text, font, x, y, 0, color,
                     7 * x_width, PDF_ALIGN_CENTER, nullptr, page)) {
      return false;
    }

    if (!pdf_barcode_eanupc_ch(x, bar_y, x_width, bar_height,
                               color, *s, 2, &x, page)) {
      return false;
    }
    s++;
  }

  pdf_barcode_eanupc_aux(x, bar_y - bar_ext, x_width,
                         bar_height + bar_ext, color, GUARD_NORMAL,
                         &x, page);

  // TODO: Cleaner to use the tables here than hard-code a width like 604.
  text[0] = '>';
  x += eanupc_dimensions[0].quiet_right * x_width -
    604.0f * font / (14.0f * 72.0f);
  if (!AddText(text, font, x, y, color, page)) {
    return false;
  }

  return true;
}

bool PDF::AddBarcodeUPCA(float x, float y, float width, float height,
                         const std::string &str, uint32_t color, Page *page) {
  const size_t len = str.size();
  if (len != 12) {
    SetErr(-EINVAL, std::format("Invalid UPC-A string length {}", len));
    return false;
  }

  /* Scale and calculate dimensions */
  float x_off, y_off, new_width, new_height;
  float x_width, bar_height, bar_ext, font;

  pdf_barcode_eanupc_calc_dims(PDF_BARCODE_UPCA, width, height, &x_off,
                               &y_off, &new_width, &new_height, &x_width,
                               &bar_height, &bar_ext, &font);

  x += x_off;
  y += y_off;
  float bar_y = y + new_height - bar_height;

  ScopedRestoreFont restore_font(this);
  // built-in monospace font.
  SetFont(COURIER);

  const char *string = str.data();

  char text[2];
  text[1] = 0;
  text[0] = *string;
  if (!AddText(text, font * 4.0f / 7.0f, x, y, color, page)) {
    return false;
  }

  x += eanupc_dimensions[1].quiet_left * x_width;
  pdf_barcode_eanupc_aux(x, bar_y - bar_ext, x_width,
                         bar_height + bar_ext, color, GUARD_NORMAL,
                         &x, page);

  for (int i = 0; i != 6; i++) {
    text[0] = *string;
    if (i) {
      if (!AddTextWrap(text, font, x, y, 0, color,
                       7 * x_width, PDF_ALIGN_CENTER, nullptr, page)) {
        return false;
      }
    }

    if (!pdf_barcode_eanupc_ch(x, bar_y - (i ? 0 : bar_ext),
                               x_width, bar_height + (i ? 0 : bar_ext),
                               color, *string, 0, &x, page)) {
      return false;
    }
    string++;
  }

  pdf_barcode_eanupc_aux(x, bar_y - bar_ext, x_width,
                         bar_height + bar_ext, color, GUARD_CENTRE,
                         &x, page);

  for (int i = 0; i != 6; i++) {
    text[0] = *string;
    if (i != 5) {
      if (!AddTextWrap(text, font, x, y, 0, color,
                       7 * x_width, PDF_ALIGN_CENTER, nullptr, page)) {
        return false;
      }
    }

    if (!pdf_barcode_eanupc_ch(
            x, bar_y - (i != 5 ? 0 : bar_ext), x_width,
            bar_height + (i != 5 ? 0 : bar_ext), color, *string, 2, &x,
            page)) {
      return false;
    }
    string++;
  }

  pdf_barcode_eanupc_aux(x, bar_y - bar_ext, x_width,
                         bar_height + bar_ext, color, GUARD_NORMAL,
                         &x, page);

  text[0] = *--string;

  x += eanupc_dimensions[1].quiet_right * x_width -
    604.0f * font * 4.0f / 7.0f / (14.0f * 72.0f);
  if (!AddText(text, font * 4.0f / 7.0f, x, y, color)) {
    return false;
  }

  return true;
}

bool PDF::AddBarcodeEAN8(float x, float y, float width, float height,
                         const std::string &str, uint32_t color,
                         Page *page) {

  const size_t len = str.size();
  if (len != 8) {
    SetErr(-EINVAL, std::format("Invalid EAN8 string length {}", len));
    return false;
  }

  const char *string = str.data();

  /* Scale and calculate dimensions */
  float x_off, y_off, new_width, new_height, x_width, bar_height, bar_ext;
  float font;

  pdf_barcode_eanupc_calc_dims(PDF_BARCODE_EAN8, width, height, &x_off,
                               &y_off, &new_width, &new_height, &x_width,
                               &bar_height, &bar_ext, &font);

  x += x_off;
  y += y_off;
  float bar_y = y + new_height - bar_height;

  ScopedRestoreFont restore_font(this);
  // built-in monospace font.
  SetFont(COURIER);

  char text[2];
  text[1] = 0;
  text[0] = '<';
  if (!AddText(text, font, x, y, color, page)) {
    return false;
  }

  x += eanupc_dimensions[2].quiet_left * x_width;
  pdf_barcode_eanupc_aux(x, bar_y - bar_ext, x_width,
                         bar_height + bar_ext, color, GUARD_NORMAL,
                         &x, page);

  for (int i = 0; i != 4; i++) {
    text[0] = *string;
    if (!AddTextWrap(text, font, x, y, 0, color,
                     7 * x_width, PDF_ALIGN_CENTER, nullptr, page)) {
      return false;
    }

    if (!pdf_barcode_eanupc_ch(x, bar_y, x_width, bar_height,
                               color, *string, 0, &x, page)) {
      return false;
    }
    string++;
  }

  pdf_barcode_eanupc_aux(x, bar_y - bar_ext, x_width,
                         bar_height + bar_ext, color, GUARD_CENTRE,
                         &x, page);

  for (int i = 0; i != 4; i++) {
    text[0] = *string;
    if (!AddTextWrap(text, font, x, y, 0, color,
                     7 * x_width, PDF_ALIGN_CENTER, nullptr, page)) {
      return false;
    }

    if (!pdf_barcode_eanupc_ch(x, bar_y, x_width, bar_height,
                               color, *string, 2, &x, page)) {
      return false;
    }
    string++;
  }

  pdf_barcode_eanupc_aux(x, bar_y - bar_ext, x_width,
                         bar_height + bar_ext, color, GUARD_NORMAL,
                         &x, page);

  text[0] = '>';
  x += eanupc_dimensions[0].quiet_right * x_width -
    604.0f * font / (14.0f * 72.0f);
  if (!AddText(text, font, x, y, color, page)) {
    return false;
  }

  return true;
}



bool PDF::AddBarcodeUPCE(float x, float y, float width, float height,
                         const std::string &str, uint32_t color, Page *page) {

  const size_t len = str.size();
  if (len != 12) {
    SetErr(-EINVAL,
           std::format("Invalid UPCE string length {}", len));
    return false;
  }

  if (str[0] != '0') {
    SetErr(-EINVAL, std::format("UPCE must start with 0; got {:c}", str[0]));
    return false;
  }

  for (size_t i = 0; i < len; i++) {
    if (!isdigit(str[i])) {
      SetErr(-EINVAL,
             std::format("Invalid UPCE char 0x{:x} at {}", str[i], i));
      return false;
    }
  }

  const char *string = str.data();

  /* Scale and calculate dimensions */
  float x_off, y_off, new_width, new_height;
  float x_width, bar_height, bar_ext, font;

  pdf_barcode_eanupc_calc_dims(PDF_BARCODE_UPCE, width, height, &x_off,
                               &y_off, &new_width, &new_height, &x_width,
                               &bar_height, &bar_ext, &font);

  x += x_off;
  y += y_off;
  float bar_y = y + new_height - bar_height;

  ScopedRestoreFont restore_font(this);
  // built-in monospace font.
  SetFont(COURIER);

  char text[2];
  text[1] = 0;
  text[0] = string[0];

  if (!AddText(text, font * 4.0f / 7.0f, x, y, color, page)) {
    return false;
  }

  x += eanupc_dimensions[2].quiet_left * x_width;
  pdf_barcode_eanupc_aux(x, bar_y, x_width, bar_height,
                         color, GUARD_NORMAL, &x, page);

  char X[6];
  if (string[5] && memcmp(string + 6, "0000", 4) == 0 &&
      '5' <= string[10] && string[10] <= '9') {
    memcpy(X, string + 1, 5);
    X[5] = string[10];
  } else if (string[4] && memcmp(string + 5, "00000", 5) == 0) {
    memcpy(X, string + 1, 4);
    X[4] = string[11];
    X[5] = 4;
  } else if ('0' <= string[3] && string[3] <= '2' &&
             memcmp(string + 4, "0000", 4) == 0) {
    X[0] = string[1];
    X[1] = string[2];
    X[2] = string[8];
    X[3] = string[9];
    X[4] = string[10];
    X[5] = string[3];
  } else if ('3' <= string[3] && string[3] <= '9' &&
             memcmp(string + 4, "00000", 5) == 0) {
    memcpy(X, string + 1, 3);
    X[3] = string[9];
    X[4] = string[10];
    X[5] = 3;
  } else {
    SetErr(-EINVAL, "Invalid UPCE string format");
    return false;
  }

  for (int i = 0; i != 6; i++) {
    text[0] = X[i];
    if (!AddTextWrap(text, font, x, y, 0, color,
                     7 * x_width, PDF_ALIGN_CENTER, nullptr, page)) {
      return false;
    }

    int set = (set_upce_encoding[string[11] - '0'] & 1 << i) ? 1 : 0;
    if (!pdf_barcode_eanupc_ch(x, bar_y, x_width, bar_height,
                               color, X[i], set, &x, page)) {
      return false;
    }
  }

  pdf_barcode_eanupc_aux(x, bar_y, x_width, bar_height,
                         color, GUARD_SPECIAL, &x, page);

  text[0] = string[11];
  x += eanupc_dimensions[0].quiet_right * x_width -
    604.0f * font * 4.0f / 7.0f / (14.0f * 72.0f);
  if (!AddText(text, font * 4.0f / 7.0f, x, y, color, page)) {
    return false;
  }

  return true;
}

bool PDF::AddQRCode(float x, float y, float size,
                    const std::string &str, uint32_t color,
                    Page *page) {
  ImageA img = QRCode::Text(str);

  float pixel = size / img.Width();

  for (int qy = 0; qy < img.Height(); qy++) {
    for (int qx = 0; qx < img.Width(); qx++) {
      if (img.GetPixel(qx, qy) == 0x00) {
        AddFilledRectangle(x + qx * pixel,
                           // Image is flipped relative to PDF
                           // coordinate system
                           size + y - (qy + 1) * pixel,
                           pixel, pixel,
                           0, color,
                           PDF_TRANSPARENT, page);
      }
    }
  }

  return true;
}


int PDF::AddBookmark(const std::string &name, int parent, Page *page) {
  if (!page)
    page = (Page *)FindLastObject(OBJ_page);

  if (!page) {
    SetErr(-EINVAL,
           "Unable to add bookmark; no pages available");
    return false;
  }

  Object *outline = nullptr;
  if (!(outline = FindFirstObject(OBJ_outline))) {
    outline = AddObject(new OutlineObj);
  }

  BookmarkObj *bobj = AddObject(new BookmarkObj);

  bobj->name = name;
  bobj->page = page;
  if (parent >= 0) {
    BookmarkObj *parent_obj = (BookmarkObj *)GetObject(parent);
    if (!parent_obj) {
      SetErr(-EINVAL, std::format("Invalid parent ID {} supplied", parent));
      return false;
    }
    bobj->parent = parent_obj;
    parent_obj->children.push_back(bobj);
  }

  return bobj->index;
}

bool PDF::AddLink(float x, float y,
                  float width, float height,
                  Page *target_page,
                  float target_x, float target_y,
                  Page *page) {
  if (!page)
    page = (Page*)FindLastObject(OBJ_page);

  if (!page) {
    SetErr(-EINVAL, "Unable to add link; no pages available");
    return false;
  }

  if (!target_page) {
    SetErr(-EINVAL, "Unable to link; no target page");
    return false;
  }

  LinkObj *lobj = AddObject(new LinkObj);
  lobj->target_page = target_page;
  lobj->target_x = target_x;
  lobj->target_y = target_y;
  lobj->llx = x;
  lobj->lly = y;
  lobj->urx = x + width;
  lobj->ury = y + height;
  page->annotations.push_back(lobj);

  return true;
}

void PDF::AddLine(float x1, float y1,
                  float x2, float y2,
                  float width, uint32_t color_rgb,
                  Page *page) {
  std::string str;
  AppendFormat(&str, "{} w\n", Float(width));
  AppendFormat(&str, "{} {} m\n", Float(x1), Float(y1));
  AppendFormat(&str, "/DeviceRGB CS\n");
  AppendFormat(&str, "{} {} {} RG\n",
               Float(PDF_RGB_R_FLOAT(color_rgb)),
               Float(PDF_RGB_G_FLOAT(color_rgb)),
               Float(PDF_RGB_B_FLOAT(color_rgb)));
  AppendFormat(&str, "{} {} l S\n",
               Float(x2),
               Float(y2));

  AppendDrawCommand(page, std::move(str));
}

void PDF::AddCubicBezier(float x1, float y1, float x2, float y2, float xq1,
                         float yq1, float xq2, float yq2, float width,
                         uint32_t color_rgb, Page *page) {
  std::string str;

  AppendFormat(&str, "{} w\n", Float(width));
  AppendFormat(&str, "{} {} m\n", Float(x1), Float(y1));
  AppendFormat(&str, "/DeviceRGB CS\n");
  AppendFormat(&str, "{} {} {} RG\n",
               Float(PDF_RGB_R_FLOAT(color_rgb)),
               Float(PDF_RGB_G_FLOAT(color_rgb)),
               Float(PDF_RGB_B_FLOAT(color_rgb)));
  AppendFormat(&str, "{} {} {} {} {} {} c S\n",
               Float(xq1),
               Float(yq1),
               Float(xq2),
               Float(yq2),
               Float(x2),
               Float(y2));

  AppendDrawCommand(page, std::move(str));
}

void PDF::AddQuadraticBezier(float x1, float y1, float x2, float y2,
                             float xq1, float yq1, float width,
                             uint32_t color_rgb, Page *page) {
  float xc1 = x1 + (xq1 - x1) * (2.0f / 3.0f);
  float yc1 = y1 + (yq1 - y1) * (2.0f / 3.0f);
  float xc2 = x2 + (xq1 - x2) * (2.0f / 3.0f);
  float yc2 = y2 + (yq1 - y2) * (2.0f / 3.0f);
  AddCubicBezier(x1, y1, x2, y2, xc1, yc1, xc2, yc2,
                 width, color_rgb, page);
}

void PDF::AddEllipse(float x, float y,
                     float xradius, float yradius,
                     float width,
                     uint32_t color, uint32_t fill_color,
                     Page *page) {
  std::string str;

  const float lx =
    (4.0f / 3.0f) * (float)(std::numbers::sqrt2 - 1.0f) * xradius;
  const float ly =
    (4.0f / 3.0f) * (float)(std::numbers::sqrt2 - 1.0f) * yradius;

  if (!PDF_IS_TRANSPARENT(fill_color)) {
    AppendFormat(&str, "/DeviceRGB CS\n");
    AppendFormat(&str, "{} {} {} rg\n",
                 Float(PDF_RGB_R_FLOAT(fill_color)),
                 Float(PDF_RGB_G_FLOAT(fill_color)),
                 Float(PDF_RGB_B_FLOAT(fill_color)));
  }

  /* stroke color */
  AppendFormat(&str, "/DeviceRGB CS\n");
  AppendFormat(&str, "{} {} {} RG\n",
               Float(PDF_RGB_R_FLOAT(color)),
               Float(PDF_RGB_G_FLOAT(color)),
               Float(PDF_RGB_B_FLOAT(color)));

  AppendFormat(&str, "{} w ", Float(width));

  // TODO: Original code had two decimal places here; we can use Float()?
  AppendFormat(&str, "{:.2f} {:.2f} m ", (x + xradius), (y));
  AppendFormat(&str,
               "{:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} c ", (x + xradius),
               (y - ly), (x + lx), (y - yradius), x, (y - yradius));
  AppendFormat(&str,
               "{:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} c ", (x - lx),
               (y - yradius), (x - xradius), (y - ly), (x - xradius), y);
  AppendFormat(&str,
               "{:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} c ", (x - xradius),
               (y + ly), (x - lx), (y + yradius), x, (y + yradius));
  AppendFormat(&str,
               "{:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} c ", (x + lx),
               (y + yradius), (x + xradius), (y + ly), (x + xradius), y);

  if (PDF_IS_TRANSPARENT(fill_color))
    AppendFormat(&str, "{}", "S ");
  else
    AppendFormat(&str, "{}", "B ");

  AppendDrawCommand(page, std::move(str));
}

void PDF::AddCircle(float xr, float yr, float radius, float width,
                    uint32_t color,
                    uint32_t fill_color, Page *page) {
  return AddEllipse(xr, yr, radius, radius, width, color,
                    fill_color, page);
}

void PDF::AddRectangle(float x, float y,
                       float width, float height, float border_width,
                       uint32_t color, Page *page) {
  std::string str;
  AppendFormat(&str, "{} {} {} RG ",
               Float(PDF_RGB_R_FLOAT(color)),
               Float(PDF_RGB_G_FLOAT(color)),
               Float(PDF_RGB_B_FLOAT(color)));

  AppendFormat(&str, "{} w ", Float(border_width));
  AppendFormat(&str, "{} {} {} {} re S",
               Float(x),
               Float(y),
               Float(width),
               Float(height));

  AppendDrawCommand(page, str);
}

void PDF::AddFilledRectangle(
    float x, float y, float width, float height,
    float border_width, uint32_t color_fill,
    uint32_t color_border, Page *page) {

  std::string str;

  AppendFormat(&str, "{} {} {} rg ",
               Float(PDF_RGB_R_FLOAT(color_fill)),
               Float(PDF_RGB_G_FLOAT(color_fill)),
               Float(PDF_RGB_B_FLOAT(color_fill)));

  if (border_width > 0) {
    AppendFormat(&str, "{} {} {} RG ",
                 Float(PDF_RGB_R_FLOAT(color_border)),
                 Float(PDF_RGB_G_FLOAT(color_border)),
                 Float(PDF_RGB_B_FLOAT(color_border)));
    AppendFormat(&str, "{} w ", Float(border_width));
    AppendFormat(&str, "{} {} {} {} re B",
                  Float(x), Float(y),
                  Float(width), Float(height));
  } else {
    AppendFormat(&str, "{} {} {} {} re f",
                 Float(x), Float(y),
                 Float(width), Float(height));
  }

  AppendDrawCommand(page, str);
}

bool PDF::AddCustomPath(const std::vector<PathOp> &ops,
                        float stroke_width,
                        uint32_t stroke_color,
                        uint32_t fill_color,
                        Page *page) {

  std::string str;

  // TODO: Use Float() in here.
  if (!PDF_IS_TRANSPARENT(fill_color)) {
    AppendFormat(&str, "/DeviceRGB CS\n");
    AppendFormat(&str, "{} {} {} rg\n",
                  PDF_RGB_R_FLOAT(fill_color),
                  PDF_RGB_G_FLOAT(fill_color),
                  PDF_RGB_B_FLOAT(fill_color));
  }

  AppendFormat(&str, "{} w\n", stroke_width);
  AppendFormat(&str, "/DeviceRGB CS\n");
  AppendFormat(&str, "{} {} {} RG\n",
               PDF_RGB_R_FLOAT(stroke_color),
               PDF_RGB_G_FLOAT(stroke_color),
               PDF_RGB_B_FLOAT(stroke_color));

  for (PathOp operation : ops) {
    switch (operation.op) {
    case 'm':
      AppendFormat(&str, "{} {} m\n", operation.x1, operation.y1);
      break;
    case 'l':
      AppendFormat(&str, "{} {} l\n", operation.x1, operation.y1);
      break;
    case 'c':
      AppendFormat(&str, "{} {} {} {} {} {} c\n", operation.x1,
                   operation.y1, operation.x2, operation.y2,
                   operation.x3, operation.y3);
      break;
    case 'v':
      AppendFormat(&str, "{} {} {} {} v\n", operation.x1, operation.y1,
                   operation.x2, operation.y2);
      break;
    case 'y':
      AppendFormat(&str, "{} {} {} {} y\n", operation.x1, operation.y1,
                   operation.x2, operation.y2);
      break;
    case 'h':
      AppendFormat(&str, "h\n");
      break;
    default:
      SetErr(-errno, "Invalid operation");
      return false;
    }
  }

  if (PDF_IS_TRANSPARENT(fill_color))
    AppendFormat(&str, "{}", "S ");
  else
    AppendFormat(&str, "{}", "B ");

  AppendDrawCommand(page, std::move(str));

  return true;
}

bool PDF::AddPolygon(
    const std::vector<std::pair<float, float>> &points,
    float border_width, uint32_t color,
    Page *page) {
  if (points.empty()) return false;

  std::string str;

  // TODO: Use Float() in here.
  AppendFormat(&str, "{} {} {} RG ",
               PDF_RGB_R_FLOAT(color),
               PDF_RGB_G_FLOAT(color),
               PDF_RGB_B_FLOAT(color));
  AppendFormat(&str, "{} w ", border_width);
  AppendFormat(&str, "{} {} m ", points[0].first, points[0].second);
  for (int i = 1; i < (int)points.size(); i++) {
    AppendFormat(&str, "{} {} l ", points[i].first, points[i].second);
  }
  AppendFormat(&str, "h S");

  AppendDrawCommand(page, str);
  return true;
}

bool PDF::AddFilledPolygon(
    const std::vector<std::pair<float, float>> &points,
    float border_width, uint32_t color,
    Page *page) {
  if (points.empty()) return false;

  std::string str;

  // TODO: Use Float() in here.
  AppendFormat(&str, "{} {} {} RG ",
               PDF_RGB_R_FLOAT(color),
               PDF_RGB_G_FLOAT(color),
               PDF_RGB_B_FLOAT(color));
  AppendFormat(&str, "{} {} {} rg ",
               PDF_RGB_R_FLOAT(color), PDF_RGB_G_FLOAT(color),
               PDF_RGB_B_FLOAT(color));
  AppendFormat(&str, "{} w ", border_width);
  AppendFormat(&str, "{} {} m ", points[0].first, points[0].second);
  for (int i = 1; i < (int)points.size(); i++) {
    AppendFormat(&str, "{} {} l ", points[i].first, points[i].second);
  }
  AppendFormat(&str, "h f");

  AppendDrawCommand(page, std::move(str));
  return true;
}

// Convert to a vector of CIDs (which are glyphs for unicode encoding).
// If the text is invalid, returns nullopt.
// If codepoints can't be encoded, they will result in CID 0.
// For WIN_ANSI encoding, the results will all be 8-bit.
static std::optional<std::vector<uint16_t>> TextToCIDs(
    std::string_view utf8_text,
    FontEncoding encoding,
    const std::unordered_map<uint32_t, uint16_t> &cmap) {
  std::vector<uint16_t> ret;
  // Typical string is ascii, so one utf-8 byte is one CID.
  ret.reserve(utf8_text.size());

  if (encoding == FontEncoding::WIN_ANSI) {
    while (!utf8_text.empty()) {
      uint8_t pdf_char = 0;
      const int code_len =
        UTF8ToWinAnsiEncoding(utf8_text.data(), utf8_text.size(), &pdf_char);
      if (code_len < 0) {
        return std::nullopt;
      }
      utf8_text.remove_prefix(code_len);

      ret.push_back(pdf_char);
    }

  } else {

    while (!utf8_text.empty()) {
      uint32_t codepoint = 0;
      int code_len = utf8_to_utf32(utf8_text.data(), utf8_text.size(),
                                   &codepoint);
      if (code_len < 0) {
        return std::nullopt;
      }
      utf8_text.remove_prefix(code_len);

      if (auto it = cmap.find(codepoint); it != cmap.end()) {
        ret.push_back(it->second);
      } else {
        // Missing a glyph; we use .notdef for this.
        ret.push_back(0);
      }
    }

  }

  return {ret};
}



// Encode the text using the given encoding, including the surrounding
// () or <> characters as appropriate.
static std::optional<std::string> EncodePDFText(
    std::string_view text,
    FontEncoding encoding,
    const std::unordered_map<uint32_t, uint16_t> &cmap) {
  if (encoding == FontEncoding::WIN_ANSI) {
    return std::format("{}", PDFDocEncodeString(text));

  } else {
    std::string ret = "<";
    ret.reserve(text.size() * 4 + 2);
    while (!text.empty()) {
      uint32_t codepoint = 0;
      int code_len = utf8_to_utf32(text.data(), text.size(), &codepoint);
      if (code_len < 0) return std::nullopt;
      text.remove_prefix(code_len);

      CHECK(codepoint <= 0xFFFF) << "Only 16-bit codepoints are "
        "currently supported. You need something like 'surrogate pairs' "
        "here. Got: U+" << std::format("{:08x}", codepoint);
      if (auto it = cmap.find(codepoint); it != cmap.end()) {
        AppendFormat(&ret, "{:04x}", it->second);
      } else {
        if (true || VERBOSE) {
          printf("Missing glyph for U+%04x\n", codepoint);
        }
        AppendFormat(&ret, "0000");
      }
    }
    ret.push_back('>');
    if (VERBOSE) {
      printf("Encoded: %s\n", ret.c_str());
    }
    return ret;
  }
}

static void SetTextPositionAndAngle(std::string *str,
                                    float xoff, float yoff, float angle) {
  if (angle != 0.0f) {
    AppendFormat(str, "{} {} {} {} {} {} Tm ",
                 Float(cosf(angle)),
                 Float(sinf(angle)),
                 Float(-sinf(angle)),
                 Float(cosf(angle)),
                 Float(xoff),
                 Float(yoff));
  } else {
    AppendFormat(str, "{} {} TD ", Float(xoff), Float(yoff));
  }
}

bool PDF::pdf_add_text_spacing(const std::string &text, float size, float xoff,
                               float yoff, uint32_t color, float spacing,
                               float angle, Page *page) {
  const int alpha = (color >> 24) >> 4;

  if (text.empty())
    return true;

  std::string str = "BT ";

  AppendFormat(&str, "/GS{} gs ", alpha);
  SetTextPositionAndAngle(&str, xoff, yoff, angle);

  AppendFormat(&str, "/F{} {} Tf ", current_font->font_index,
               Float(size));

  uint8_t r = PDF_RGB_R(color);
  uint8_t g = PDF_RGB_G(color);
  uint8_t b = PDF_RGB_B(color);

  // If greyscale, we can make this smaller with "0 g" to "1 g" (white).
  if (r == g && r == b) {
    if (r == 0) {
      AppendFormat(&str, "0 g ");
    } else {
      AppendFormat(&str, "{} g ",
                   Float(PDF_COLOR_FLOAT(r)));
    }
  } else {
    AppendFormat(&str, "{} {} {} rg ",
                 Float(PDF_COLOR_FLOAT(r)),
                 Float(PDF_COLOR_FLOAT(g)),
                 Float(PDF_COLOR_FLOAT(b)));
  }
  AppendFormat(&str, "{} Tc ", Float(spacing));

  if (const std::optional<std::string> encoded_text =
      EncodePDFText(text, current_font->encoding,
                    current_font->glyph_from_codepoint)) {
    str.append(encoded_text.value());
  } else {
    SetErr(-EINVAL, "Could not encode text in PDF encoding.");
    return false;
  }

  AppendFormat(&str,
               " Tj "
               "ET");

  AppendDrawCommand(page, std::move(str));
  return true;
}

bool PDF::AddText(const std::string &text, float size,
                  float xoff, float yoff,
                  uint32_t color, Page *page) {
  return pdf_add_text_spacing(text, size, xoff, yoff, color, 0, 0, page);
}

bool PDF::AddTextRotate(const std::string &text,
                        float size, float xoff, float yoff,
                        float angle, uint32_t color, Page *page) {
  return pdf_add_text_spacing(text, size, xoff, yoff, color, 0, angle, page);
}

bool PDF::AddSpacedLine(const SpacedLine &line,
                        float size,
                        float xoff, float yoff,
                        uint32_t color,
                        float angle,
                        Page *page) {
  const int alpha = (color >> 24) >> 4;

  if (line.empty())
    return true;

  std::string str = "BT ";

  AppendFormat(&str, "/GS{} gs ", alpha);
  SetTextPositionAndAngle(&str, xoff, yoff, angle);

  AppendFormat(&str, "/F{} {} Tf ", current_font->font_index,
               Float(size));
  AppendFormat(&str, "{} {} {} rg ",
               Float(PDF_RGB_R_FLOAT(color)),
               Float(PDF_RGB_G_FLOAT(color)),
               Float(PDF_RGB_B_FLOAT(color)));
  // AppendFormat(&str, "{} Tc ", spacing);

  AppendFormat(&str, "[ ");

  // PDF uses 1/1000ths of a unit here, and defaults to negative space
  // (this is typically used for kerning). This is independent of the
  // font size.
  const double gap_scale = -1000.0;

  for (int i = 0; i < (int)line.size(); i++) {
    const auto &[text, gap] = line[i];
    if (const std::optional<std::string> encoded_text =
        EncodePDFText(text, current_font->encoding,
                      current_font->glyph_from_codepoint)) {
      AppendFormat(&str, "{} ", encoded_text.value());
    } else {
      SetErr(-EINVAL, "Could not encode text in PDF encoding.");
      return false;
    }

    // The last spacing is ignored (and the syntax does not allow it).
    if (i != (int)line.size() - 1) {
      AppendFormat(&str, "{} ", Float(gap * gap_scale));
    }
  }
  AppendFormat(&str,
               "] TJ ET");

  AppendDrawCommand(page, std::move(str));
  return true;
}

// TODO: Use BoxesAndGlue library for this.
std::vector<PDF::SpacedLine> PDF::SpaceLines(const std::string &text,
                                             double line_width,
                                             const Font *font) const {
  static constexpr bool LOCAL_VERBOSE = false;

  if (font == nullptr) font = current_font;
  CHECK(current_font);

  // First, split into words with their widths.
  std::vector<std::string> words = Util::Tokens(text, Util::IsWhitespace);
  std::vector<double> sizes;
  sizes.reserve(words.size());

  const double space_width = font->CharWidth(' ');
  for (int i = 0; i < (int)words.size(); i++) {
    sizes.push_back(font->GetKernedWidth(words[i]));
  }

  if (LOCAL_VERBOSE) {
    printf("\n"
           ABGCOLOR(60, 60, 180,
                    AFGCOLOR(255, 255, 255, "=== SpaceLines ===")) "\n"
           "Space into " ABLUE("%.6f") ":  (' ' is " AYELLOW("%.6f") ")\n"
           "[", line_width, space_width);
    double total = 0.0;
    for (int i = 0; i < (int)words.size(); i++) {
      printf(AWHITE("%s") " " APURPLE("%.4f") " ",
             words[i].c_str(), sizes[i]);
      total += sizes[i];
    }
    printf("]\n"
           "Total word width: " ACYAN("%.6f")
           " w/ space: " AYELLOW("%.6f") "\n",
           total, total + ((int)words.size() - 1) * space_width);
  }

  // Value is a pair: The penalty, and a bool indicating that it is
  // better to break after the word.
  std::unordered_map<std::pair<int, int>, std::pair<double, bool>,
    Hashing<std::pair<int, int>>> memo_table;

  // Same arguments as the penalty function below. Gets the width
  // of the text up to and including word_idx on this line.
  auto GetWidthBefore = [&](int word_idx, int words_before) -> double {
      // Otherwise, solve recursively.
      double width_used = 0.0;
      const int before_start = word_idx - words_before;
      CHECK(before_start >= 0);
      for (int b = 0; b < words_before; b++) {
        // Add the word's length and the space after it.
        CHECK(before_start + b < (int)sizes.size());
        width_used += sizes[before_start + b];
        width_used += space_width;
      }
      return width_used;
    };

  CHECK(words.size() == sizes.size());

  // Since the recursion depth can get kinda high here, we need to solve
  // this one bottom-up.
  for (int word_idx = (int)words.size() - 1; word_idx >= 0; word_idx--) {
    // As a loop invariant, we have memo_table filled in for every greater
    // word_idx.

    auto Get = [&words, &memo_table](int w, int b) {
        // Base case is no penalty; no breaks.
        if (w >= (int)words.size()) return std::make_pair(0.0, false);
        auto mit = memo_table.find(std::make_pair(w, b));
        CHECK(mit != memo_table.end()) << "Later table entries should "
          "be filled in!" << w << "," << b;
        return mit->second;
      };

    auto Set = [&words,
                &memo_table](int w, int b, double p, bool brk) {
        CHECK(!memo_table.contains(std::make_pair(w, b))) <<
          "Duplicate entries?";
        if (LOCAL_VERBOSE) {
          if (w < (int)words.size()) {
            printf(
                "  Penalty ..%d.. [" AWHITE("%s") "] = " ARED("%.4f") " %s\n",
                b, words[w].c_str(), p, brk ? AYELLOW("break") : "no");
          }
        }
        memo_table[std::make_pair(w, b)] =
          std::make_pair(p, brk);
      };

    // PERF: We can (and should) cut this off once we exceed the
    // length of a line.
    for (int words_before = 0; words_before <= word_idx; words_before++) {

      if (LOCAL_VERBOSE) {
        printf("[%d,%d] Check", word_idx, words_before);
        const int before_start = word_idx - words_before;
        CHECK(before_start >= 0);
        for (int b = 0; b < words_before; b++) {
          // Add the word's length and the space after it.
          CHECK(before_start + b < (int)words.size());
          printf(" " AGREY("%s"), words[before_start + b].c_str());
        }
        printf(" " AWHITE("%s") "\n", words[word_idx].c_str());
      }

      // PERF: Can compute this incrementally in the loop.
      CHECK(word_idx >= 0 && word_idx < (int)sizes.size()) << word_idx;
      // This includes trailing space, unless we're at the beginning of
      // the line.
      const double width_before = GetWidthBefore(word_idx, words_before);
      // And always add the word.
      CHECK(word_idx < (int)sizes.size()) << word_idx
                                          << " vs " << sizes.size();
      const double width_word = sizes[word_idx];

      double penalty_word = 0.0;
      // Add the penalty for the word, which applies whether we break
      // or not.
      const double total_width = width_before + width_word;
      if (LOCAL_VERBOSE) {
        printf("  %.4f > %.4f? ",
               total_width, line_width);
      }
      if (total_width > line_width) {
        if (LOCAL_VERBOSE) {
          printf(ABGCOLOR(255,0,0, "OVER"));
        }
        if (width_before > line_width) {
          // We were already over. So just add the word's size.
          // This might be wrong wrt the trailing space, although
          // in this case these details just amount to tweaks to the
          // multiplier.
          penalty_word += width_word;
          if (LOCAL_VERBOSE) printf(" full penalty %.3f", width_word);
        } else {
          // Since this is the word that puts us over, only count
          // the amount that it's over.
          penalty_word += (total_width - line_width);
          if (LOCAL_VERBOSE) printf(" part penalty %.3f", penalty_word);
        }
      }
      // But the overage penalty is scaled.
      if (penalty_word > 0.0) {
        penalty_word = (1.0 + penalty_word);
        penalty_word = penalty_word * penalty_word * penalty_word;
      }
      if (LOCAL_VERBOSE) {
        printf("\n");
      }


      // Now we can either break here, or continue.
      // If we break, then the penalty is the amount of space left.

      const double penalty_slack = std::max(line_width - total_width, 0.0);
      // ... plus the penalty for the remainder, starting on a new line.
      const double p_rest = Get(word_idx + 1, 0).first;

      const double penalty_break = penalty_word + penalty_slack + p_rest;

      if (LOCAL_VERBOSE) {
        printf("  width used "
               ABLUE("%.4f") ". Word penalty " APURPLE("%.4f") ".\n"
               "    w/break " AORANGE("%.4f")
               " " AGREY("(slack)") " + " AYELLOW("%.4f")
               " " AGREY("(rest)") " = " ARED("%.4f") "\n",
               total_width, penalty_word,
               penalty_slack, p_rest, penalty_break);
      }

      // The case where we do not break.
      if (true || total_width < line_width) {
        const double p_rest_nobreak = Get(word_idx + 1, words_before + 1).first;
        const double penalty_nobreak = penalty_word + p_rest_nobreak;
        if (LOCAL_VERBOSE) {
          printf("    or without break: "
                 AGREEN("%.4f") " = " ARED("%.4f") "\n",
                 p_rest_nobreak, penalty_nobreak);
        }

        if (penalty_break < penalty_nobreak) {
          Set(word_idx, words_before, penalty_break, true);
        } else {
          Set(word_idx, words_before, penalty_nobreak, false);
        }

      } else {
        Set(word_idx, words_before, penalty_break, true);
      }
    }
  }

  // Now lay it out using the memo table we already computed.
  std::vector<SpacedLine> ret;
  int before = 0;
  SpacedLine current_line;
  for (int w = 0; w < (int)words.size(); w++) {
    // Get the data from the table.
    const auto mit = memo_table.find(std::make_pair(w, before));
    CHECK(mit != memo_table.end()) << "Bug: This should have been computed by "
      "the recursive procedure above; we're retracing its steps here: " <<
      w << "," << before;

    // We always put the word.
    if (before > 0) {
      CHECK(!current_line.empty());
      // Add the space char; no kerning here.
      current_line.emplace_back(" ", 0);
    } else {
      CHECK(current_line.empty());
    }

    SpacedLine spaced_word = font->KernText(words[w]);
    for (const auto &part : spaced_word)
      current_line.push_back(part);

    // Now, do we break or not?
    const auto &[p, brk] = mit->second;
    if (LOCAL_VERBOSE) {
      printf("After [" AWHITE("%s") "]? Penalty " ARED("%.4f") " %s\n",
             words[w].c_str(), p, brk ? AYELLOW("break") : AGREY("no"));
    }
    if (brk) {
      // current_line.push_back(std::make_pair("\\n", 0.0f));
      ret.push_back(std::move(current_line));
      current_line.clear();
      before = 0;
    } else {
      before++;
    }
  }

  if (!current_line.empty())
    ret.push_back(std::move(current_line));

  return ret;
}


// The width of each character, in points, at size 14.
static constexpr const uint16_t helvetica_widths[256] = {
  280, 280, 280, 280,  280, 280, 280, 280,  280,  280, 280,  280, 280,
  280, 280, 280, 280,  280, 280, 280, 280,  280,  280, 280,  280, 280,
  280, 280, 280, 280,  280, 280, 280, 280,  357,  560, 560,  896, 672,
  192, 335, 335, 392,  588, 280, 335, 280,  280,  560, 560,  560, 560,
  560, 560, 560, 560,  560, 560, 280, 280,  588,  588, 588,  560, 1023,
  672, 672, 727, 727,  672, 615, 784, 727,  280,  504, 672,  560, 839,
  727, 784, 672, 784,  727, 672, 615, 727,  672,  951, 672,  672, 615,
  280, 280, 280, 472,  560, 335, 560, 560,  504,  560, 560,  280, 560,
  560, 223, 223, 504,  223, 839, 560, 560,  560,  560, 335,  504, 280,
  560, 504, 727, 504,  504, 504, 336, 262,  336,  588, 352,  560, 352,
  223, 560, 335, 1008, 560, 560, 335, 1008, 672,  335, 1008, 352, 615,
  352, 352, 223, 223,  335, 335, 352, 560,  1008, 335, 1008, 504, 335,
  951, 352, 504, 672,  280, 335, 560, 560,  560,  560, 262,  560, 335,
  742, 372, 560, 588,  335, 742, 335, 403,  588,  335, 335,  335, 560,
  541, 280, 335, 335,  367, 560, 840, 840,  840,  615, 672,  672, 672,
  672, 672, 672, 1008, 727, 672, 672, 672,  672,  280, 280,  280, 280,
  727, 727, 784, 784,  784, 784, 784, 588,  784,  727, 727,  727, 727,
  672, 672, 615, 560,  560, 560, 560, 560,  560,  896, 504,  560, 560,
  560, 560, 280, 280,  280, 280, 560, 560,  560,  560, 560,  560, 560,
  588, 615, 560, 560,  560, 560, 504, 560,  504,
};

static constexpr const uint16_t helvetica_bold_widths[256] = {
  280,  280, 280,  280, 280, 280, 280, 280,  280, 280, 280, 280,  280, 280,
  280,  280, 280,  280, 280, 280, 280, 280,  280, 280, 280, 280,  280, 280,
  280,  280, 280,  280, 280, 335, 477, 560,  560, 896, 727, 239,  335, 335,
  392,  588, 280,  335, 280, 280, 560, 560,  560, 560, 560, 560,  560, 560,
  560,  560, 335,  335, 588, 588, 588, 615,  982, 727, 727, 727,  727, 672,
  615,  784, 727,  280, 560, 727, 615, 839,  727, 784, 672, 784,  727, 672,
  615,  727, 672,  951, 672, 672, 615, 335,  280, 335, 588, 560,  335, 560,
  615,  560, 615,  560, 335, 615, 615, 280,  280, 560, 280, 896,  615, 615,
  615,  615, 392,  560, 335, 615, 560, 784,  560, 560, 504, 392,  282, 392,
  588,  352, 560,  352, 280, 560, 504, 1008, 560, 560, 335, 1008, 672, 335,
  1008, 352, 615,  352, 352, 280, 280, 504,  504, 352, 560, 1008, 335, 1008,
  560,  335, 951,  352, 504, 672, 280, 335,  560, 560, 560, 560,  282, 560,
  335,  742, 372,  560, 588, 335, 742, 335,  403, 588, 335, 335,  335, 615,
  560,  280, 335,  335, 367, 560, 840, 840,  840, 615, 727, 727,  727, 727,
  727,  727, 1008, 727, 672, 672, 672, 672,  280, 280, 280, 280,  727, 727,
  784,  784, 784,  784, 784, 588, 784, 727,  727, 727, 727, 672,  672, 615,
  560,  560, 560,  560, 560, 560, 896, 560,  560, 560, 560, 560,  280, 280,
  280,  280, 615,  615, 615, 615, 615, 615,  615, 588, 615, 615,  615, 615,
  615,  560, 615,  560,
};

static constexpr const uint16_t helvetica_bold_oblique_widths[256] = {
  280,  280, 280,  280, 280, 280, 280, 280,  280, 280, 280, 280,  280, 280,
  280,  280, 280,  280, 280, 280, 280, 280,  280, 280, 280, 280,  280, 280,
  280,  280, 280,  280, 280, 335, 477, 560,  560, 896, 727, 239,  335, 335,
  392,  588, 280,  335, 280, 280, 560, 560,  560, 560, 560, 560,  560, 560,
  560,  560, 335,  335, 588, 588, 588, 615,  982, 727, 727, 727,  727, 672,
  615,  784, 727,  280, 560, 727, 615, 839,  727, 784, 672, 784,  727, 672,
  615,  727, 672,  951, 672, 672, 615, 335,  280, 335, 588, 560,  335, 560,
  615,  560, 615,  560, 335, 615, 615, 280,  280, 560, 280, 896,  615, 615,
  615,  615, 392,  560, 335, 615, 560, 784,  560, 560, 504, 392,  282, 392,
  588,  352, 560,  352, 280, 560, 504, 1008, 560, 560, 335, 1008, 672, 335,
  1008, 352, 615,  352, 352, 280, 280, 504,  504, 352, 560, 1008, 335, 1008,
  560,  335, 951,  352, 504, 672, 280, 335,  560, 560, 560, 560,  282, 560,
  335,  742, 372,  560, 588, 335, 742, 335,  403, 588, 335, 335,  335, 615,
  560,  280, 335,  335, 367, 560, 840, 840,  840, 615, 727, 727,  727, 727,
  727,  727, 1008, 727, 672, 672, 672, 672,  280, 280, 280, 280,  727, 727,
  784,  784, 784,  784, 784, 588, 784, 727,  727, 727, 727, 672,  672, 615,
  560,  560, 560,  560, 560, 560, 896, 560,  560, 560, 560, 560,  280, 280,
  280,  280, 615,  615, 615, 615, 615, 615,  615, 588, 615, 615,  615, 615,
  615,  560, 615,  560,
};

static constexpr const uint16_t helvetica_oblique_widths[256] = {
  280, 280, 280, 280,  280, 280, 280, 280,  280,  280, 280,  280, 280,
  280, 280, 280, 280,  280, 280, 280, 280,  280,  280, 280,  280, 280,
  280, 280, 280, 280,  280, 280, 280, 280,  357,  560, 560,  896, 672,
  192, 335, 335, 392,  588, 280, 335, 280,  280,  560, 560,  560, 560,
  560, 560, 560, 560,  560, 560, 280, 280,  588,  588, 588,  560, 1023,
  672, 672, 727, 727,  672, 615, 784, 727,  280,  504, 672,  560, 839,
  727, 784, 672, 784,  727, 672, 615, 727,  672,  951, 672,  672, 615,
  280, 280, 280, 472,  560, 335, 560, 560,  504,  560, 560,  280, 560,
  560, 223, 223, 504,  223, 839, 560, 560,  560,  560, 335,  504, 280,
  560, 504, 727, 504,  504, 504, 336, 262,  336,  588, 352,  560, 352,
  223, 560, 335, 1008, 560, 560, 335, 1008, 672,  335, 1008, 352, 615,
  352, 352, 223, 223,  335, 335, 352, 560,  1008, 335, 1008, 504, 335,
  951, 352, 504, 672,  280, 335, 560, 560,  560,  560, 262,  560, 335,
  742, 372, 560, 588,  335, 742, 335, 403,  588,  335, 335,  335, 560,
  541, 280, 335, 335,  367, 560, 840, 840,  840,  615, 672,  672, 672,
  672, 672, 672, 1008, 727, 672, 672, 672,  672,  280, 280,  280, 280,
  727, 727, 784, 784,  784, 784, 784, 588,  784,  727, 727,  727, 727,
  672, 672, 615, 560,  560, 560, 560, 560,  560,  896, 504,  560, 560,
  560, 560, 280, 280,  280, 280, 560, 560,  560,  560, 560,  560, 560,
  588, 615, 560, 560,  560, 560, 504, 560,  504,
};

static constexpr const uint16_t symbol_widths[256] = {
  252, 252, 252, 252,  252, 252, 252,  252, 252,  252,  252, 252, 252, 252,
  252, 252, 252, 252,  252, 252, 252,  252, 252,  252,  252, 252, 252, 252,
  252, 252, 252, 252,  252, 335, 718,  504, 553,  839,  784, 442, 335, 335,
  504, 553, 252, 553,  252, 280, 504,  504, 504,  504,  504, 504, 504, 504,
  504, 504, 280, 280,  553, 553, 553,  447, 553,  727,  672, 727, 616, 615,
  769, 607, 727, 335,  636, 727, 691,  896, 727,  727,  774, 746, 560, 596,
  615, 695, 442, 774,  650, 801, 615,  335, 869,  335,  663, 504, 504, 636,
  553, 553, 497, 442,  525, 414, 607,  331, 607,  553,  553, 580, 525, 553,
  553, 525, 553, 607,  442, 580, 718,  691, 496,  691,  497, 483, 201, 483,
  553, 0,   0,   0,    0,   0,   0,    0,   0,    0,    0,   0,   0,   0,
  0,   0,   0,   0,    0,   0,   0,    0,   0,    0,    0,   0,   0,   0,
  0,   0,   0,   0,    0,   0,   756,  624, 248,  553,  168, 718, 504, 759,
  759, 759, 759, 1050, 994, 607, 994,  607, 403,  553,  414, 553, 553, 718,
  497, 463, 553, 553,  553, 553, 1008, 607, 1008, 663,  829, 691, 801, 994,
  774, 774, 829, 774,  774, 718, 718,  718, 718,  718,  718, 718, 774, 718,
  796, 796, 897, 829,  553, 252, 718,  607, 607,  1050, 994, 607, 994, 607,
  497, 331, 796, 796,  792, 718, 387,  387, 387,  387,  387, 387, 497, 497,
  497, 497, 0,   331,  276, 691, 691,  691, 387,  387,  387, 387, 387, 387,
  497, 497, 497, 0,
};

static constexpr const uint16_t times_roman_widths[256] = {
  252, 252, 252, 252, 252, 252, 252, 252,  252, 252, 252, 252,  252, 252,
  252, 252, 252, 252, 252, 252, 252, 252,  252, 252, 252, 252,  252, 252,
  252, 252, 252, 252, 252, 335, 411, 504,  504, 839, 784, 181,  335, 335,
  504, 568, 252, 335, 252, 280, 504, 504,  504, 504, 504, 504,  504, 504,
  504, 504, 280, 280, 568, 568, 568, 447,  928, 727, 672, 672,  727, 615,
  560, 727, 727, 335, 392, 727, 615, 896,  727, 727, 560, 727,  672, 560,
  615, 727, 727, 951, 727, 727, 615, 335,  280, 335, 472, 504,  335, 447,
  504, 447, 504, 447, 335, 504, 504, 280,  280, 504, 280, 784,  504, 504,
  504, 504, 335, 392, 280, 504, 504, 727,  504, 504, 447, 483,  201, 483,
  545, 352, 504, 352, 335, 504, 447, 1008, 504, 504, 335, 1008, 560, 335,
  896, 352, 615, 352, 352, 335, 335, 447,  447, 352, 504, 1008, 335, 987,
  392, 335, 727, 352, 447, 727, 252, 335,  504, 504, 504, 504,  201, 504,
  335, 766, 278, 504, 568, 335, 766, 335,  403, 568, 302, 302,  335, 504,
  456, 252, 335, 302, 312, 504, 756, 756,  756, 447, 727, 727,  727, 727,
  727, 727, 896, 672, 615, 615, 615, 615,  335, 335, 335, 335,  727, 727,
  727, 727, 727, 727, 727, 568, 727, 727,  727, 727, 727, 727,  560, 504,
  447, 447, 447, 447, 447, 447, 672, 447,  447, 447, 447, 447,  280, 280,
  280, 280, 504, 504, 504, 504, 504, 504,  504, 568, 504, 504,  504, 504,
  504, 504, 504, 504,
};

static constexpr const uint16_t times_bold_widths[256] = {
  252, 252, 252, 252,  252, 252, 252, 252,  252,  252,  252,  252,  252,
  252, 252, 252, 252,  252, 252, 252, 252,  252,  252,  252,  252,  252,
  252, 252, 252, 252,  252, 252, 252, 335,  559,  504,  504,  1008, 839,
  280, 335, 335, 504,  574, 252, 335, 252,  280,  504,  504,  504,  504,
  504, 504, 504, 504,  504, 504, 335, 335,  574,  574,  574,  504,  937,
  727, 672, 727, 727,  672, 615, 784, 784,  392,  504,  784,  672,  951,
  727, 784, 615, 784,  727, 560, 672, 727,  727,  1008, 727,  727,  672,
  335, 280, 335, 585,  504, 335, 504, 560,  447,  560,  447,  335,  504,
  560, 280, 335, 560,  280, 839, 560, 504,  560,  560,  447,  392,  335,
  560, 504, 727, 504,  504, 447, 397, 221,  397,  524,  352,  504,  352,
  335, 504, 504, 1008, 504, 504, 335, 1008, 560,  335,  1008, 352,  672,
  352, 352, 335, 335,  504, 504, 352, 504,  1008, 335,  1008, 392,  335,
  727, 352, 447, 727,  252, 335, 504, 504,  504,  504,  221,  504,  335,
  752, 302, 504, 574,  335, 752, 335, 403,  574,  302,  302,  335,  560,
  544, 252, 335, 302,  332, 504, 756, 756,  756,  504,  727,  727,  727,
  727, 727, 727, 1008, 727, 672, 672, 672,  672,  392,  392,  392,  392,
  727, 727, 784, 784,  784, 784, 784, 574,  784,  727,  727,  727,  727,
  727, 615, 560, 504,  504, 504, 504, 504,  504,  727,  447,  447,  447,
  447, 447, 280, 280,  280, 280, 504, 560,  504,  504,  504,  504,  504,
  574, 504, 560, 560,  560, 560, 504, 560,  504,
};

static constexpr const uint16_t times_bold_italic_widths[256] = {
  252, 252, 252, 252, 252, 252, 252, 252,  252, 252, 252, 252,  252, 252,
  252, 252, 252, 252, 252, 252, 252, 252,  252, 252, 252, 252,  252, 252,
  252, 252, 252, 252, 252, 392, 559, 504,  504, 839, 784, 280,  335, 335,
  504, 574, 252, 335, 252, 280, 504, 504,  504, 504, 504, 504,  504, 504,
  504, 504, 335, 335, 574, 574, 574, 504,  838, 672, 672, 672,  727, 672,
  672, 727, 784, 392, 504, 672, 615, 896,  727, 727, 615, 727,  672, 560,
  615, 727, 672, 896, 672, 615, 615, 335,  280, 335, 574, 504,  335, 504,
  504, 447, 504, 447, 335, 504, 560, 280,  280, 504, 280, 784,  560, 504,
  504, 504, 392, 392, 280, 560, 447, 672,  504, 447, 392, 350,  221, 350,
  574, 352, 504, 352, 335, 504, 504, 1008, 504, 504, 335, 1008, 560, 335,
  951, 352, 615, 352, 352, 335, 335, 504,  504, 352, 504, 1008, 335, 1008,
  392, 335, 727, 352, 392, 615, 252, 392,  504, 504, 504, 504,  221, 504,
  335, 752, 268, 504, 610, 335, 752, 335,  403, 574, 302, 302,  335, 580,
  504, 252, 335, 302, 302, 504, 756, 756,  756, 504, 672, 672,  672, 672,
  672, 672, 951, 672, 672, 672, 672, 672,  392, 392, 392, 392,  727, 727,
  727, 727, 727, 727, 727, 574, 727, 727,  727, 727, 727, 615,  615, 504,
  504, 504, 504, 504, 504, 504, 727, 447,  447, 447, 447, 447,  280, 280,
  280, 280, 504, 560, 504, 504, 504, 504,  504, 574, 504, 560,  560, 560,
  560, 447, 504, 447,
};

static constexpr const uint16_t times_italic_widths[256] = {
  252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252,  252, 252,
  252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252, 252,  252, 252,
  252, 252, 252, 252, 252, 335, 423, 504, 504, 839, 784, 215,  335, 335,
  504, 680, 252, 335, 252, 280, 504, 504, 504, 504, 504, 504,  504, 504,
  504, 504, 335, 335, 680, 680, 680, 504, 927, 615, 615, 672,  727, 615,
  615, 727, 727, 335, 447, 672, 560, 839, 672, 727, 615, 727,  615, 504,
  560, 727, 615, 839, 615, 560, 560, 392, 280, 392, 425, 504,  335, 504,
  504, 447, 504, 447, 280, 504, 504, 280, 280, 447, 280, 727,  504, 504,
  504, 504, 392, 392, 280, 504, 447, 672, 447, 447, 392, 403,  277, 403,
  545, 352, 504, 352, 335, 504, 560, 896, 504, 504, 335, 1008, 504, 335,
  951, 352, 560, 352, 352, 335, 335, 560, 560, 352, 504, 896,  335, 987,
  392, 335, 672, 352, 392, 560, 252, 392, 504, 504, 504, 504,  277, 504,
  335, 766, 278, 504, 680, 335, 766, 335, 403, 680, 302, 302,  335, 504,
  527, 252, 335, 302, 312, 504, 756, 756, 756, 504, 615, 615,  615, 615,
  615, 615, 896, 672, 615, 615, 615, 615, 335, 335, 335, 335,  727, 672,
  727, 727, 727, 727, 727, 680, 727, 727, 727, 727, 727, 560,  615, 504,
  504, 504, 504, 504, 504, 504, 672, 447, 447, 447, 447, 447,  280, 280,
  280, 280, 504, 504, 504, 504, 504, 504, 504, 680, 504, 504,  504, 504,
  504, 447, 504, 447,
};

static constexpr const uint16_t zapf_dingbats_widths[256] = {
  0,   0,   0,   0,   0,    0,   0,   0,   0,   0,   0,   0,   0,   0,
  0,   0,   0,   0,   0,    0,   0,   0,   0,   0,   0,   0,   0,   0,
  0,   0,   0,   0,   280,  981, 968, 981, 987, 724, 795, 796, 797, 695,
  967, 946, 553, 861, 918,  940, 918, 952, 981, 761, 852, 768, 767, 575,
  682, 769, 766, 765, 760,  497, 556, 541, 581, 697, 792, 794, 794, 796,
  799, 800, 822, 829, 795,  847, 829, 839, 822, 837, 930, 749, 728, 754,
  796, 798, 700, 782, 774,  798, 765, 712, 713, 687, 706, 832, 821, 795,
  795, 712, 692, 701, 694,  792, 793, 718, 797, 791, 797, 879, 767, 768,
  768, 765, 765, 899, 899,  794, 790, 441, 139, 279, 418, 395, 395, 673,
  673, 0,   393, 393, 319,  319, 278, 278, 513, 513, 413, 413, 235, 235,
  336, 336, 0,   0,   0,    0,   0,   0,   0,   0,   0,   0,   0,   0,
  0,   0,   0,   0,   0,    0,   0,   737, 548, 548, 917, 672, 766, 766,
  782, 599, 699, 631, 794,  794, 794, 794, 794, 794, 794, 794, 794, 794,
  794, 794, 794, 794, 794,  794, 794, 794, 794, 794, 794, 794, 794, 794,
  794, 794, 794, 794, 794,  794, 794, 794, 794, 794, 794, 794, 794, 794,
  794, 794, 901, 844, 1024, 461, 753, 931, 753, 925, 934, 935, 935, 840,
  879, 834, 931, 931, 924,  937, 938, 466, 890, 842, 842, 873, 873, 701,
  701, 880, 0,   880, 766,  953, 777, 871, 777, 895, 974, 895, 837, 879,
  934, 977, 925, 0,
};

static constexpr const uint16_t courier_widths[256] = {
  604, 604, 604, 604, 604, 604, 604, 604, 604, 604, 604, 604, 604, 604, 604,
  604, 604, 604, 604, 604, 604, 604, 604, 604, 604, 604, 604, 604, 604, 604,
  604, 604, 604, 604, 604, 604, 604, 604, 604, 604, 604, 604, 604, 604, 604,
  604, 604, 604, 604, 604, 604, 604, 604, 604, 604, 604, 604, 604, 604, 604,
  604, 604, 604, 604, 604, 604, 604, 604, 604, 604, 604, 604, 604, 604, 604,
  604, 604, 604, 604, 604, 604, 604, 604, 604, 604, 604, 604, 604, 604, 604,
  604, 604, 604, 604, 604, 604, 604, 604, 604, 604, 604, 604, 604, 604, 604,
  604, 604, 604, 604, 604, 604, 604, 604, 604, 604, 604, 604, 604, 604, 604,
  604, 604, 604, 604, 604, 604, 604, 604, 604, 604, 604, 604, 604, 604, 604,
  604, 604, 604, 604, 604, 604, 604, 604, 604, 604, 604, 604, 604, 604, 604,
  604, 604, 604, 604, 604, 604, 604, 604, 604, 604, 604, 604, 604, 604, 604,
  604, 604, 604, 604, 604, 604, 604, 604, 604, 604, 604, 604, 604, 604, 604,
  604, 604, 604, 604, 604, 604, 604, 604, 604, 604, 604, 604, 604, 604, 604,
  604, 604, 604, 604, 604, 604, 604, 604, 604, 604, 604, 604, 604, 604, 604,
  604, 604, 604, 604, 604, 604, 604, 604, 604, 604, 604, 604, 604, 604, 604,
  604, 604, 604, 604, 604, 604, 604, 604, 604, 604, 604, 604, 604, 604, 604,
  604, 604, 604, 604, 604, 604, 604, 604, 604, 604, 604, 604, 604, 604, 604,
  604,
};

// PERF for fixed-width fonts, no need for table.
// We only use this to fill in the widths obj, so it could take a
// std::array<uint16_t, 256> to fill in, or something like that.
static const uint16_t *find_font_widths(PDF::BuiltInFont font) {
  switch (font) {
  case PDF::HELVETICA: return helvetica_widths;
  case PDF::HELVETICA_BOLD: return helvetica_bold_widths;
  case PDF::HELVETICA_OBLIQUE: return helvetica_oblique_widths;
  case PDF::HELVETICA_BOLD_OBLIQUE: return helvetica_bold_oblique_widths;
  case PDF::COURIER:
  case PDF::COURIER_BOLD:
  case PDF::COURIER_OBLIQUE:
  case PDF::COURIER_BOLD_OBLIQUE:
    return courier_widths;
  case PDF::TIMES_ROMAN: return times_roman_widths;
  case PDF::TIMES_BOLD: return times_bold_widths;
  case PDF::TIMES_ITALIC: return times_italic_widths;
  case PDF::TIMES_BOLD_ITALIC: return times_bold_italic_widths;
  case PDF::SYMBOL: return symbol_widths;
  case PDF::ZAPF_DINGBATS: return zapf_dingbats_widths;
  default:
    LOG(FATAL) << "Unknown built-in font";
    return nullptr;
  }
}

bool PDF::PointWidthOfText(const char *text,
                           ptrdiff_t text_len, float size,
                           const Font *font,
                           float *point_width) {
  // Width at 1 point.
  double norm_width = 0;
  if (text_len < 0)
    text_len = strlen(text);
  *point_width = 0.0f;

  // XXX just pass string_view
  std::string_view utf8_text(text, text_len);

  if (VERBOSE) {
    printf("PointWidthOfText [%s]\n", std::string(utf8_text).c_str());
  }

  std::optional<std::vector<uint16_t>> ocids =
    TextToCIDs(utf8_text, font->encoding,
               font->glyph_from_codepoint);
  if (!ocids.has_value()) return false;

  if (VERBOSE) {
    printf("CIDs:\n");
    for (uint16_t cid : ocids.value()) printf("%04x ", cid);
    printf("\n");
  }

  // Note: This used to filter out \n, \r. I think that was
  // probably not best, but it could be the cause of issues?
  for (uint16_t cid : ocids.value()) {
    double w = font->CIDWidth(cid);
    norm_width += w;
  }

  *point_width = norm_width * size;
  return true;
}

uint16_t PDF::Font::GetCID(uint32_t codepoint) const {
  if (encoding == FontEncoding::WIN_ANSI) {
    return MapCodepointWinAnsi(codepoint).value_or(0);
  } else {
    CHECK(encoding == FontEncoding::UNICODE);
    auto it = glyph_from_codepoint.find(codepoint);
    if (it == glyph_from_codepoint.end()) return 0;
    return it->second;
  }
}

double PDF::Font::CharWidth(int codepoint) const {
  return CIDWidth(GetCID(codepoint));
}

double PDF::Font::CIDWidth(uint16_t cid) const {
  auto it = widths.find(cid);
  // Unmapped codepoint or no width.
  if (it == widths.end()) {
    return 0.0;
  }

  const int w = it->second;
  return w * (1.0 / (14.0 * 72.0));
}

bool PDF::GetTextWidth(const std::string &text,
                       float size, float *text_width,
                       const Font *font) {
  if (font == nullptr) font = current_font;
  CHECK(font != nullptr);

  return PointWidthOfText(text.c_str(), -1, size, font, text_width);
}

static const char *find_word_break(const char *str) {
  if (!str)
    return nullptr;

  // Skip over the actual word.
  while (*str && !isspace(*str))
    str++;

  return str;
}

bool PDF::AddTextWrap(const std::string &text,
                      float size, float xoff, float yoff,
                      float angle, uint32_t color, float wrap_width,
                      Alignment align, float *height, Page *page) {
  // Move through the text string, stopping at word boundaries,
  // trying to find the longest text string we can fit in the given width.
  const char *start = text.data();
  const char *last_best = text.data();
  const char *end = text.data();
  // TODO: Don't use fixed size buffers!
  char line[512];
  float orig_yoff = yoff;

  while (start && *start) {
    const char *new_end = find_word_break(end + 1);
    float line_width;
    int output = 0;
    float xoff_align = xoff;

    end = new_end;

    if (!PointWidthOfText(start, end - start, size,
                          current_font, &line_width)) {
      return false;
    }

    if (line_width >= wrap_width) {
      if (last_best == start) {
        // There is a single word that is too long for the line.
        ptrdiff_t i;
        // Find the best character to chop it at.
        for (i = end - start - 1; i > 0; i--) {
          float this_width;
          // Don't look at places that are in the middle of a utf-8
          // sequence.
          if ((start[i - 1] & 0xc0) == 0xc0 ||
              ((start[i - 1] & 0xc0) == 0x80 &&
               (start[i] & 0xc0) == 0x80))
            continue;
          if (!PointWidthOfText(start, i, size,
                                current_font, &this_width)) {
            return false;
          }
          if (this_width < wrap_width) {
            break;
          }
        }
        if (i == 0) {
          SetErr(-EINVAL, "Unable to find suitable line break");
          return false;
        }

        end = start + i;
      } else {
        end = last_best;
      }
      output = 1;
    }

    if (*end == '\0')
      output = 1;

    if (*end == '\n' || *end == '\r')
      output = 1;

    if (output) {
      int len = end - start;
      float char_spacing = 0;
      if (len >= (int)sizeof(line))
        len = (int)sizeof(line) - 1;
      strncpy(line, start, len);
      line[len] = '\0';

      if (!PointWidthOfText(start, len, size,
                            current_font, &line_width)) {
        return false;
      }

      switch (align) {
      case PDF_ALIGN_LEFT:
        // Nothing.
        break;
      case PDF_ALIGN_RIGHT:
        xoff_align += wrap_width - line_width;
        break;
      case PDF_ALIGN_CENTER:
        xoff_align += (wrap_width - line_width) / 2;
        break;
      case PDF_ALIGN_JUSTIFY:
        if ((len - 1) > 0 && *end != '\r' && *end != '\n' &&
            *end != '\0')
          char_spacing = (wrap_width - line_width) / (len - 1);
        break;
      case PDF_ALIGN_JUSTIFY_ALL:
        if ((len - 1) > 0)
          char_spacing = (wrap_width - line_width) / (len - 1);
        break;
      case PDF_ALIGN_NO_WRITE:
        // Doesn't matter, since we're not writing.
        break;
      }

      if (align != PDF_ALIGN_NO_WRITE) {
        pdf_add_text_spacing(line, size, xoff_align, yoff,
                             color, char_spacing, angle, page);
      }

      if (*end == ' ')
        end++;

      start = last_best = end;
      yoff -= size;
    } else {
      last_best = end;
    }
  }

  if (height) {
    *height = orig_yoff - yoff;
  }

  return true;
}

PDF::ImageObj *PDF::pdf_add_raw_grayscale8(const uint8_t *data,
                                           uint32_t width,
                                           uint32_t height) {
  const size_t data_len = (size_t)width * (size_t)height;

  const int idx = (int)objects.size();
  std::string str =
    std::format("<<\n"
                "  /Type /XObject\n"
                "  /Name /Image{}\n"
                "  /Subtype /Image\n"
                "  /ColorSpace /DeviceGray\n"
                "  /Height {}\n"
                "  /Width {}\n"
                "  /BitsPerComponent 8\n"
                "  /Length {}\n"
                ">>stream\n",
                idx, height, width,
                (unsigned long)(data_len + 1));

  str.append((const char *)data, width * height);
  AppendFormat(&str, ">\nendstream\n");

  ImageObj *iobj = AddObject(new ImageObj);
  CHECK(iobj);
  iobj->stream = std::move(str);

  return iobj;
}


// Get the display dimensions of an image, respecting the images aspect ratio
// if only one desired display dimension is defined.
// The pdf parameter is only used for setting the error value.
static void get_img_display_dimensions(uint32_t img_width,
                                       uint32_t img_height,
                                       float *display_width,
                                       float *display_height) {
  CHECK(display_width != nullptr);
  CHECK(display_height != nullptr);
  const float display_width_in = *display_width;
  const float display_height_in = *display_height;

  CHECK(!(display_width_in < 0 && display_height_in < 0)) << "At least "
    "one display dimension needs to be provided.";

  CHECK(img_width > 0 && img_height > 0) << "Invalid image dimensions.";

  if (display_width_in < 0) {
    // Set width, keeping aspect ratio
    *display_width = display_height_in * ((float)img_width / img_height);
  } else if (display_height_in < 0) {
    // Set height, keeping aspect ratio
    *display_height = display_width_in * ((float)img_height / img_width);
  }
}

bool PDF::pdf_add_image(ImageObj *image, float x, float y,
                        float width, float height, Page *page) {

  if (!page)
    page = (Page *)FindLastObject(OBJ_page);

  if (!page) {
    SetErr(-EINVAL, "Invalid pdf page");
    return false;
  }

  if (image->type != OBJ_image) {
    SetErr(-EINVAL,
           std::format("adding an image, but wrong object type {}",
                       (int)image->type));
    return false;
  }

  if (image->page != nullptr) {
    SetErr(-EEXIST, "image already on a page");
    return false;
  }

  image->page = page;

  std::string str;
  AppendFormat(&str, "q ");
  AppendFormat(&str, "{} 0 0 {} {} {} cm ",
               Float(width), Float(height),
               Float(x), Float(y));
  AppendFormat(&str, "/Image{} Do ", image->index);
  AppendFormat(&str, "Q");

  pdf_add_stream(page, std::move(str));
  return true;
}

#if 0

static Object *pdf_add_raw_rgb24(pdf_doc *pdf,
                                 const uint8_t *data,
                                 uint32_t width, uint32_t height) {
  Object *obj;
  size_t len;
  const char *endstream = ">\nendstream\n";
  dstr str = INIT_DSTR;
  size_t data_len = (size_t)width * (size_t)height * 3;

  AppendFormat(&str,
               "<<\n"
               "  /Type /XObject\n"
               "  /Name /Image{}\n"
               "  /Subtype /Image\n"
               "  /ColorSpace /DeviceRGB\n"
               "  /Height {}\n"
               "  /Width {}\n"
               "  /BitsPerComponent 8\n"
               "  /Length {}\n"
               ">>stream\n",
               flexarray_size(&pdf->objects), height, width,
               (unsigned long)(data_len + 1));

  len = dstr_len(&str) + data_len + strlen(endstream) + 1;
  if (dstr_ensure(&str, len) < 0) {
      dstr_free(&str);
      SetErr(-ENOMEM,
                  "Unable to allocate %lu bytes memory for image",
                  (unsigned long)len);
      return nullptr;
  }
  dstr_append_data(&str, data, data_len);
  dstr_append(&str, endstream);

  obj = pdf_add_object(pdf, OBJ_image);
  if (!obj) {
      dstr_free(&str);
      return nullptr;
  }
  obj->stream.stream = str;

  return obj;
}

// Probably unnecessary--we would always use PNG?
int pdf_add_rgb24(pdf_doc *pdf, Object *page, float x,
                  float y, float display_width, float display_height,
                  const uint8_t *data, uint32_t width, uint32_t height) {
  Object *obj;

  obj = pdf_add_raw_rgb24(pdf, data, width, height);
  if (!obj)
      return pdf->errval;

  get_img_display_dimensions(width, height, &display_width,
                             &display_height));
  return pdf_add_image(pdf, page, obj, x, y, display_width, display_height);
}

// Probably unnecessary--we would always use PNG?
int pdf_add_grayscale8(pdf_doc *pdf, Object *page, float x,
                       float y, float display_width, float display_height,
                       const uint8_t *data, uint32_t width, uint32_t height) {
  Object *obj;

  obj = pdf_add_raw_grayscale8(pdf, data, width, height);
  if (!obj)
      return pdf->errval;

  get_img_display_dimensions(width, height, &display_width,
                             &display_height);

  return pdf_add_image(pdf, page, obj, x, y, display_width, display_height);
}

#endif

static std::optional<JPGHeader> parse_jpeg_header(
    const uint8_t *data, size_t length, std::string *error) {
  if (length < JPG_FILE_MIN_SIZE) {
    if (error != nullptr) {
      *error = "Too small to be a JPEG file.";
    }
    return std::nullopt;
  }

  JPGHeader header;

  // See http://www.videotechnology.com/jpeg/j1.html for details
  if (length >= 4 && data[0] == 0xFF && data[1] == 0xD8) {
    for (size_t i = 2; i < length; i++) {
      if (data[i] != 0xff) {
        break;
      }
      while (++i < length && data[i] == 0xff)
        ;
      if (i + 2 >= length) {
        break;
      }
      int len = data[i + 1] * 256 + data[i + 2];
      /* Search for SOFn marker and decode jpeg details */
      if ((data[i] & 0xf4) == 0xc0) {
        if (len >= 9 && i + len + 1 < length) {
          header.height = data[i + 4] * 256 + data[i + 5];
          header.width = data[i + 6] * 256 + data[i + 7];
          header.ncolors = data[i + 8];
          return {header};
        }
        break;
      }
      i += len;
    }
  }

  if (error != nullptr) {
    *error = "Invalid JPEG header.";
  }
  return std::nullopt;
}

bool PDF::pdf_add_jpeg_data(float x, float y, float display_width,
                            float display_height,
                            const uint8_t *jpeg_data,
                            size_t len,
                            Page *page) {
  std::string error;
  std::optional<JPGHeader> oheader = parse_jpeg_header(jpeg_data, len, &error);
  if (!oheader.has_value()) {
    SetErr(-EINVAL, std::format("Couldn't parse jpeg: {}", error));
    return false;
  }

  CHECK(page != nullptr);

  const JPGHeader &header = oheader.value();

  ImageObj *obj = AddObject(new ImageObj);
  CHECK(obj != nullptr);

  int index = (int)objects.size();
  AppendFormat(&obj->stream,
               "<<\n"
               "  /Type /XObject\n"
               "  /Name /Image{}\n"
               "  /Subtype /Image\n"
               "  /ColorSpace {}\n"
               "  /Width {}\n"
               "  /Height {}\n"
               "  /BitsPerComponent 8\n"
               "  /Filter /DCTDecode\n"
               "  /Length {}\n"
               ">>stream\n",
               index,
               (header.ncolors == 1) ? "/DeviceGray" : "/DeviceRGB",
               header.width, header.height, (unsigned long)len);
  obj->stream.append((const char*)jpeg_data, len);
  AppendFormat(&obj->stream, "\nendstream\n");

  get_img_display_dimensions(header.width,
                             header.height,
                             &display_width, &display_height);

  return pdf_add_image(obj, x, y, display_width, display_height, page);
}


bool PDF::pdf_add_png_data(float x, float y,
                           float display_width,
                           float display_height,
                           const uint8_t *png_data,
                           size_t png_data_length, Page *page) {

  // Any valid png file must be at least this large. Then we can skip
  // length checks when parsing the header.
  CHECK(png_data_length >= PNG_FILE_MIN_SIZE) << png_data_length;

  // Parse the png data from memory.
  PNGHeader header = ReadPngHeader(png_data);

  CHECK(page != nullptr);

  // indicates if we return an error or add the img at the
  // end of the function
  bool success = false;

  // string stream used for writing color space (and palette) info
  // into the pdf
  std::string color_space;

  ImageObj *obj = nullptr;
  uint32_t pos = 0;
  uint8_t *png_data_temp = nullptr;
  size_t png_data_total_length = 0;
  uint8_t ncolors = 0;
  std::string final_data;

  // Stores palette information for indexed PNGs
  struct rgb_value *palette_buffer = nullptr;
  size_t palette_buffer_length = 0;

  // Father info from png header
  switch (header.color_type) {
  case PNG_COLOR_GREYSCALE:
    ncolors = 1;
    break;
  case PNG_COLOR_RGB:
    ncolors = 3;
    break;
  case PNG_COLOR_INDEXED:
    ncolors = 1;
    break;
    // PNG_COLOR_RGBA and PNG_COLOR_GREYSCALE_A are unsupported
  default:
    SetErr(-EINVAL, std::format("PNG has unsupported color type: {}",
                                (int)header.color_type));
    goto free_buffers;
    break;
  }

  /* process PNG chunks */
  static constexpr int PNG_MAGIC_SIZE = 8;
  pos = PNG_MAGIC_SIZE;

  while (1) {
    if (pos + 8 > png_data_length - 4) {
      SetErr(-EINVAL, "PNG file too short");
      goto free_buffers;
    }

    PNGChunk chunk = ReadPngChunk(&png_data[pos]);
    pos += 8;

    const uint32_t chunk_length = chunk.length;
    // chunk length + 4-bytes of CRC
    if (chunk_length > png_data_length - pos - 4) {
      SetErr(-EINVAL,
             std::format("PNG chunk exceeds file: {} vs {}",
                         chunk_length, (long)(png_data_length - pos - 4)));
      goto free_buffers;
    }
    if (strncmp(chunk.type, png_chunk_header, 4) == 0) {
      // Ignoring the header, since it was parsed
      // before calling this function.
    } else if (strncmp(chunk.type, png_chunk_palette, 4) == 0) {
      // Palette chunk
      if (header.color_type == PNG_COLOR_INDEXED) {
        // palette chunk is needed for indexed images
        if (palette_buffer) {
          SetErr(-EINVAL,
                 "PNG contains multiple palette chunks");
          goto free_buffers;
        }
        if (chunk_length % 3 != 0) {
          SetErr(-EINVAL,
                 "PNG format error: palette chunk length is "
                 "not divisbly by 3!");
          goto free_buffers;
        }
        palette_buffer_length = (size_t)(chunk_length / 3);
        if (palette_buffer_length > 256 ||
            palette_buffer_length == 0) {
          SetErr(-EINVAL,
                 std::format("PNG palette length invalid: {}",
                             palette_buffer_length));
          goto free_buffers;
        }
        palette_buffer = (struct rgb_value *)malloc(
            palette_buffer_length * sizeof(struct rgb_value));
        CHECK(palette_buffer != nullptr);

        for (size_t i = 0; i < palette_buffer_length; i++) {
          size_t offset = (i * 3) + pos;
          palette_buffer[i].red = png_data[offset];
          palette_buffer[i].green = png_data[offset + 1];
          palette_buffer[i].blue = png_data[offset + 2];
        }
      } else if (header.color_type == PNG_COLOR_RGB ||
                 header.color_type == PNG_COLOR_RGBA) {
        // palette chunk is optional for RGB(A) images
        // but we do not process them
      } else {
        SetErr(-EINVAL,
               std::format("Unexpected palette chunk for color type {}",
                           (int)header.color_type));
        goto free_buffers;
      }
    } else if (strncmp(chunk.type, png_chunk_data, 4) == 0) {
      if (chunk_length > 0 && chunk_length < png_data_length - pos) {
        uint8_t *data = (uint8_t *)realloc(
            png_data_temp, png_data_total_length + chunk_length);
        // (uint8_t *)realloc(info.data, info.length + chunk_length);
        CHECK(data != nullptr);
        png_data_temp = data;
        memcpy(&png_data_temp[png_data_total_length], &png_data[pos],
               chunk_length);
        png_data_total_length += chunk_length;
      }
    } else if (strncmp(chunk.type, png_chunk_end, 4) == 0) {
      /* end of file, exit */
      break;
    }

    if (chunk_length >= png_data_length) {
      SetErr(-EINVAL, "PNG chunk length larger than file");
      goto free_buffers;
    }

    pos += chunk_length;     // add chunk length
    pos += sizeof(uint32_t); // add CRC length
  }

  /* if no length was found */
  if (png_data_total_length == 0) {
    SetErr(-EINVAL, "PNG file has zero length");
    goto free_buffers;
  }

  switch (header.color_type) {
  case PNG_COLOR_GREYSCALE:
    AppendFormat(&color_space, "/DeviceGray");
    break;
  case PNG_COLOR_RGB:
    AppendFormat(&color_space, "/DeviceRGB");
    break;
  case PNG_COLOR_INDEXED:
    if (palette_buffer_length == 0) {
      SetErr(-EINVAL, "Indexed PNG contains no palette");
      goto free_buffers;
    }
    // Write the color palette to the color_palette buffer
    AppendFormat(&color_space,
                 "[ /Indexed\n"
                 "  /DeviceRGB\n"
                 "  {}\n"
                 "  <",
                 (unsigned long)(palette_buffer_length - 1));
    // write individual palette values
    // the index value for every RGB value is determined by its position
    // (0, 1, 2, ...)
    for (size_t i = 0; i < palette_buffer_length; i++) {
      AppendFormat(&color_space, "{:02X}{:02X}{:02X} ",
                   palette_buffer[i].red,
                   palette_buffer[i].green,
                   palette_buffer[i].blue);
    }
    AppendFormat(&color_space, ">\n]");
    break;

  default:
    SetErr(-EINVAL,
           std::format("Cannot map PNG color type {} to PDF color space",
                       (int)header.color_type));
    goto free_buffers;
    break;
  }

  final_data.reserve(png_data_total_length + 1024 + color_space.size());
  // final_data = (uint8_t *)malloc(png_data_total_length + 1024 +
  // color_space.size());
  // CHECK(final_data != nullptr);

  // Write image information to PDF
  {
    const int idx = (int)objects.size();
    final_data =
      std::format(
          "<<\n"
          "  /Type /XObject\n"
          "  /Name /Image{}\n"
          "  /Subtype /Image\n"
          "  /ColorSpace {}\n"
          "  /Width {}\n"
          "  /Height {}\n"
          "  /Interpolate true\n"
          "  /BitsPerComponent {}\n"
          "  /Filter /FlateDecode\n"
          "  /DecodeParms << /Predictor 15 /Colors {} "
          "/BitsPerComponent {} /Columns {} >>\n"
          "  /Length {}\n"
          ">>stream\n",
          idx, color_space,
          header.width, header.height, header.bit_depth, ncolors,
          header.bit_depth, header.width, png_data_total_length);

    final_data.append(std::string_view((const char*)png_data_temp,
                                       png_data_total_length));
    AppendFormat(&final_data, "\nendstream\n");
  }

  obj = AddObject(new ImageObj);

  obj->stream = std::move(final_data);

  get_img_display_dimensions(header.width, header.height,
                             &display_width, &display_height);

  success = true;

 free_buffers:
  if (palette_buffer)
    free(palette_buffer);
  if (png_data_temp)
    free(png_data_temp);

  if (success)
    return pdf_add_image(obj, x, y, display_width, display_height, page);
  else {
    return false;
  }
}

bool PDF::AddImageRGB(float x, float y,
                      // If one of width or height is negative, then the
                      // value is determined from the other, preserving the
                      // aspect ratio.
                      float display_width, float display_height,
                      const ImageRGB &img,
                      CompressionType compression,
                      Page *page) {

  if (!page)
    page = (Page*)FindLastObject(OBJ_page);

  CHECK(page != nullptr);

  // Also easy to support JPG here.
  if (compression == CompressionType::PNG) {
    std::vector<uint8_t> png =
      ZIP::RGBEncodeAsPNG(img.Width(), img.Height(), img.ToBuffer8());
    return pdf_add_png_data(x, y,
                            display_width, display_height,
                            png.data(), png.size(),
                            page);
  }

  int jpeg_level = 100;
  switch (compression) {
  default:
  case CompressionType::JPG_0: jpeg_level = 0; break;
  case CompressionType::JPG_10: jpeg_level = 10; break;
  case CompressionType::JPG_20: jpeg_level = 20; break;
  case CompressionType::JPG_30: jpeg_level = 30; break;
  case CompressionType::JPG_40: jpeg_level = 40; break;
  case CompressionType::JPG_50: jpeg_level = 50; break;
  case CompressionType::JPG_60: jpeg_level = 60; break;
  case CompressionType::JPG_70: jpeg_level = 70; break;
  case CompressionType::JPG_80: jpeg_level = 80; break;
  case CompressionType::JPG_90: jpeg_level = 90; break;
  case CompressionType::JPG_100: jpeg_level = 100; break;
  }

  std::vector<uint8_t> jpg = img.SaveJPGToVec(jpeg_level);
  return pdf_add_jpeg_data(x, y,
                           display_width, display_height,
                           jpg.data(), jpg.size(),
                           page);
}

namespace {
// TODO: To utf8.h?
struct UTF8Codepoints {
  UTF8Codepoints(std::string_view s) :
    begin_it(s.data(), s.data() + s.size()),
    end_it(s.data() + s.size(), s.data() + s.size()) {}

  struct const_iterator {
    constexpr const_iterator(const char *ptr, const char *limit) :
      ptr(ptr), limit(limit) {}
    constexpr const_iterator(const const_iterator &other) = default;
    constexpr bool operator ==(const const_iterator &other) const {
      return other.ptr == ptr;
    }
    constexpr bool operator !=(const const_iterator &other) const {
      return other.ptr != ptr;
    }
    const_iterator &operator ++() {
      // prefix.
      uint32_t code_unused = 0;
      const int code_len = utf8_to_utf32(ptr, limit - ptr, &code_unused);
      CHECK(code_len > 0) << "Invalid UTF-8 encoding.";
      ptr += code_len;
      return *this;
    }
    const_iterator operator ++(int postfix) {
      auto old = *this;
      uint32_t code_unused = 0;
      const int code_len = utf8_to_utf32(ptr, limit - ptr, &code_unused);
      CHECK(code_len > 0) << "Invalid UTF-8 encoding.";
      ptr += code_len;
      return old;
    }

    uint32_t operator *() const {
      uint32_t code = 0;
      (void)utf8_to_utf32(ptr, limit - ptr, &code);
      return code;
    }

  private:
    const char *ptr = nullptr;
    const char *limit = nullptr;
  };

  constexpr const_iterator begin() const {
    return begin_it;
  }

  constexpr const_iterator end() const {
    return end_it;
  }

private:
  const const_iterator begin_it, end_it;
};
}  // namespace

double PDF::Font::GetKernedWidth(std::string_view text) const {
  uint32_t prev_cp = 0;
  double width = 0.0;
  for (uint32_t cp : UTF8Codepoints(text)) {
    width += CharWidth(cp);
    auto kit = kerning.find(std::make_pair((int)prev_cp, (int)cp));
    if (kit != kerning.end()) {
      if (VERBOSE) {
        printf("kern %c+%c with %.6f\n", prev_cp, cp, kit->second);
      }
      width += kit->second;
    }
    prev_cp = cp;
  }

  return width;
}

PDF::SpacedLine PDF::Font::KernText(std::string_view text) const {
  PDF::SpacedLine ret;

  uint32_t prev_cp = 0;
  std::string chunk;
  for (uint32_t cp : UTF8Codepoints(text)) {
    auto kit = kerning.find(std::make_pair((int)prev_cp, (int)cp));
    if (kit == kerning.end()) {
      // Keep accumulating with the default spacing.
    } else {
      // kern.
      float k = kit->second;
      ret.emplace_back(chunk, k);
      chunk.clear();
    }
    chunk += UTF8::Encode(cp);
    prev_cp = cp;
  }

  if (!chunk.empty()) {
    ret.emplace_back(chunk, 0.0f);
  }

  return ret;
}


std::string PDF::AddTTF(std::string_view filename,
                        FontEncoding encoding) {
  std::vector<uint8_t> ttf_bytes = Util::ReadFileBytes(filename);
  CHECK(!ttf_bytes.empty()) << filename;

  stbtt_fontinfo ttf;
  int offset = stbtt_GetFontOffsetForIndex(ttf_bytes.data(), 0);
  CHECK(offset != -1);
  CHECK(stbtt_InitFont(&ttf, ttf_bytes.data(), ttf_bytes.size(), offset)) <<
    "Failed to load " << filename;

  // Needed?
  int native_ascent = 0, native_descent = 0;
  int native_linegap = 0;

  stbtt_GetFontVMetrics(
      &ttf, &native_ascent, &native_descent, &native_linegap);

  Font *font = new Font;
  std::string font_name = std::format("Font{}", next_font_index);
  font->font_index = next_font_index;
  next_font_index++;
  embedded_fonts[font_name] = font;

  int space_width = 0;
  stbtt_GetCodepointHMetrics(&ttf, ' ', &space_width, nullptr);

  const float scale = stbtt_ScaleForMappingEmToPixels(&ttf, 72.0f);

  // width tables use 14 points.
  const float scale_14pt = scale * 14.0f;

  if (VERBOSE) {
    printf("Scale 14pt: %.6f\n", scale_14pt);
  }

  if (VERBOSE) {
    printf("[%s] font tables:\n", std::string(filename).c_str());
    stbtt__print_tables(&ttf);
  }

  // Glyph -> codepoint(s) map. We use this temporarily to
  // load the kerning table in terms of codepoints, and then
  // to populate the ToUnicode CMap.
  std::unordered_map<uint16_t, std::vector<uint32_t>> codepoint_from_glyph =
    stbtt_GetGlyphs(&ttf);

  // The set of all glyphs we can access.
  std::unordered_set<int> all_glyphs;
  for (const auto &[glyph, _] : codepoint_from_glyph) {
    all_glyphs.insert(glyph);
  }

  // Get widths for mapped codepoints.
  // fobj->widths.resize(256, (uint16_t)std::round(space_width * scale_14pt));
  auto ScaleWidth = [scale_14pt](int width_unscaled) {
      return scale_14pt * width_unscaled;
    };

  if (encoding == FontEncoding::WIN_ANSI) {
    auto GetWidth = [&ttf, space_width](int codepoint) {
        // Treat missing glyphs as the space character.
        // (Perhaps should be space/2?)
        if (stbtt_FindGlyphIndex(&ttf, codepoint) == 0)
          return space_width;

        int width = 0;
        stbtt_GetCodepointHMetrics(&ttf, codepoint, &width, nullptr);
        return width;
      };

    auto SetCodepointWidth = [&](int codepoint) {
        if (std::optional<int> co = MapCodepointWinAnsi(codepoint)) {
          int cid = co.value();
          CHECK(cid >= 0 && cid < 256) << codepoint << " " << cid;
          int width_unscaled = GetWidth(codepoint);
          float width = ScaleWidth(width_unscaled);
          font->widths[cid] = (uint16_t)std::round(width);
          if (VERBOSE && isalnum(codepoint)) {
            printf("'%c' (%d): %d -> %.5f\n",
                   codepoint, codepoint, width_unscaled, width);
          }
        }
      };

    for (int cp = 0; cp < 128; cp++)
      SetCodepointWidth(cp);
    for (uint16_t cp : MAPPED_CODEPOINTS)
      SetCodepointWidth(cp);
  } else {

    for (const auto &[glyph_id, cp_] : codepoint_from_glyph) {
      int width_unscaled = 0;
      stbtt_GetGlyphHMetrics(&ttf, glyph_id, &width_unscaled, nullptr);
      font->widths[glyph_id] = ScaleWidth(width_unscaled);
    }
  }

  if (VERBOSE) {
    // XXX
    printf("%s Widths:\n", std::string(filename).c_str());
    for (const auto &[cid, w] : font->widths) {
      printf("%04x=%d ", cid, w);
    }
    printf("\n");
  }

  // Load the Kerning table. The same glyph can be used for multiple
  // codepoints. Note we store it for each codepoint pair, which means
  // it could be quadratic size.
  std::unordered_map<std::pair<int, int>, double,
    Hashing<std::pair<int, int>>> kerning;

  double kerning_scale = scale / 72.0;

  static constexpr bool KERNING_TABLE_ONLY = false;
  if constexpr (KERNING_TABLE_ONLY) {
    // TODO: This is only the first kerning table.
    const int table_size = stbtt_GetKerningTableLength(&ttf);
    if (VERBOSE) {
      printf("TTF kerning table: %d.\n", table_size);
    }
    std::vector<stbtt_kerningentry> table;
    table.resize(table_size);
    CHECK(table_size ==
          stbtt_GetKerningTable(&ttf, table.data(), table_size));
    // The kerning table is given using glyphs. Convert to
    // codepoints.
    for (const stbtt_kerningentry &kern : table) {
      double advance = kerning_scale * kern.advance;
      auto cit1 = codepoint_from_glyph.find(kern.glyph1);
      auto cit2 = codepoint_from_glyph.find(kern.glyph2);
      if (cit1 != codepoint_from_glyph.end() &&
          cit2 != codepoint_from_glyph.end()) {
        for (uint32_t c1 : cit1->second) {
          for (uint32_t c2 : cit2->second) {
            kerning[std::make_pair(c1, c2)] = advance;
            if (VERBOSE) {
              printf("'%c' '%c': %d (= %.5f)\n",
                     c1, c2, kern.advance, advance);
            }
          }
        }
      }
    }
  } else {
    // This approach accesses both the kerning table and GPOS data, which
    // is used in a lot of more modern fonts.
    //
    // TODO PERF: It would be better if stb_truetype gave us access to the GPOS
    // table directly. In addition to this being an n^2 loop over glyph pairs
    // (even in the common situation that a character has no kerning data),
    // the call is looping over the table internally to find the glyphs.
    for (int g1 : all_glyphs) {
      for (int g2 : all_glyphs) {
        int kern = stbtt_GetGlyphKernAdvance(&ttf, g1, g2);

        // In principle there could be a kerning entry with 0, but we treat
        // this as no kerning entry.
        if (kern != 0) {
          double advance = kerning_scale * kern;
          // XXX just loop over the glyph -> codepoint mapping?
          auto cit1 = codepoint_from_glyph.find(g1);
          auto cit2 = codepoint_from_glyph.find(g2);
          if (cit1 != codepoint_from_glyph.end() &&
              cit2 != codepoint_from_glyph.end()) {
            for (uint32_t c1 : cit1->second) {
              for (uint32_t c2 : cit2->second) {
                kerning[std::make_pair(c1, c2)] = advance;
                if (VERBOSE) {
                  printf("U+%04x U+%04x = '%s' '%s': %d (= %.5f)\n",
                         c1, c2,
                         UTF8::Encode(c1).c_str(),
                         UTF8::Encode(c2).c_str(),
                         kern, advance);
                }
              }
            }
          }
        }
      }
    }
  }

  printf("[%s] There are %d (codepoint) kerning pairs.\n",
         std::string(filename).c_str(),
         (int)kerning.size());

  // Create the stream for the embedded data.
  // We also put common metadata in here (e.g. bbox) that the
  // FontDescriptor needs when we embed this TTF.
  // XXX might be cleaner if this represented the FontDescriptor
  // object, and we used an indirect reference to it.
  Object *ttf_obj = TrueTypeFontDescriptor(
      &ttf,
      ttf_bytes,
      font->font_index);

  font->kerning = std::move(kerning);

  // Create the stream for the cmap, if applicable.
  StreamObj *cmap = nullptr;
  if (encoding == FontEncoding::UNICODE) {
    // XXX this might be arbitrary, and we might want to
    // generate a different one per CMap. But this is
    // adobe's (unexplained) example.
    std::string cmap_name =
      // "Adobe-Identity-UCS";
      std::format("CMap{}", font->font_index);
    std::string resource;

    // Following example 5.17 in PDF 1.3 spec.
    AppendFormat(
        &resource,
        "/CIDInit /ProcSet findresource begin\n"
        "12 dict begin\n"
        "begincmap\n"
        "/CIDSystemInfo <<\n"
        "  /Registry (Adobe) def\n"
        "  /Ordering (UCS) def\n"
        "  /Supplement 0\n"
        ">> def\n");

    AppendFormat(
        &resource,
        "/CMapName /{} def\n"
        // type 2 is a ToUnicode CMap.
        "/CMapType 2 def\n", cmap_name);

    AppendFormat(
        &resource,
        // This basically says that we are using 16-bit CIDs.
        // 1 is the number of rows in here.
        "1 begincodespacerange\n"
        "<0000> <FFFF>\n"
        "endcodespacerange\n");

    std::vector<std::pair<uint16_t, uint32_t>> mapping;
    mapping.reserve(codepoint_from_glyph.size());
    for (const auto &[glyph, codepoints] : codepoint_from_glyph) {
      for (uint32_t codepoint : codepoints) {
        mapping.emplace_back(glyph, codepoint);
      }
    }
    std::sort(mapping.begin(), mapping.end(),
              [](const auto &a, const auto &b) {
                return a.first < b.first;
              });

    // PERF: We can use bfrange here for more efficient encoding.

    // We have to output the number of rows, so first prepare it.
    std::vector<std::string> rows;
    rows.reserve(mapping.size());
    for (const auto &[glyph, codepoint] : mapping) {
      // Maps 16-bit glyph id to *strings* in Unicode (I think UTF-16).
      // You could map a custom ligature glyph to a sequence of normal
      // unicode characters, for example.
      if (codepoint < 0xFFFF) {
/*
      CHECK(codepoint <= 0xFFFF) << "This code needs to be extended "
        "to represent codepoints beyond the BMP. It should be totally "
        "doable with like, surrogate pairs. Alternatively, you could "
        "just map the glyph to something else (or omit it) and the "
        "only thing that would go wrong is copy/paste and search. "
        << std::format("Saw: U+%{:08x}", codepoint);
*/
        rows.push_back(
            std::format("<{:04x}> <{:04x}> <{:04x}>\n",
                        glyph, glyph, codepoint));
      }
    }

    AppendFormat(&resource,
                 "{} beginbfrange\n",
                 rows.size());
    for (const std::string &row : rows) {
      AppendFormat(&resource, "{}\n", row);
    }
    resource.append("endbfrange\n");

    AppendFormat(
        &resource,
        "endcmap\n"
        "CMapName currentdict /CMap defineresource pop\n"
        "end\n"
        "end\n");
    // the endstream is supplied by the below.

    // Adobe's example (p344) actually doesn't have /Type /CMap..
    cmap = AddStreamObject({{"/Type", "/CMap"}}, resource);


    for (const auto &[glyph, codepoints] : codepoint_from_glyph) {
      for (uint32_t codepoint : codepoints) {
        CHECK(!font->glyph_from_codepoint.contains(codepoint));
        font->glyph_from_codepoint[codepoint] = glyph;
      }
    }

    if (VERBOSE) {
      std::vector<std::pair<uint32_t, uint16_t>> sorted =
        MapToSortedVec(font->glyph_from_codepoint);
      for (const auto &[codepoint, glyph] : sorted) {
        printf("Codepoint U+%04x -> %d\n", codepoint, glyph);
      }
    }
  }
  font->encoding = encoding;

  // Output the widths object for this font.
  WidthsObj *wobj = AddObject(new WidthsObj);
  wobj->encoding = encoding;
  if (encoding == FontEncoding::WIN_ANSI) {
    // We only use pdf encoding for now.
    wobj->firstchar = 0;
    wobj->lastchar = 255;
    wobj->widths8.reserve(256);
    for (int i = wobj->firstchar; i < wobj->lastchar + 1; i++) {
      wobj->widths8.push_back(font->widths[i]);
    }
  } else {
    // Need the whole map.
    // XXX set default_width? how?
    wobj->widths16 = font->widths;
  }

  // Create the appropriate objects.
  if (encoding == FontEncoding::WIN_ANSI) {
    Font8Obj *fobj = AddObject(new Font8Obj);
    fobj->font = font;
    fobj->ttf = ttf_obj;
    fobj->widths_obj = wobj;
    font->f8obj = fobj;
  } else {
    CHECK(encoding == FontEncoding::UNICODE);

    // Here we need *two* objects: A Type0 font containing
    // the CIDFontType2.
    Font0Obj *f0obj = AddObject(new Font0Obj);
    FontCIDObj *fcobj = AddObject(new FontCIDObj);
    f0obj->fcobj = fcobj;
    f0obj->font = font;
    f0obj->cmap_obj = cmap;

    fcobj->font = font;
    fcobj->ttf = ttf_obj;
    fcobj->widths_obj = wobj;

    font->f0obj = f0obj;
  }

  return font_name;
}

std::optional<double>
PDF::Font::GetKerning(int codepoint1, int codepoint2) const {
  const auto it = kerning.find(std::make_pair(codepoint1, codepoint2));
  if (it == kerning.end()) return std::nullopt;
  return it->second;
}

std::string PDF::Font::BaseFont() const {
  if (builtin_font.has_value()) {
    return BuiltInFontName(builtin_font.value());
  } else {
    return std::format("Font{}", font_index);
  }
}

void PDF::SetDimensions(float ww, float hh) {
  CHECK(nullptr == (Page*)FindLastObject(OBJ_page)) << "It is not (yet?) "
    "supported to change the dimensions of a PDF document once you've added "
    "a page. It might just work; you could try removing this error.";

  document_width = ww;
  document_height = hh;
}

/**
 * PDF HINTS & TIPS
 * The specification can be found at
 * https://www.adobe.com/content/dam/acom/en/devnet/pdf/pdfs/pdf_reference_archives/PDFReference.pdf
 * The following sites have various bits & pieces about PDF document
 * generation
 * http://www.mactech.com/articles/mactech/Vol.15/15.09/PDFIntro/index.html
 * http://gnupdf.org/Introduction_to_PDF
 * http://www.planetpdf.com/mainpage.asp?WebPageID=63
 * http://archive.vector.org.uk/art10008970
 * http://www.adobe.com/devnet/acrobat/pdfs/pdf_reference_1-7.pdf
 * https://blog.idrsolutions.com/2013/01/understanding-the-pdf-file-format-overview/
 *
 * To validate the PDF output, there are several online validators:
 * http://www.validatepdfa.com/online.htm
 * http://www.datalogics.com/products/callas/callaspdfA-onlinedemo.asp
 * http://www.pdf-tools.com/pdf/validate-pdfa-online.aspx
 *
 * In addition the 'pdftk' server can be used to analyse the output:
 * https://www.pdflabs.com/docs/pdftk-cli-examples/
 *
 * PDF page markup operators:
 * b    closepath, fill,and stroke path.
 * B    fill and stroke path.
 * b*   closepath, eofill,and stroke path.
 * B*   eofill and stroke path.
 * BI   begin image.
 * BMC  begin marked content.
 * BT   begin text object.
 * BX   begin section allowing undefined operators.
 * c    curveto.
 * cm   concat. Concatenates the matrix to the current transform.
 * cs   setcolorspace for fill.
 * CS   setcolorspace for stroke.
 * d    setdash.
 * Do   execute the named XObject.
 * DP   mark a place in the content stream, with a dictionary.
 * EI   end image.
 * EMC  end marked content.
 * ET   end text object.
 * EX   end section that allows undefined operators.
 * f    fill path.
 * f*   eofill Even/odd fill path.
 * g    setgray (fill).
 * G    setgray (stroke).
 * gs   set parameters in the extended graphics state.
 * h    closepath.
 * i    setflat.
 * ID   begin image data.
 * j    setlinejoin.
 * J    setlinecap.
 * k    setcmykcolor (fill).
 * K    setcmykcolor (stroke).
 * l    lineto.
 * m    moveto.
 * M    setmiterlimit.
 * n    end path without fill or stroke.
 * q    save graphics state.
 * Q    restore graphics state.
 * re   rectangle.
 * rg   setrgbcolor (fill).
 * RG   setrgbcolor (stroke).
 * s    closepath and stroke path.
 * S    stroke path.
 * sc   setcolor (fill).
 * SC   setcolor (stroke).
 * sh   shfill (shaded fill).
 * Tc   set character spacing.
 * Td   move text current point.
 * TD   move text current point and set leading.
 * Tf   set font name and size.
 * Tj   show text.
 * TJ   show text, allowing individual character positioning.
 * TL   set leading.
 * Tm   set text matrix.
 * Tr   set text rendering mode.
 * Ts   set super/subscripting text rise.
 * Tw   set word spacing.
 * Tz   set horizontal scaling.
 * T*   move to start of next line.
 * v    curveto.
 * w    setlinewidth.
 * W    clip.
 * y    curveto.
 */
