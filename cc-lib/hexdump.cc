#include "hexdump.h"

#include <string>
#include <span>

#include "ansi.h"
#include "base/logging.h"
#include "base/stringprintf.h"

std::string HexDump::Color(std::span<const uint8_t> bytes,
                           uint32_t start_addr) {
  size_t lines = bytes.size() / 16;
  if (lines * 16 < bytes.size()) lines++;
  CHECK(lines * 16 >= bytes.size()) << lines << " " << bytes.size();
  std::string ret;
  ret.reserve(lines * 128);

  for (size_t p = 0; p < lines; p++) {
    // Print line: first addre, then hex, then ascii.
    uint32_t line_addr = start_addr + p * 16;
    StringAppendF(&ret, ANSI_FG(86, 82, 97) "%08x " ANSI_RESET,
                  line_addr);
    for (int i = 0; i < 16; i++) {
      size_t addr = p * 16 + i;
      if (addr < bytes.size()) {
        char c = bytes[addr];
        // TODO: color
        StringAppendF(&ret, "%02x ", (uint8_t)c);
      } else {
        ret.append(ANSI_FG(56, 5, 2) " ⋯ ");
      }
    }

    ret.append(ANSI_FG(128, 128, 64) "| ");

    for (size_t i = 0; i < 16; i++) {
      size_t addr = p * 16 + i;
      if (addr < bytes.size()) {
        char c = bytes[addr];
        if (c >= 32 && c < 127) {
          ret.append(ANSI_FG(155, 164, 232));
          ret.push_back(c);
        } else {
          ret.append(ANSI_FG(11, 42, 92) "•");
        }
      } else {
        ret.append(ANSI_FG(56, 5, 2) "⏹");
      }
    }

    ret.append(ANSI_RESET "\n");
  }
  return ret;
}
