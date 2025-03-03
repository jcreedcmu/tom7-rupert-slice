#ifndef _CC_LIB_HEXDUMP_H
#define _CC_LIB_HEXDUMP_H

#include <cstdint>
#include <span>

// "Pretty"-print binary data.
struct HexDump {
  // Using ANSI color codes.
  static std::string Color(std::span<const uint8_t> bytes,
                           // Address of the first byte in bytes;
                           // displayed in the left column.
                           uint32_t start_addr = 0x00000000);

 private:
  // All static.
  HexDump() = delete;
};

#endif
