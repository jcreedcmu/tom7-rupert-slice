#ifndef _CC_LIB_PI_PI_UTIL_H
#define _CC_LIB_PI_PI_UTIL_H

#include <cstdint>
#include <optional>

struct PiUtil {

  // Get 64-bit serial number from /proc/cpuinfo.
  static std::optional<uint64_t> GetSerial();

};

#endif
