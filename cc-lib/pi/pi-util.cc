#include "pi-util.h"

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <linux/if_link.h>

#include <string>
#include <optional>
#include <cstdint>
#include <vector>

#include "util.h"

using namespace std;

optional<uint64_t> PiUtil::GetSerial() {
  vector<string> cpuinfo = Util::ReadFileToLines("/proc/cpuinfo");
  for (string &line : cpuinfo) {
    if (Util::TryStripPrefix("Serial", &line)) {
      line = Util::LoseWhiteL(line);
      if (Util::TryStripPrefix(":", &line)) {
        line = Util::LoseWhiteL(line);
        const uint64_t serial =
          strtoll(line.c_str(), nullptr, 16);
        if (serial != 0)
          return {serial};
      }
    }
  }
  return {};
}
