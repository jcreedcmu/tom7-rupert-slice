
#include "png.h"

#include <cstdint>
#include <cstdio>
#include <span>
#include <string_view>

#include "base/logging.h"

#define CHECK_EQUAL(a, b) do {                  \
    auto aa = (a);                              \
    auto bb = (b);                              \
    CHECK(aa == bb) << #a " vs " #b "\n"        \
                    << "which is\n"             \
                    << aa << " vs " << bb;      \
  } while (0)


inline std::span<const uint8_t> StringSpan(std::string_view s) {
  return std::span<const uint8_t>((const uint8_t*)s.data(),
                                  s.size());
}

static void TestCRC() {
  CHECK_EQUAL(PNG::CRC(StringSpan("table task")), (uint32_t)0x632AECF0);
  CHECK_EQUAL(PNG::CRC(StringSpan("IEND")), (uint32_t)0xAE426082);
}

int main(int argc, char **argv) {
  TestCRC();


  printf("OK");
  return 0;
}
