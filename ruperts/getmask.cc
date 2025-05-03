
#include <cstdio>

#include "ansi.h"
#include "big-polyhedra.h"
#include "patches.h"
#include "util.h"

constexpr int DIGITS = 24;

int main(int argc, char **argv) {
  ANSI::Init();

  CHECK(argc == 2) << "./getmask.exe binary-code";

  std::optional<uint64_t> ocode = Util::ParseBinary(argv[1]);
  CHECK(ocode.has_value()) << "Provide a 64-bit binary constant.\n";
  const uint64_t code = ocode.value();

  Boundaries boundaries(BigScube(DIGITS));
  uint64_t mask = GetCodeMask(boundaries, code, true);

  printf("Mask: %s\n"
         "Code: %s\n",
         std::format("{:b}", mask).c_str(),
         boundaries.ColorMaskedBits(code, mask).c_str());

  return 0;
}
