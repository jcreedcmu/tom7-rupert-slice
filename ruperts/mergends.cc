
#include <cstddef>
#include <cstdio>

#include "nd-solutions.h"

static constexpr int TARGET_SAMPLES = 1'000'000;

int main(int argc, char **argv) {
  NDSolutions<6> dst("456b29ba-45202bba.nds");
  NDSolutions<6> src("456b29ba-45202bba.nds.1");

  printf("dst: %lld\n", dst.Size());
  printf("src: %lld\n", src.Size());

  size_t idx = 0;
  while (idx < src.Size() && dst.Size() < TARGET_SAMPLES) {
    const auto &[key, score, outer, inner] = src[idx];
    dst.Add(key, score, outer, inner);
  }

  printf("OK, now have size %lld\n", dst.Size());

  dst.Save();

  return 0;
}
