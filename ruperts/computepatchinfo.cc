
#include <cstdio>
#include <string>

#include "ansi.h"
#include "big-polyhedra.h"
#include "patches.h"

constexpr int DIGITS = 24;

static void ComputePatchInfo() {
  PatchInfo info = EnumeratePatches(BigScube(DIGITS));
  const std::string filename = "scube-patchinfo.txt";
  SavePatchInfo(info, filename);
  printf("Wrote %s\n", filename.c_str());
}

int main(int argc, char **argv) {
  ANSI::Init();

  ComputePatchInfo();

  return 0;
}
