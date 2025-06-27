
#include <cstdio>
#include <format>
#include <string>
#include <string_view>

#include "ansi.h"
#include "big-polyhedra.h"
#include "patches.h"

constexpr int DIGITS = 24;

constexpr std::string_view FILENAME = "scube-patchinfo.txt";

static void ComputePatchInfo() {
  PatchInfo info = EnumeratePatches(BigScube(DIGITS));
  SavePatchInfo(info, FILENAME);
  printf("Wrote %s\n", std::string(FILENAME).c_str());
}

[[maybe_unused]]
static void AddHullsToPatchInfo() {
  PatchInfo info = LoadPatchInfo(FILENAME);
  BigPoly poly = BigScube(DIGITS);
  Boundaries boundaries(poly);

  AddHulls(boundaries, &info);
  SavePatchInfo(info, FILENAME);
  printf("Wrote %s\n", std::string(FILENAME).c_str());
}

[[maybe_unused]]
static void DumpHulls() {
  PatchInfo info = LoadPatchInfo(FILENAME);
  for (const auto &[code, canon] : info.canonical) {
    printf("%s:\n", std::format("{:b}", code).c_str());
    for (int v : canon.hull) {
      printf("  %d\n", v);
    }
  }
}

int main(int argc, char **argv) {
  ANSI::Init();

  ComputePatchInfo();
  // AddHullsToPatchInfo();
  // DumpHulls();

  return 0;
}
