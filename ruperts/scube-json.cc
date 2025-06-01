
// Export the vertices and faces of the scube (rational approximation to
// snub cube) in an ad hoc JSON format.

#include <cstdio>
#include <format>
#include <string>
#include <vector>

#include "ansi.h"
#include "base/stringprintf.h"
#include "big-polyhedra.h"
#include "bignum/big.h"

static constexpr int DIGITS = 24;

int main(int argc, char **argv) {
  ANSI::Init();

  BigPoly scube = BigScube(DIGITS);

  auto Rat = [](const BigRat &r) {
      const auto &[n, d] = r.Parts();
      return std::format("{{'n': '{}', 'd': '{}'}}",
                         n.ToString(), d.ToString());
    };

  std::string json = "{'v': [\n";
  for (const BigVec3 &v : scube.vertices) {
    AppendFormat(&json, "  {{'x': {}, 'y': {}, 'z': {}}},\n",
                 Rat(v.x), Rat(v.y), Rat(v.z));
  }
  json.append("],");

  json.append("'f': [\n");
  for (const std::vector<int> &face : scube.faces->v) {
    json.append("  [");
    for (int i : face) AppendFormat(&json, "{}, ", i);
    json.append("],\n");
  }

  json.append("]}");

  printf("%s\n", json.c_str());

  return 0;
}
