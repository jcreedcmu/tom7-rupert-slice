

#include <string>
#include <vector>

#include "ansi.h"
#include "util.h"

#include "solutions.h"

static void Exec(const std::string &query) {
  SolutionDB db;

  db.ExecuteAndPrint(query);
}

int main(int argc, char **argv) {
  ANSI::Init();

  CHECK(argc > 1) << "./query.exe \"sql query to execute\"";

  std::vector<std::string> parts;
  for (int i = 1; i < argc; i++) parts.emplace_back(argv[i]);
  Exec(Util::Join(parts, " "));

  return 0;
}
