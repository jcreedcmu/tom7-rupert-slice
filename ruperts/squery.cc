

#include <string>
#include <vector>

#include "ansi.h"
#include "util.h"

#include "shrinklutions.h"

static void Exec(const std::string &query) {
  ShrinklutionDB db;

  db.ExecuteAndPrint(query);
}

int main(int argc, char **argv) {
  ANSI::Init();

  CHECK(argc > 1) << "./squery.exe \"sql query to execute\"";

  std::vector<std::string> parts;
  for (int i = 1; i < argc; i++) parts.emplace_back(argv[i]);
  Exec(Util::Join(parts, " "));

  return 0;
}
