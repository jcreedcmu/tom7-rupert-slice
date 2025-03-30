
#include "ansi.h"

#include <cstdlib>
#include <string_view>
#include <string>

#include "base/stringprintf.h"
#include "polyhedra.h"
#include "solutions.h"
#include "yocto_matht.h"

using frame3 = yocto::frame<double, 3>;
using Nopert = SolutionDB::Nopert;


static void Save(std::string_view name, std::string_view file) {
  SolutionDB db;
  Polyhedron polyhedron = NormalizeRadius(db.AnyPolyhedronByName(name));
  // This orients the mesh:
  SaveAsSTL(polyhedron, file);
}


int main(int argc, char **argv) {
  ANSI::Init();

  CHECK(argc > 2) << "./tostl.exe polyhedronname [output.stl]";
  std::string name = argv[1];
  std::string file = StringPrintf("%s.stl", std::string(name).c_str());

  if (argc >= 3) file = argv[2];

  Save(argv[1], argv[2]);

  return 0;
}
