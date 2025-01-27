
#include "ansi.h"

#include <cstdio>
#include <cstdlib>
#include <optional>
#include <string_view>
#include <string>
#include <utility>
#include <vector>

#include "base/stringprintf.h"
#include "polyhedra.h"
#include "solutions.h"
#include "util.h"
#include "yocto_matht.h"

using frame3 = yocto::frame<double, 3>;
using Nopert = SolutionDB::Nopert;


static Polyhedron GetPolyhedron(std::string_view name) {
  if (Util::TryStripPrefix("nopert_", &name)) {
    SolutionDB db;
    int id = atoi(std::string(name).c_str());
    CHECK(id > 0) << name;
    Nopert nopert = db.GetNopert(id);

    std::optional<Polyhedron> opoly =
      PolyhedronFromVertices(nopert.vertices, "nopert");
    if (!opoly.has_value()) {
      printf("Error constructing nopert #%d\n", id);
      for (const vec3 &v : nopert.vertices) {
        printf("  %s\n", VecString(v).c_str());
      }
      CHECK(opoly.has_value()) << "Couldn't make polyhedron " << name;
    }
    return std::move(opoly.value());
  }

  return NormalizeRadius(PolyhedronByName(name));
}

static void Save(std::string_view name, std::string_view file) {
  Polyhedron polyhedron = GetPolyhedron(name);
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
