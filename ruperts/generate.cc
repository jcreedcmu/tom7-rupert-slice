// Meant to be thrown into
// https://sourceforge.net/p/tom7misc/svn/HEAD/tree/trunk/ruperts/
// with an extra clause like
//
// generate: generate.o polyhedra.o $(CC_LIB_OBJECTS)
// 	@$(CXX) $^ -o $@ $(LFLAGS)
// 	@echo -n "!"
//
// in the makefile. When run, prints some json to stdout.
// This was incorporated into src/raw-poly.ts

#include "polyhedra.h"
#include <vector>

int main() {

  std::vector<Polyhedron> polys;

  polys.emplace_back(Tetrahedron());
  polys.emplace_back(Cube());
  polys.emplace_back(Dodecahedron());
  polys.emplace_back(Icosahedron());
  polys.emplace_back(Octahedron());

// Archimedean
  polys.emplace_back(TruncatedTetrahedron());
  polys.emplace_back(Cuboctahedron());
  polys.emplace_back(TruncatedCube());
  polys.emplace_back(TruncatedOctahedron());
  polys.emplace_back(Rhombicuboctahedron());
  polys.emplace_back(TruncatedCuboctahedron());
  polys.emplace_back(SnubCube());
  polys.emplace_back(Icosidodecahedron());
  polys.emplace_back(TruncatedDodecahedron());
  polys.emplace_back(TruncatedIcosahedron());
  polys.emplace_back(Rhombicosidodecahedron());
  polys.emplace_back(TruncatedIcosidodecahedron());
  polys.emplace_back(SnubDodecahedron());

// Catalan
  polys.emplace_back(TriakisTetrahedron());
  polys.emplace_back(RhombicDodecahedron());
  polys.emplace_back(TriakisOctahedron());
  polys.emplace_back(TetrakisHexahedron());
  polys.emplace_back(DeltoidalIcositetrahedron());
  polys.emplace_back(DisdyakisDodecahedron());
  //  polys.emplace_back(PentagonalIcositetrahedron()); // unimp

  std::cout << R"(
import { Point3 } from "./lib/types";

export type RawPoly = { name: string, verts: Point3[], faces: number[][] };
export const rawPolys: RawPoly[] =
)";

  std::cout << "[";
  const Polyhedron *last = &polys.back();
  for (const Polyhedron &poly : polys) {
    std::cout << "{\"name\": \"" << poly.name << "\",\n";
    std::cout << "\"verts\": ";
    std::cout << "[";
    const auto *last_vert = &poly.vertices.back();
    for (const auto &vert : poly.vertices) {
      std::cout << "[" << vert[0] << "," << vert[1] << "," << vert[2] << "]";
      if (&vert != last_vert) {
        std::cout << ",\n";
      }
    }
    std::cout << "]";
    std::cout << ",\n";
    std::cout << "\"faces\": ";
    std::cout << "[";
    const auto *last_face = &poly.faces->v.back();
    for (const auto &face : poly.faces->v) {

      std::cout << "[";
      const auto *last_ix = &face.back();
      for (const auto &ix : face) {

        std::cout << ix;
        if (&ix != last_ix) {
          std::cout << ",";
        }
      }
      std::cout << "]";

      if (&face != last_face) {
        std::cout << ",\n";
      }
    }
    std::cout << "]";
    std::cout << "}\n";
    if (&poly != last) {
      std::cout << ",\n";
    }
  }
  std::cout << "]";
}
