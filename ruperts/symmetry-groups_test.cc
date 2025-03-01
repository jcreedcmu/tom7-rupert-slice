
#include "symmetry-groups.h"
#include <cstdio>

#include "ansi.h"

static void Size() {
  SymmetryGroups groups;

  printf(
      AYELLOW("tetrahedron") " has " ACYAN("%d") " rotations, %d pts\n",
      (int)groups.tetrahedron.rots.size(), groups.tetrahedron.points);
  printf(
      AYELLOW("octahedron") " has " ACYAN("%d") " rotations, %d pts\n",
      (int)groups.octahedron.rots.size(), groups.octahedron.points);
  printf(
      AYELLOW("icosahedron") " has " ACYAN("%d") " rotations, %d pts\n",
      (int)groups.icosahedron.rots.size(), groups.icosahedron.points);
}

int main(int argc, char **argv) {
  ANSI::Init();

  Size();

  printf("OK\n");
  return 0;
}
