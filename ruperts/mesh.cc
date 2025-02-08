
#include "mesh.h"

Mesh3D ImproveMesh(const Mesh3D &mesh) {

  // If we have two colinear edges a-b and b-c,
  // like this:
  //
  //       d      //
  //      /|\     //
  //     / | \    //
  //    /  |  \   //
  //   a---b---c  //
  //    \  |  /   //
  //     \ | /    //
  //      \|/     //
  //       e      //
  //
  // Then a-b-e and b-c-e are coplanar, and
  // a-d-b and d-c-b are coplanar. (But these
  // pairs need not be). We can collapse a-b-c to
  // a single edge:
  //
  //       d      //
  //      / \     //
  //     /   \    //
  //    /     \   //
  //   a-------c  //
  //    \     /   //
  //     \   /    //
  //      \ /     //
  //       e      //




}
