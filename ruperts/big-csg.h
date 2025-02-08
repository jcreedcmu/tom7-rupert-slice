
#ifndef _RUPERTS_BIG_CSG_H
#define _RUPERTS_BIG_CSG_H

#include <vector>

#include "polyhedra.h"
#include "yocto_matht.h"

// Internally uses rational coordinates. Expose this in interface too?
//
// Polyhedron should be a convex polyhedron that contains the origin.
// Polygon should be a simple convex polygon that contains the origin.
// The polygon represents an infinitely tall extrusion along the
// z axis. The result is the solid that results from making a hole
// in the polyhedron with the extrusion (CSG subtraction).
Mesh3D BigMakeHole(const Polyhedron &polyhedron,
                   const std::vector<vec2> &polygon);

#endif
