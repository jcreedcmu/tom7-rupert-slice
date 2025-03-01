
#ifndef _RUPERTS_SYMMETRY_GROUPS_H
#define _RUPERTS_SYMMETRY_GROUPS_H

#include <vector>
#include <numbers>

#include "yocto_matht.h"
#include "polyhedra.h"

// The platonic solids are a nice way to get to their corresponding
// symmetry groups. We normalize them and convert each vertex to
// a rotation frame.
struct SymmetryGroups {
  using vec2 = yocto::vec<double, 2>;
  using vec3 = yocto::vec<double, 3>;
  using vec4 = yocto::vec<double, 4>;
  using mat4 = yocto::mat<double, 4>;
  using quat4 = yocto::quat<double, 4>;
  using frame3 = yocto::frame<double, 3>;

  struct Group {
    std::vector<frame3> rots;
    int points = 0;
  };
  Group tetrahedron, octahedron, icosahedron;
  SymmetryGroups() {
    {
      Polyhedron t = Tetrahedron();
      VertexRotationsTriangular(t, &tetrahedron.rots);
      tetrahedron.points = 4;
      delete t.faces;
    }

    {
      Polyhedron o = Octahedron();
      VertexRotationsQuadrilateral(o, &octahedron.rots);
      EdgeRotations(o, &octahedron.rots);
      octahedron.points = 6;
      delete o.faces;
    }

    {
      Polyhedron i = Icosahedron();
      VertexRotationsPentagonal(i, &icosahedron.rots);
      EdgeRotations(i, &icosahedron.rots);
      icosahedron.points = 20;
      delete i.faces;
    }
  }

 private:

  // For tetrahedron.
  void VertexRotationsTriangular(const Polyhedron &poly,
                                 std::vector<frame3> *rots) {
    for (const vec3 &v : poly.vertices) {
      vec3 axis = normalize(v);
      rots->push_back(
          yocto::rotation_frame(
              yocto::rotation_quat(axis, 2.0 / 3.0 * std::numbers::pi)));
      rots->push_back(
          yocto::rotation_frame(
              yocto::rotation_quat(axis, -2.0 / 3.0 * std::numbers::pi)));
    }
  }

  // For octahedron.
  void VertexRotationsQuadrilateral(const Polyhedron &poly,
                                    std::vector<frame3> *rots) {
    for (const vec3 &v : poly.vertices) {
      vec3 axis = normalize(v);
      for (int quarter_turn = 1; quarter_turn < 4; quarter_turn++) {
        rots->push_back(
            yocto::rotation_frame(
                yocto::rotation_quat(
                    axis, quarter_turn * std::numbers::pi / 2.0)));
      }
    }
  }

  // For icosahedron.
  void VertexRotationsPentagonal(const Polyhedron &poly,
                                 std::vector<frame3> *rots) {

    for (const vec3 &v : poly.vertices) {
      vec3 axis = normalize(v);
      for (int fifth_turn = 1; fifth_turn < 5; fifth_turn++) {
        rots->push_back(
            yocto::rotation_frame(
                yocto::rotation_quat(
                    axis, fifth_turn / 5.0 * std::numbers::pi / 2.0)));
      }
    }
  }


  // Take an edge and rotate it 180 degrees, using the axis that runs
  // from the origin to its midpoint. This flips the edge around (so it
  // ends up the same).
  void EdgeRotations(const Polyhedron &poly, std::vector<frame3> *rots);
};

#endif
