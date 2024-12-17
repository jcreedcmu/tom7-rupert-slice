
// Even though this is called "tosvg", it just generates PNGs.
// That was enough for me to debug my problem!

#include "ansi.h"

#include "polyhedra.h"
#include "rendering.h"
#include "yocto_matht.h"
#include <vector>

static void Good() {
  Polyhedron polyhedron = SnubCube();

  /*
  frame3 outer_frame{
    .x = vec3(-0.59663484814079548, 0.7269835374195569, -0.33988497216699254),
    .y = vec3(-0.73349173011445112, -0.66583355628313601, -0.13658534760749652),
    .z = vec3(-0.32560211890852109, 0.16781123814661303, 0.93049591536662946),
    .o = vec3(0, 0, 0)};

  frame3 inner_frame{
    .x = vec3(0.74055732836780208, 0.67089398317286308, -0.038420134606650683),
    .y = vec3(0.098978897195982268, -0.16544920698620627, -0.98123887907965734),
    .z = vec3(-0.66466384083280039, 0.7228608602283606, -0.1889289692961463),
    .o = vec3(-0.010851383842952218, -0.0047912447470160862, 0.019844067206751154)};
  */

  frame3 outer_frame{
    .x = vec3(0.82678470481919319, 0.43109236735507755, -0.36136743445589403),
    .y = vec3(-0.50719611043958268, 0.8491187479370319, -0.14747697263236231),
    .z = vec3(0.24326766622796103, 0.3052158624810361, 0.92068676533224858),
    .o = vec3(0, 0, 0)};
  frame3 inner_frame{
    .x = vec3(-0.50819172510764021, 0.8604012159509512, 0.038090919157259893),
    .y = vec3(-0.19149020183281779, -0.069760780660713728, -0.97901222468549098),
    .z = vec3(-0.83968605629374782, -0.50481994916180961, 0.20021025396847339),
    .o = vec3(-0.0047312234995444446, 0.0190775636296374, 0.028260252348308361)};

  Polyhedron outer = Rotate(polyhedron, outer_frame);
  Mesh2D souter = Shadow(outer);

  Polyhedron inner = Rotate(polyhedron, inner_frame);
  Mesh2D sinner = Shadow(inner);

  {
    Rendering rendering(polyhedron, 3840 * 2, 2160 * 2);
    rendering.RenderMesh(souter);
    rendering.DarkenBG();

    rendering.RenderMesh(sinner);
    rendering.RenderBadPoints(sinner, souter);

    rendering.Save("tosvg.png");
  }

  std::vector<int> outer_hull = QuickHull(souter.vertices);
  std::vector<int> inner_hull = QuickHull(sinner.vertices);

  {
    Rendering rendering(polyhedron, 3840 * 2, 2160 * 2);
    rendering.RenderHullDistance(souter, outer_hull);
    rendering.RenderHull(souter, outer_hull, 0x0000FFFF);
    rendering.RenderHull(sinner, inner_hull, 0x00FF0077);
    rendering.Save("tosvg-hulls.png");
  }
}


int main(int argc, char **argv) {
  ANSI::Init();

  Good();

  return 0;
}
