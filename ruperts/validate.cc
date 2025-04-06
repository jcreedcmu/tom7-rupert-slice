
#include <cstdlib>
#include <cstdio>
#include <format>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "ansi.h"
#include "base/logging.h"
#include "base/stringprintf.h"
#include "big-polyhedra.h"
#include "bignum/big.h"
#include "polyhedra.h"
#include "rendering.h"
#include "solutions.h"
#include "yocto_matht.h"

// We use rational numbers with at least this many digits of
// precision.
static constexpr int DIGITS = 1000;

double IsNear(double a, double b) {
  return std::abs(a - b) < 0.0000001;
}

#define CHECK_NEAR(f, g) do {                                           \
  const double fv = (f);                                                \
  const double gv = (g);                                                \
  const double e = std::abs(fv - gv);                                   \
  CHECK(e < 0.0000001) << "Expected " << #f << " and " << #g <<         \
    " to be close, but got: " <<                                        \
    StringPrintf("%.17g and %.17g, with err %.17g", fv, gv, e);         \
  } while (0)

static void Validate() {
  BigPoly ridode = BigRidode(DIGITS);

  SolutionDB db;
  SolutionDB::Solution sol = db.GetSolution(455);
  printf("Solution:\nOuter: %s\nInner: %s\n",
         FrameString(sol.outer_frame).c_str(),
         FrameString(sol.inner_frame).c_str());

  const auto &[douter_rot, dotrans] =
    UnpackFrame(sol.outer_frame);
  const auto &[dinner_rot, ditrans] =
    UnpackFrame(sol.inner_frame);

  CHECK(dotrans.x == 0.0 &&
        dotrans.y == 0.0 &&
        dotrans.z == 0.0) << "This can be handled, but we expect "
    "exact zero translation for the outer frame.";

  // z component does not matter, because we project along z.
  BigVec2 itrans(BigRat::FromDouble(ditrans.x),
                 BigRat::FromDouble(ditrans.y));


  printf("Outer (dbl):\n%s\n", QuatString(douter_rot).c_str());
  printf("Inner (dbl):\n%s\n", QuatString(dinner_rot).c_str());

  BigQuat oq(BigRat::FromDouble(douter_rot.x),
             BigRat::FromDouble(douter_rot.y),
             BigRat::FromDouble(douter_rot.z),
             BigRat::FromDouble(douter_rot.w));

  BigQuat iq(BigRat::FromDouble(dinner_rot.x),
             BigRat::FromDouble(dinner_rot.y),
             BigRat::FromDouble(dinner_rot.z),
             BigRat::FromDouble(dinner_rot.w));

  printf("Outer (r):\n%s\n", QuatString(oq).c_str());
  printf("Inner (r):\n%s\n", QuatString(iq).c_str());

  // Represent the double quaternions as rational with a lot of
  // digits of precision.
  printf("Normalize Outer:\n");
  // BigQuat outer_rot = Normalize(oq, DIGITS);

  printf("Normalize Inner:\n");
  // BigQuat inner_rot = Normalize(iq, DIGITS);

  // printf("Norm Outer (r):\n%s\n", QuatString(outer_rot).c_str());
  // printf("Norm Inner (r):\n%s\n", QuatString(inner_rot).c_str());

  // XXX just testing a very non-normalized vector.
  iq.x *= BigRat(2, 3);
  iq.y *= BigRat(2, 3);
  iq.z *= BigRat(2, 3);
  iq.w *= BigRat(2, 3);

  const BigQuat &outer_rot = oq;
  const BigQuat &inner_rot = iq;

  {
    // Convert back and render.
    Polyhedron poly = SmallPoly(ridode);

    const frame3 outer_frame =
      yocto::rotation_frame(normalize(SmallQuat(outer_rot)));
    // Note: ignoring translation
    const frame3 inner_frame =
      yocto::rotation_frame(normalize(SmallQuat(inner_rot)));

    Polyhedron outer = Rotate(poly, outer_frame);
    Polyhedron inner = Rotate(poly, inner_frame);
    Mesh2D souter = Shadow(outer);
    Mesh2D sinner = Shadow(inner);
    // std::vector<int> outer_hull = QuickHull(souter.vertices);
    // std::vector<int> inner_hull = QuickHull(sinner.vertices);

    Rendering rendering(poly, 3840, 2160);
    rendering.RenderTriangulation(souter, 0xAA0000FF);
    rendering.RenderTriangulation(sinner, 0x00FF00AA);
    // This looks correct.
    rendering.Save(std::format("validate-recreate-{}.png", poly.name));
  }

  printf("Made BigQuats\n");

  BigFrame big_outer_frame = NonUnitRotationFrame(outer_rot);
  BigFrame big_inner_frame = NonUnitRotationFrame(inner_rot);

  // BigPoly outer = Rotate(outer_rot, ridode);
  // BigPoly inner = Rotate(inner_rot, ridode);

  BigPoly outer = Rotate(big_outer_frame, ridode);
  BigPoly inner = Rotate(big_inner_frame, ridode);

  {
    Polyhedron poly = SmallPoly(inner);
    Mesh2D souter = Shadow(poly);
    Rendering rendering(poly, 1920, 1080);
    rendering.RenderTriangulation(souter, 0xAA0000FF);
    // This looks correct.
    rendering.Save(std::format("validate-rotated-{}.png", poly.name));
  }

  printf("Rotated\n");

  BigMesh2D souter = Shadow(outer);
  BigMesh2D sinner = Translate(itrans, Shadow(inner));

  printf("Check:\n");
  // Now check
  bool valid = true;
  std::vector<int> ins, outs;

  Polyhedron renderpoly = SmallPoly(ridode);
  Mesh2D small_souter = SmallMesh(souter);
  Mesh2D small_sinner = SmallMesh(sinner);
  auto RenderInside = [&](
      const std::optional<std::tuple<int, int, int>> &triangle,
      int ptidx) {

      Rendering rendering(renderpoly, 1920, 1080);

      // Polyhedron outer = Rotate(renderpoly, sol.outer_frame);
      // Polyhedron inner = Rotate(renderpoly, sol.inner_frame);
      // Mesh2D souter = Shadow(outer);
      // Mesh2D sinner = Shadow(inner);

      rendering.RenderMesh(small_souter);
      rendering.DarkenBG();

      if (triangle.has_value()) {
        const auto &[a, b, c] = triangle.value();
        rendering.RenderTriangle(small_souter, a, b, c, 0x3333AAAA);
        rendering.MarkPoints(small_sinner, {ptidx}, 20.0f, 0x00FF00AA);
      } else {
        rendering.MarkPoints(small_sinner, {ptidx}, 20.0f, 0xFF0000AA);
      }

      rendering.Save(StringPrintf("validate-%d.png", ptidx));
    };

  // std::vector<int> inner_hull = BigHull(sinner.vertices);

  // for (int i = 0; i < sinner.vertices.size(); i++) {
  // for (int i : inner_hull) {
  for (int i = 0; i < sinner.vertices.size(); i++) {
    const BigVec2 &v = sinner.vertices[i];
    const std::optional<std::tuple<int, int, int>> triangle =
      InMeshExhaustive(souter, v);
    bool in = triangle.has_value();
    printf("Point %d is %s\n", i, in ? AGREEN("in") : ARED("out"));
    valid = valid && in;
    if (in) {
      RenderInside(triangle, i);
      ins.push_back(i);
    } else {
      int closest = GetClosestPoint(souter, v);
      printf("  Point at %s.\n"
             "  Closest: %s\n",
             VecString(v).c_str(),
             VecString(souter.vertices[closest]).c_str());
      outs.push_back(i);
    }
  }

  {
    Rendering rendering(renderpoly, 1920, 1080);

    rendering.RenderTriangulation(small_souter, 0xFF000044);
    rendering.RenderTriangulation(small_sinner, 0x00FF0044);

    rendering.MarkPoints(small_sinner, ins, 10.0f, 0x22FF22AA);
    rendering.MarkPoints(small_sinner, outs, 20.0f, 0xFF2211AA);
    rendering.Save("validate.png");
  }

  printf("Done\n");
}

int main(int argc, char **argv) {
  ANSI::Init();

  Validate();

  printf("Done.\n");
  return 0;
}
