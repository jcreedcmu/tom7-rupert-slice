
#include "ansi.h"
#include <cstdio>
#include <optional>
#include <vector>

#include "polyhedra.h"
#include "yocto_matht.h"
#include "rendering.h"

using vec2 = yocto::vec<double, 2>;
using vec3 = yocto::vec<double, 3>;
using vec4 = yocto::vec<double, 4>;
using mat4 = yocto::mat<double, 4>;
using quat4 = yocto::quat<double, 4>;
using frame3 = yocto::frame<double, 3>;

static void DTest() {

  const Polyhedron poly = []() {
      auto opoly = PolyhedronFromConvexVertices(
          std::vector<vec3>{
            {-0.7888254090932487,0.1809666339064286,0.5873717318542367},
            {-0.7275239611943676,-0.5176379718006908,-0.45028859194756815},
            {-0.46761652842176743,-8.458653528836922e-05,0.8839314312727501},
            {0.012255643227897352,0.6097001740588195,0.7925373789050683},
            {-0.39543790974206744,-0.460739697734118,0.7945739679039946},
            {-0.6113020439400112,0.2438926374222257,-0.752878604083019}
          });

      CHECK(opoly.has_value());
      return opoly.value();
    }();

  frame3 outer_frame =
    frame3{vec3{-0.28122289782544885,0.8771668061830272,-0.3892198297483371},
           vec3{0.9013115757915298,0.10217934077188853,-0.4209475331482672},
           vec3{-0.3294709776032578,-0.46918842318421455,-0.8193357666226122},
           vec3{0,0,0}};
  frame3 inner_frame =
    frame3{vec3{-0.9900711776109041,0.10560397200855935,0.09277318772238792},
           vec3{-0.10077001875482698,-0.9933693480580881,0.05534204241624576},
           vec3{0.09800238050328079,0.04544380523971492,0.9941480744743338},
           vec3{-0.4725374179836337,-0.20260639570538497,0}};

  Rendering rendering(poly, 1920, 1080);
  rendering.RenderSolution(poly, outer_frame, inner_frame);
  rendering.Save("dtest.png");

  double loss = LossFunction(poly, outer_frame, inner_frame);
  printf("loss = %.17g\n", loss);

  double loss_origin =
    LossFunctionContainsOrigin(poly, outer_frame, inner_frame);
  printf("loss (assuming contains origin) = %.17g\n", loss_origin);

  std::optional<double> oclearance =
    GetClearance(poly, outer_frame, inner_frame);
  if (oclearance.has_value()) {
    printf("clearance = %.17g\n", oclearance.value());
  } else {
    printf("Clearance: Invalid!\n");
  }

  // SaveAsJSON(poly, "/tmp/test.json");
}

int main(int argc, char **argv) {
  ANSI::Init();
  printf("-----------\n");

  DTest();

  return 0;
}
