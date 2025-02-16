
#include "mesh.h"

#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "base/logging.h"
#include "yocto_matht.h"
#include "util.h"

using vec3 = yocto::vec<double, 3>;

TriangularMesh3D LoadSTL(std::string_view filename) {
  /*
  solid mesh
  facet normal -0.2838546953049397 -0.6963007001010147 0.659235805302
    outer loop
      vertex -0.874933610646791 1.4504050705684777 -0.36168537187702393
      vertex -0.7418220995666276 1.0305256485708716 -0.7478567630973156
      vertex -0.19613114259134246 1.0086609614714392 -0.5359863788361463
    endloop
  endfacet
  */

  TriangularMesh3D mesh;
  auto AddVertex = [&mesh](const vec3 &v) {
      for (int i = 0; i < mesh.vertices.size(); i++) {
        if (mesh.vertices[i] == v) return i;
      }

      int next = (int)mesh.vertices.size();
      mesh.vertices.push_back(v);
      return next;
    };


  std::vector<std::string> lines = Util::ReadFileToLines(filename);
  CHECK(!lines.empty()) << filename;

  CHECK(Util::TryStripPrefix("solid ", &lines[0])) << filename;
  // Then just ignore the rest of the line.
  lines[0] = "";

  // The remainder of the file is not whitespace sensitive, so turn it
  // into one long line.
  std::string stl_contents = Util::NormalizeWhitespace(Util::Join(lines, " "));
  std::string_view stl{stl_contents};

  auto Vec3 = [&stl, filename]() {
      std::string_view x = Util::NextToken(&stl, ' ');
      CHECK(!x.empty()) << filename;
      std::string_view y = Util::NextToken(&stl, ' ');
      CHECK(!y.empty()) << filename;
      std::string_view z = Util::NextToken(&stl, ' ');
      CHECK(!z.empty()) << filename;

      const std::optional<double> xo = Util::ParseDouble(x);
      const std::optional<double> yo = Util::ParseDouble(y);
      const std::optional<double> zo = Util::ParseDouble(z);
      CHECK(xo.has_value() && yo.has_value() && zo.has_value());
      return vec3(xo.value(), yo.value(), zo.value());
    };

  for (;;) {
    //   facet normal -0.2838546953049397 -0.6963007001010147 0.659235805302
    std::string_view cmd = Util::NextToken(&stl, ' ');
    CHECK(!cmd.empty()) << filename;
    if (cmd == "endsolid") break;

    CHECK(cmd == "facet") << filename;
    CHECK(Util::NextToken(&stl, ' ') == "normal");
    // Ignore normal.
    (void)Vec3();

    CHECK(Util::NextToken(&stl, ' ') == "outer");
    CHECK(Util::NextToken(&stl, ' ') == "loop");

    CHECK(Util::NextToken(&stl, ' ') == "vertex");
    int a = AddVertex(Vec3());
    CHECK(Util::NextToken(&stl, ' ') == "vertex");
    int b = AddVertex(Vec3());
    CHECK(Util::NextToken(&stl, ' ') == "vertex");
    int c = AddVertex(Vec3());
    mesh.triangles.emplace_back(a, b, c);

    CHECK(Util::NextToken(&stl, ' ') == "endloop");
    CHECK(Util::NextToken(&stl, ' ') == "endfacet");
  }

  return mesh;
}
