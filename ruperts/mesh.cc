
#include "mesh.h"

#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>

#include "base/logging.h"
#include "hashing.h"
#include "sorting-network.h"
#include "util.h"
#include "yocto_matht.h"

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

enum FaceType {
  ALL_ABOVE,
  ALL_BELOW,
  MIXED,
};

static FaceType ClassifyTriangle(const TriangularMesh3D &mesh,
                                 int a, int b, int c) {
  const vec3 &v0 = mesh.vertices[a];
  const vec3 &v1 = mesh.vertices[b];
  const vec3 &v2 = mesh.vertices[c];

  const vec3 normal =
    yocto::normalize(yocto::cross(v1 - v0, v2 - v0));

  bool above = false, below = false;

  // Now for every other vertex, check which side they are on.
  for (int o = 0; o < mesh.vertices.size(); o++) {
    if (o != a && o != b && o != c) {
      const vec3 &v = mesh.vertices[o];
      double dot = yocto::dot(v - v0, normal);
      if (dot < -0.00001) {
        if (above) return MIXED;
        below = true;
      } else if (dot > 0.00001) {
        if (below) return MIXED;
        above = true;
      } else {
        // On plane.
        continue;
      }
    }
  }

  CHECK(above || below) << "Degenerate: All faces are coplanar.";
  CHECK(!(above && below)) << "Impossible";
  if (above) return ALL_ABOVE;
  else return ALL_BELOW;
}

static std::tuple<int, int, int> GetOrientedTriangle(const TriangularMesh3D &mesh) {
  // Start by orienting one triangle. We find one for which the
  // entire rest of the mesh is on one side of it (or coplanar).

  for (const auto &[a, b, c] : mesh.triangles) {
    switch (ClassifyTriangle(mesh, a, b, c)) {
    case ALL_ABOVE:
      return std::make_tuple(a, b, c);
    case ALL_BELOW:
      return std::make_tuple(c, b, a);
    case MIXED:
      // Try the next one.
      continue;
    }
  }

  LOG(FATAL) << "No triangle could be oriented? This is bad news!";
}

static inline bool SameTriangle(int a, int b, int c,
                                int aa, int bb, int cc) {
  auto t1 = std::make_tuple(a, b, c);
  auto t2 = std::make_tuple(aa, bb, cc);

  FixedSort<3>(&t1);
  FixedSort<3>(&t2);
  return t1 == t2;
}

void OrientMesh(TriangularMesh3D *mesh) {
  const auto &[oa, ob, oc] = GetOrientedTriangle(*mesh);

  // Now iteratively orient triangles that share an edge.

  // Edges that have been processed, in *forward* order.
  // (i.e. we already inserted a triangle in the correct orientation
  // a-b-c, and so this set contains a-b. etc.)
  std::unordered_set<std::pair<int, int>, Hashing<std::pair<int, int>>>
    edges;

  std::vector<std::tuple<int, int, int>> out, remaining;
  out.reserve(mesh->triangles.size());
  remaining.reserve(mesh->triangles.size());

  auto AddTriangle = [&](int a, int b, int c) {
      out.emplace_back(a, b, c);
      edges.insert(std::make_pair(a, b));
      edges.insert(std::make_pair(b, c));
      edges.insert(std::make_pair(c, a));
    };

  for (const auto &[a, b, c] : mesh->triangles) {
    if (SameTriangle(oa, ob, oc, a, b, c)) {
      AddTriangle(a, b, c);
    } else {
      remaining.emplace_back(a, b, c);
    }
  }

  while (!remaining.empty()) {
    bool progress = false;
    std::vector<std::tuple<int, int, int>> new_remaining;
    for (const auto &[a, b, c] : remaining) {
      if (edges.contains({b, a}) ||
          edges.contains({c, b}) ||
          edges.contains({a, c})) {
        // Already in correct orientation
        AddTriangle(a, b, c);
        progress = true;
      } else if (edges.contains({a, b}) ||
                 edges.contains({b, c}) ||
                 edges.contains({c, a})) {
        // Reversed.
        AddTriangle(c, b, a);
        progress = true;
      } else {
        new_remaining.emplace_back(a, b, c);
      }
    }

    CHECK(progress) << "Surface is disconnected";
    remaining = std::move(new_remaining);
  }

  CHECK(out.size() == mesh->triangles.size()) <<
    mesh->triangles.size() << " became " << out.size();
  mesh->triangles = std::move(out);
}
