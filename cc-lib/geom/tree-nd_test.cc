
#include "tree-nd.h"

#include <string>

#include "base/logging.h"

static std::vector<int> Pos3(int x, int y, int z) {
  return std::vector<int>({x, y, z});
}

// Test from 2D tree, but using 3 dimensions; everything in plane.
static void TestInts2D() {
  TreeND<int, std::string> tree(3);
  CHECK(tree.Size() == 0);
  CHECK(tree.Empty());

  using Pos = TreeND<int, std::string>::Pos;

  std::vector<std::tuple<int, int, int, std::string>> points = {
    {0, 4, 0, "a"},
    {-1, -1, 0, "b"},
    {100, 300, 0, "c"},
    {100, 301, 0, "d"},
    {100, 299, 0, "e"},
    {-100, 300, 0, "f"},
    {-100, 299, 0, "g"},
    {-100, 301, 0, "h"},
    {-99, 300, 0, "i"},
    {-99, 299, 0, "j"},
    {-99, 301, 0, "k"},
    {-101, 300, 0, "l"},
    {-101, 299, 0, "m"},
    {-101, 301, 0, "n"},
  };

  for (const auto &[x, y, z, s] : points) {
    tree.Insert(Pos3(x, y, z), s);
  }

  tree.DebugPrint();

  for (const auto &[x, y, z, s] : points) {
    CHECK(z == 0);

    for (int radius : {1, 2, 10}) {
      for (int d : {0, 1, -1}) {
        for (bool dir : {false, true}) {
          int qx = x + (dir ? d * radius : 0.0);
          int qy = y + (dir ? 0.0 : d * radius);
          std::vector<std::tuple<Pos, std::string, double>> near =
            tree.LookUp(Pos3(qx, qy, 0), radius);
          bool has = [&]() {
              for (const auto &[pos, ss, dist] : near) {
                CHECK(pos.size() == 3);
                CHECK(pos[2] == 0);
                if (pos[0] == x && pos[1] == y && ss == s)
                  return true;
              }
              return false;
            }();

          CHECK(has) << "Didn't find point " << s << " with radius "
                     << radius << ". The query was (" << qx << ","
                     << qy << "). Num near: " << near.size();
        }
      }
    }

  }

  CHECK(tree.Size() == points.size());
  CHECK(!tree.Empty());

  {
    const auto &[p, s, d] = tree.Closest(Pos3(0, 0, 0));
    CHECK(p == Pos3(-1, -1, 0));
    CHECK(s == "b");
    CHECK(d > 1.4 && d < 1.5);
  }

  {
    const auto &[p, s, d] = tree.Closest(Pos3(9999, 9999, 0));
    CHECK(p == Pos3(100, 301, 0));
    CHECK(s == "d");
    CHECK(d > 9999);
  }

  for (const auto &[x, y, z, s] : points) {
    CHECK(tree.Remove(Pos3(x, y, 0))) << x << "," << y;
  }

  CHECK(tree.Size() == 0);
  CHECK(tree.Empty());

  // They should already be gone.
  for (const auto &[x, y, z, s] : points) {
    CHECK(!tree.Remove(Pos3(x, y, z))) << x << "," << y;
  }

  CHECK(tree.Size() == 0);
  CHECK(tree.Empty());
}

// Basic 3D test.
static void TestInts3D() {
  TreeND<int, std::string> tree(3);

  std::vector<std::tuple<int, int, int, std::string>> points = {
    {1, 1, 1, "a"},
    {-1, 2, -1, "b"},
    {0, 0, 0, "c"},
    {10, 10, 10, "d"},
    {2, 1, 1, "e"},
  };

  for (const auto &[x, y, z, s] : points) {
    tree.Insert(Pos3(x, y, z), s);
  }
  CHECK(tree.Size() == points.size());

  // C is at the origin.
  {
    auto near = tree.LookUp(Pos3(0, 0, 0), 1.5);
    CHECK(near.size() == 1);
    const auto &[p, s, d] = near[0];
    CHECK(p == Pos3(0, 0, 0));
    CHECK(s == "c");
    CHECK(d == 0.0);
  }

  {
    auto near = tree.LookUp(Pos3(1, 1, 1), 1.1);
    // Should find 'a' (dist 0) and 'e' (dist 1)
    CHECK(near.size() == 2);
    bool found_a = false;
    bool found_e = false;
    for (const auto &[p, s, d] : near) {
      if (s == "a") {
        CHECK(p == Pos3(1, 1, 1));
        CHECK(d == 0.0);
        found_a = true;
      } else if (s == "e") {
        CHECK(p == Pos3(2, 1, 1));
        CHECK(std::abs(d - 1.0) < 1e-9);
        found_e = true;
      }
    }
    CHECK(found_a);
    CHECK(found_e);
  }

  // Closest to origin (exact)
  {
    const auto &[p, s, d] = tree.Closest(Pos3(0, 0, 0));
    CHECK(p == Pos3(0, 0, 0));
    CHECK(s == "c");
    CHECK(d == 0.0);
  }

  // Closest to origin
  {
    const auto &[p, s, d] = tree.Closest(Pos3(0, 0, 1));
    CHECK(p == Pos3(0, 0, 0));
    CHECK(s == "c");
    CHECK(std::abs(d - 1.0) < 1e-9);
  }

  {
    const auto &[p, s, d] = tree.Closest(Pos3(1, 3, 0));
    CHECK(p == Pos3(1, 1, 1));
    CHECK(s == "a");
    CHECK(std::abs(d - sqrt(0 + 2 * 2 + 1 * 1)) < 1e-9);
  }

  // Closest to far point
  {
    const auto &[p, s, d] = tree.Closest(Pos3(100, 100, 100));
    CHECK(p == Pos3(10, 10, 10));
    CHECK(s == "d");
    CHECK(std::abs(d - sqrt(90 * 90 + 90 * 90 + 90 * 90)) < 1e-9);
  }

  // Remove points
  for (const auto &[x, y, z, s] : points) {
    CHECK(tree.Remove(Pos3(x, y, z)));
  }

  CHECK(tree.Size() == 0);
  CHECK(tree.Empty());
}

int main(int argc, char **argv) {

  TestInts2D();
  TestInts3D();

  // TODO: Some random tests?

  printf("OK");
  return 0;
}
