
#include "tree-nd.h"

#include <span>
#include <string>
#include <unordered_set>

#include "ansi.h"
#include "arcfour.h"
#include "base/logging.h"
#include "periodically.h"
#include "randutil.h"
#include "status-bar.h"

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

  std::unordered_set<std::string> saw;
  tree.App([&saw](std::span<const int> pos, const std::string &s) {
      CHECK(!saw.contains(s));
      saw.insert(s);
      if (s == "i") {
        CHECK(pos[0] == -99);
        CHECK(pos[1] == 300);
        CHECK(pos[2] == 0);
      }
    });

  for (const auto &[x, y, z, s] : points) {
    CHECK(saw.contains(s));
  }

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

static void TestDouble3D() {
  ArcFour rc("deterministic");

  using vec3 = std::vector<double>;
  auto Vec3 = [](double x, double y, double z) -> vec3 {
      return vec3{{x, y, z}};
    };

  auto Distance = [](const vec3 &a, const vec3 &b) {
      auto dx = a[0] - b[0];
      auto dy = a[1] - b[1];
      auto dz = a[2] - b[2];
      return std::sqrt(dx * dx + dy * dy + dz * dz);
    };

  double lookup_sec = 0.0;

  StatusBar status(1);
  Periodically status_per(1.0);
  static constexpr int NUM_PASSES = 300;
  for (int pass = 0; pass < NUM_PASSES; pass++) {
    TreeND<double, int> tree(3);

    std::vector<std::pair<vec3, int>> points;
    for (int i = 0; i < 1000; i++) {
      double x = RandDouble(&rc) * 4 - 2;
      double y = RandDouble(&rc) * 4 - 2;
      double z = RandDouble(&rc) * 4 - 2;

      // 1/3 of the time, all on a sphere
      if (pass % 3 == 0) {
        double norm = Distance(Vec3(0, 0, 0), Vec3(x, y, z));
        x /= norm;
        y /= norm;
        z /= norm;
      }

      vec3 v = Vec3(x, y, z);
      points.emplace_back(v, i);
      tree.Insert(v, i);
    }

    CHECK(tree.Size() == points.size());

    Timer lookup_timer;
    for (const auto &[pos, idx] : points) {
      // Note that there could technically be duplicate points.
      const auto &[pp, ii, dd] = tree.Closest(pos);
      CHECK(std::abs(dd) < 1.0e-6);
      CHECK(Distance(points[ii].first, pos) < 1.0e-6);
    }
    lookup_sec += lookup_timer.Seconds();

    // Random points
    for (int r = 0; r < 1000; r++) {
      double x = RandDouble(&rc) * 4 - 2;
      double y = RandDouble(&rc) * 4 - 2;
      double z = RandDouble(&rc) * 4 - 2;
      vec3 v = Vec3(x, y, z);

      Timer tt;
      const auto &[pp, ii, dd] = tree.Closest(v);
      lookup_sec += tt.Seconds();
      CHECK(std::abs(dd - Distance(v, pp)) < 1.0e-6);

      // Make sure nothing is strictly closer
      for (const auto &[pos, idx] : points) {
        if (idx == ii) continue;
        CHECK(Distance(v, pos) >= dd);
      }
    }

    // XXX check LookUp too.
    status_per.RunIf([&]() {
        status.Progressf(pass, NUM_PASSES, "Testing...");
      });
  }

  printf("Total Closest time: %s\n", ANSI::Time(lookup_sec).c_str());
}

int main(int argc, char **argv) {
  ANSI::Init();

  TestInts2D();
  TestInts3D();

  TestDouble3D();

  // TODO: Some random tests?

  printf("OK");
  return 0;
}
