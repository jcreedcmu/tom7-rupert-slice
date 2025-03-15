
#ifndef _RUPERTS_SHRINKLUTIONS_H
#define _RUPERTS_SHRINKLUTIONS_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "base/logging.h"
#include "database.h"
#include "yocto_matht.h"

struct ShrinklutionDB {
  using vec3 = yocto::vec<double, 3>;
  using frame3 = yocto::frame<double, 3>;

  static constexpr const char *DBFILE = "shrink.sqlite";

  static constexpr int METHOD_INVALID = 0;
  static constexpr int METHOD_RANDOM = 1;
  static constexpr int METHOD_MANUAL = 2;
  static constexpr int METHOD_SAME_ANGLE = 3;

  static const char *MethodName(int m) {
    switch (m) {
    case METHOD_INVALID: return "INVALID";
    case METHOD_RANDOM: return "METHOD_RANDOM";
    case METHOD_MANUAL: return "METHOD_MANUAL";
    case METHOD_SAME_ANGLE: return "METHOD_SAME_ANGLE";
    default: return "UNKNOWN";
    }
  }

  using Query = Database::Query;
  using Row = Database::Row;

  ShrinklutionDB() {
    db = Database::Open(DBFILE);
    CHECK(db.get() != nullptr) << DBFILE;

    Init();
  }

  static std::string FrameString(const frame3 &frame);
  static std::optional<frame3> StringFrame(std::string_view s);

  static std::string CubesString(const std::vector<frame3> &cubes);
  static std::optional<std::vector<frame3>> StringCubes(std::string_view s);

  void Init();

  struct Solution {
    int id = 0;
    int num = 0;
    std::vector<frame3> cubes;
    int method = 0;
    int source = 0;
    int64_t createdate = 0;
    double radius = 0;
  };

  // By default these do not return solutions marked as invalid.
  std::vector<Solution> GetAllSolutions();
  // Or abort. Allows fetching a solution marked invalid.
  Solution GetSolution(int id);

  std::vector<Solution> GetSolutionsFor(int num);
  // Or abort.
  Solution GetBestSolutionFor(int num);

  // To avoid reinserting a manual solution, for example.
  // Tolerance is 1e-10.
  bool HasSolutionWithRadius(int num, double radius);

  void AddSolution(int num,
                   const std::vector<frame3> &cubes,
                   int method, int source,
                   double radius);

  template<size_t NUM_CUBES>
  void AddSolution(const std::array<frame3, NUM_CUBES> &cubes,
                   int method, int source,
                   double radius) {
    std::vector<frame3> vcubes;
    for (int i = 0; i < NUM_CUBES; i++) {
      vcubes.push_back(cubes[i]);
    }
    return AddSolution(NUM_CUBES, vcubes, method, source, radius);
  }


  void ExecuteAndPrint(const std::string &s) {
    db->ExecuteAndPrint(s);
  }

 private:
  std::unique_ptr<Database> db;
};


#endif
