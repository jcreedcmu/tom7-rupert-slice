
#ifndef _RUPERTS_SOLUTIONS_H
#define _RUPERTS_SOLUTIONS_H

#include <cstdint>
#include <ctime>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "auto-histo.h"
#include "base/logging.h"
#include "database.h"
#include "polyhedra.h"
#include "yocto_matht.h"

struct SolutionDB {
  using vec3 = yocto::vec<double, 3>;
  using frame3 = yocto::frame<double, 3>;

  static constexpr const char *DBFILE = "ruperts.sqlite";

  static constexpr int METHOD_HULL = 1;
  static constexpr int METHOD_SIMUL = 2;
  static constexpr int METHOD_MAX = 3;
  static constexpr int METHOD_PARALLEL = 4;
  static constexpr int METHOD_GPU1 = 5;
  static constexpr int METHOD_SPECIAL = 6;
  static constexpr int METHOD_ORIGIN = 7;
  static constexpr int METHOD_ALMOST_ID = 8;

  static const char *MethodName(int m) {
    switch (m) {
    case METHOD_HULL: return "METHOD_HULL";
    case METHOD_SIMUL: return "METHOD_SIMUL";
    case METHOD_MAX: return "METHOD_MAX";
    case METHOD_PARALLEL: return "METHOD_PARALLEL";
    case METHOD_GPU1: return "METHOD_GPU1";
    case METHOD_SPECIAL: return "METHOD_SPECIAL";
    case METHOD_ORIGIN: return "METHOD_ORIGIN";
    case METHOD_ALMOST_ID: return "METHOD_ALMOST_ID";
    default: return "UNKNOWN";
    }
  }

  using Query = Database::Query;
  using Row = Database::Row;

  SolutionDB() {
    db = Database::Open(DBFILE);
    CHECK(db.get() != nullptr) << DBFILE;

    Init();
  }

  static std::string FrameString(const frame3 &frame);
  static std::optional<frame3> StringFrame(const std::string &s);

  void Init();

  struct Solution {
    std::string polyhedron;
    frame3 outer_frame;
    frame3 inner_frame;
    int method = 0;
    int64_t createdate = 0;
    // The ratio of the shadow's areas (note this is not the
    // same as the volume!)
    double ratio = 0.0;
  };

  struct Attempt {
    std::string polyhedron;
    int method = 0;
    int64_t createdate = 0;
    double best_error = 0.0;
    int64_t iters = 0;
    int64_t evals = 0;
  };

  std::vector<Solution> GetAllSolutions();
  std::vector<Attempt> GetAllAttempts();

  std::vector<Solution> GetSolutionsFor(const std::string &name);

  void AddSolution(const std::string &polyhedron,
                   const frame3 &outer_frame,
                   const frame3 &inner_frame,
                   int method,
                   double ratio);

  void AddAttempt(const std::string &poly, int method,
                  double best_error, int64_t iters,
                  int64_t evals);

  static constexpr int NOPERT_METHOD_RANDOM = 1;
  void AddNopertAttempt(int points, int64_t attempts,
                        const AutoHisto &iterhisto,
                        int method);

  void AddNopert(const Polyhedron &poly, int method);

  void ExecuteAndPrint(const std::string &s) {
    db->ExecuteAndPrint(s);
  }

 private:
  std::unique_ptr<Database> db;
};


#endif
