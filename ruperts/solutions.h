
#ifndef _RUPERTS_SOLUTIONS_H
#define _RUPERTS_SOLUTIONS_H

#include <cstdint>
#include <ctime>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "auto-histo.h"
#include "base/logging.h"
#include "base/stringprintf.h"
#include "database.h"
#include "polyhedra.h"
#include "yocto_matht.h"

struct SolutionDB {
  using vec3 = yocto::vec<double, 3>;
  using frame3 = yocto::frame<double, 3>;

  // static constexpr const char *DBFILE = "ruperts.sqlite";

  static constexpr int METHOD_INVALID = 0;
  static constexpr int METHOD_HULL = 1;
  static constexpr int METHOD_SIMUL = 2;
  static constexpr int METHOD_MAX = 3;
  static constexpr int METHOD_PARALLEL = 4;
  static constexpr int METHOD_GPU1 = 5;
  static constexpr int METHOD_SPECIAL = 6;
  static constexpr int METHOD_ORIGIN = 7;
  static constexpr int METHOD_ALMOST_ID = 8;
  static constexpr int METHOD_IMPROVE_RATIO = 9;
  static constexpr int METHOD_RATIONAL_OLD = 10;
  static constexpr int METHOD_RATIONAL = 11;
  static constexpr int METHOD_IMPROVE_CLEARANCE = 12;

  static constexpr int FIRST_METHOD = 1;
  static constexpr int LAST_METHOD = 12;

  static const char *MethodName(int m) {
    switch (m) {
    case METHOD_INVALID: return "INVALID";
    case METHOD_HULL: return "METHOD_HULL";
    case METHOD_SIMUL: return "METHOD_SIMUL";
    case METHOD_MAX: return "METHOD_MAX";
    case METHOD_PARALLEL: return "METHOD_PARALLEL";
    case METHOD_GPU1: return "METHOD_GPU1";
    case METHOD_SPECIAL: return "METHOD_SPECIAL";
    case METHOD_ORIGIN: return "METHOD_ORIGIN";
    case METHOD_ALMOST_ID: return "METHOD_ALMOST_ID";
    case METHOD_IMPROVE_RATIO: return "METHOD_IMPROVE_RATIO";
    case METHOD_RATIONAL_OLD: return "METHOD_RATIONAL_OLD";
    case METHOD_RATIONAL: return "METHOD_RATIONAL";
    case METHOD_IMPROVE_CLEARANCE: return "METHOD_IMPROVE_CLEARANCE";
    default: return "UNKNOWN";
    }
  }

  using Query = Database::Query;
  using Row = Database::Row;

  // Also looks for the database in parent directories
  // before creating it.
  SolutionDB(const char *dbfile = "ruperts.sqlite");

  static std::string FrameString(const frame3 &frame);
  static std::optional<frame3> StringFrame(const std::string &s);

  void Init();

  struct Solution {
    int id = 0;
    std::string polyhedron;
    frame3 outer_frame;
    frame3 inner_frame;
    int method = 0;
    int source = 0;
    int64_t createdate = 0;
    // The ratio of the shadow's areas (note this is not the
    // same as the volume!)
    double ratio = 0.0;
    // Minimum clearance between the inner and outer hulls;
    // this is scale-dependent.
    double clearance = 0.0;
  };

  struct Attempt {
    int id = 0;
    std::string polyhedron;
    int method = 0;
    int source = 0;
    int64_t createdate = 0;
    double best_error = 0.0;
    int64_t iters = 0;
    int64_t evals = 0;
  };

  struct Nopert {
    int id = 0;
    std::vector<vec3> vertices;
    int64_t createdate = 0;
    int method = 0;
  };

  struct NopertAttempt {
    int id = 0;
    int points = 0;
    int64_t attempts = 0;
    std::string iterhisto;
    int64_t createdate = 0;
    int method = 0;
  };

  // By default these do not return solutions marked as invalid.
  std::vector<Solution> GetAllSolutions();
  std::vector<Attempt> GetAllAttempts();
  // Or abort. Allows fetching a solution marked invalid.
  Solution GetSolution(int id);

  std::vector<Solution> GetSolutionsFor(std::string_view name);
  // Or abort.
  Solution GetBestSolutionFor(std::string_view name,
                              // false = lowest ratio
                              // true = highest clearance
                              bool use_clearance = false);
  std::vector<Solution> GetAllNopertSolutions();

  std::pair<
    // By lowest ratio
    std::unordered_map<std::string, Solution>,
    // By highest clearance
    std::unordered_map<std::string, Solution>> BestSolutions();

  static std::string NopertName(int id) {
    return StringPrintf("nopert_%d", id);
  }

  void AddSolution(const std::string &polyhedron,
                   const frame3 &outer_frame,
                   const frame3 &inner_frame,
                   int method, int source,
                   double ratio, double clearance);

  void AddAttempt(const std::string &poly, int method,
                  int source,
                  double best_error, int64_t iters,
                  int64_t evals);

  static constexpr int NOPERT_METHOD_RANDOM = 1;
  static constexpr int NOPERT_METHOD_CYCLIC = 2;
  static constexpr int NOPERT_METHOD_SYMMETRIC = 3;
  static constexpr int NOPERT_METHOD_REDUCE_SC = 4;
  static constexpr int NOPERT_METHOD_ADVERSARY = 5;
  static constexpr int NOPERT_METHOD_RHOMBIC = 6;
  static constexpr int NOPERT_METHOD_UNOPT = 7;
  static constexpr int NOPERT_METHOD_CHURRO = 8;

  static const char *NopertMethodName(int m) {
    switch (m) {
    case NOPERT_METHOD_RANDOM: return "NOPERT_METHOD_RANDOM";
    case NOPERT_METHOD_CYCLIC: return "NOPERT_METHOD_CYCLIC";
    case NOPERT_METHOD_SYMMETRIC: return "NOPERT_METHOD_SYMMETRIC";
    case NOPERT_METHOD_REDUCE_SC: return "NOPERT_METHOD_REDUCE_SC";
    case NOPERT_METHOD_ADVERSARY: return "NOPERT_METHOD_ADVERSARY";
    case NOPERT_METHOD_RHOMBIC: return "NOPERT_METHOD_RHOMBIC";
    case NOPERT_METHOD_UNOPT: return "NOPERT_METHOD_UNOPT";
    case NOPERT_METHOD_CHURRO: return "NOPERT_METHOD_CHURRO";
    default: return "UNKNOWN";
    }
  }

  void AddNopertAttempt(int points, int64_t attempts,
                        const AutoHisto &iterhisto,
                        int method);

  std::vector<NopertAttempt> GetAllNopertAttempts();

  void AddNopert(const Polyhedron &poly, int method);
  // Or abort.
  Nopert GetNopert(int id);

  std::vector<Nopert> GetAllNoperts();

  void ExecuteAndPrint(const std::string &s) {
    db->ExecuteAndPrint(s);
  }

  // Get a built-in polyhedron ("cube", etc.) or a nopert
  // from the database ("nopert_37", etc.). Aborts if not found.
  Polyhedron AnyPolyhedronByName(std::string_view name);

 private:
  std::unique_ptr<Database> db;
};


#endif
