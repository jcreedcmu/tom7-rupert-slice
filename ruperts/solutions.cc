#include "solutions.h"

#include <cstdint>
#include <ctime>
#include <format>
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
#include "util.h"

using frame3 = SolutionDB::frame3;

using Solution = SolutionDB::Solution;

SolutionDB::SolutionDB(const char *dbfile) {
  std::string f = dbfile;
  // XXX would be good if we could determine when this
  // has reached the root?
  for (int i = 0; i < 5; i++) {
    if (Util::ExistsFile(f)) {
      db = Database::Open(f);
      CHECK(db.get() != nullptr) << f;
      break;
    }

    f = StringPrintf("../%s", f.c_str());
  }

  if (db.get() == nullptr) {
    // Otherwise create it here.
    db = Database::Open(dbfile);
    CHECK(db.get() != nullptr) << dbfile;
  }

  Init();
}


std::string SolutionDB::FrameString(const frame3 &frame) {
  return StringPrintf(
      // x
      "%.17g,%.17g,%.17g,"
      // y
      "%.17g,%.17g,%.17g,"
      // z
      "%.17g,%.17g,%.17g,"
      // o
      "%.17g,%.17g,%.17g",
      frame.x.x, frame.x.y, frame.x.z,
      frame.y.x, frame.y.y, frame.y.z,
      frame.z.x, frame.z.y, frame.z.z,
      frame.o.x, frame.o.y, frame.o.z);
}


std::optional<frame3> SolutionDB::StringFrame(const std::string &s) {
  std::vector<std::string> parts = Util::Split(s, ',');
  if (parts.size() != 12) return std::nullopt;
  std::vector<double> ds;
  for (const std::string &s : parts) {
    if (std::optional<double> od = Util::ParseDoubleOpt(s)) {
      ds.push_back(od.value());
    } else {
      return std::nullopt;
    }
  }

  CHECK(ds.size() == 12);
  return {frame3{
      .x = vec3{ds[0], ds[1], ds[2]},
      .y = vec3{ds[3], ds[4], ds[5]},
      .z = vec3{ds[6], ds[7], ds[8]},
      .o = vec3{ds[9], ds[10], ds[11]},
    }};
}

void SolutionDB::Init() {
  db->ExecuteAndPrint("create table "
                      "if not exists "
                      "solutions ("
                      "id integer primary key, "
                      "polyhedron string not null, "
                      // frames as strings
                      "outerframe string not null, "
                      "innerframe string not null, "
                      "method integer not null, "
                      // source entry for imperts strategy
                      "source integer not null default 0, "
                      "createdate integer not null, "
                      // area of inner hull / outer hull
                      "ratio real not null, "
                      // min distance between hulls
                      "clearance real not null default -1.0, "
                      // marked 1 for broken solutions
                      "invalid boolean not null default 0"
                      ")");

  db->ExecuteAndPrint("create table "
                      "if not exists "
                      "attempts ("
                      "id integer primary key, "
                      "polyhedron string not null, "
                      "method integer not null, "
                      // best error seen
                      "best real not null, "
                      // An iteration (generally descending
                      // until reaching a local minimum).
                      "iters integer not null, "
                      // Number of times we tested a pair
                      // of polygons for a possible solution.
                      "evals integer not null, "
                      // source entry for imperts strategy
                      "source integer not null default 0, "
                      "createdate integer not null"
                      ")");

  db->ExecuteAndPrint("create table "
                      "if not exists "
                      "nopertattempts ("
                      "id integer primary key, "
                      "points integer not null, "
                      // Number of random polyhedra solved
                      "attempts integer not null, "
                      "iterhisto string not null, "
                      "createdate integer not null, "
                      "method integer not null"
                      ")");

  db->ExecuteAndPrint("create table "
                      "if not exists "
                      "noperts ("
                      "id integer primary key, "
                      "points integer not null, "
                      // comma-separated doubles in xyz order.
                      "vertices string not null, "
                      "createdate integer not null, "
                      "method integer not null"
                      ")");
}

void SolutionDB::AddNopertAttempt(int points, int64_t attempts,
                                  const AutoHisto &iterhisto,
                                  int method) {
  AutoHisto::Histo h = iterhisto.GetHisto(20);
  std::string histo;
  for (int i = 0; i < h.buckets.size(); i++) {
    if (!histo.empty()) histo.push_back(',');
    double start = h.BucketLeft(i);
    if (start >= 0.0 ||
        h.buckets[i] > 0.0) {
      StringAppendF(&histo, "%.3f=%lld",
                    start, (int64_t)h.buckets[i]);
    }
  }

  db->ExecuteAndPrint(
      StringPrintf(
          "insert into nopertattempts "
          "(points, attempts, iterhisto, createdate, method) "
          "values (%d, %lld, '%s', %lld, %d)",
          points, attempts, histo.c_str(), time(nullptr), method));
}

void SolutionDB::AddNopert(const Polyhedron &poly, int method) {
  std::string vs;
  for (const vec3 &v : poly.vertices) {
    if (!vs.empty()) vs.push_back(',');
    StringAppendF(&vs, "%.17g,%.17g,%.17g", v.x, v.y, v.z);
  }

  db->ExecuteAndPrint(
      StringPrintf(
          "insert into noperts "
          "(points, vertices, createdate, method) "
          "values (%d, '%s', %lld, %d)",
          (int)poly.vertices.size(),
          vs.c_str(), time(nullptr), method));
}

// Expects a specific column order; see below.
static std::vector<Solution> GetSolutionsForQuery(
    std::unique_ptr<Database::Query> q) {
  std::vector<Solution> ret;
  while (std::unique_ptr<Database::Row> r = q->NextRow()) {
    Solution sol;
    sol.id = r->GetInt(0);
    sol.polyhedron = r->GetString(1);
    sol.method = r->GetInt(2);
    auto oo = SolutionDB::StringFrame(r->GetString(3));
    auto io = SolutionDB::StringFrame(r->GetString(4));
    if (!oo.has_value() || !io.has_value()) continue;
    sol.outer_frame = oo.value();
    sol.inner_frame = io.value();
    sol.createdate = r->GetInt(5);
    sol.ratio = r->GetFloat(6);
    sol.clearance = r->GetFloat(7);
    sol.source = r->GetInt(8);
    ret.push_back(std::move(sol));
  }
  return ret;
}

std::vector<Solution> SolutionDB::GetAllSolutions() {
  return GetSolutionsForQuery(
    db->ExecuteString(
        "select "
        "id, polyhedron, method, outerframe, innerframe, "
        "createdate, ratio, clearance, source "
        "from solutions "
        "where invalid = 0"));
}

Solution SolutionDB::GetBestSolutionFor(std::string_view name,
                                                    bool use_clearance) {
  std::vector<Solution> sols = GetSolutionsForQuery(
    db->ExecuteString(
        std::format(
            "select "
            "id, polyhedron, method, outerframe, innerframe, "
            "createdate, ratio, clearance, source "
            "from solutions "
            "where polyhedron = '{}' "
            "and invalid = 0 "
            "{} "
            "limit 1",
            name,
            use_clearance ? "order by clearance desc" : "order by ratio")));
  CHECK(!sols.empty()) << "No solution for " << name;
  return sols[0];
}

std::pair<std::unordered_map<std::string, Solution>,
          std::unordered_map<std::string, Solution>> SolutionDB::BestSolutions() {
  std::unordered_map<std::string, Solution> lowest_ratio;
  std::unordered_map<std::string, Solution> highest_clearance;

  for (const Solution &sol : GetAllSolutions()) {
    if (!lowest_ratio.contains(sol.polyhedron) ||
        lowest_ratio[sol.polyhedron].ratio > sol.ratio) {
      lowest_ratio[sol.polyhedron] = sol;
    }

    if (!highest_clearance.contains(sol.polyhedron) ||
        highest_clearance[sol.polyhedron].clearance < sol.clearance) {
      highest_clearance[sol.polyhedron] = sol;
    }
  }

  return std::make_pair(std::move(lowest_ratio), std::move(highest_clearance));
}

std::vector<Solution> SolutionDB::GetAllNopertSolutions() {
  return GetSolutionsForQuery(
    db->ExecuteString(
        "select "
        "id, polyhedron, method, outerframe, innerframe, "
        "createdate, ratio, clearance, source "
        "from solutions where polyhedron like 'nopert_%' "
        "and invalid = 0"));
}

Solution SolutionDB::GetSolution(int id) {
  std::vector<Solution> sols = GetSolutionsForQuery(
    db->ExecuteString(
        std::format(
            "select "
            "id, polyhedron, method, outerframe, innerframe, "
            "createdate, ratio, clearance, source "
            "from solutions "
            "where id = {}", id)));
  CHECK(sols.size() == 1) << "Solution " << id << " not found";
  return sols[0];
}

std::vector<Solution> SolutionDB::GetSolutionsFor(
    std::string_view name) {
  return GetSolutionsForQuery(
    db->ExecuteString(
        std::format(
            "select "
            "id, polyhedron, method, outerframe, innerframe, "
            "createdate, ratio, clearance, source "
            "from solutions "
            "where polyhedron = '{}' "
            "and invalid = 0",
            name)));
}

std::vector<SolutionDB::Attempt> SolutionDB::GetAllAttempts() {
  std::unique_ptr<Query> q =
    db->ExecuteString(
        "select "
        "id, polyhedron, method, createdate, best, iters, evals, source "
        "from attempts");

  std::vector<Attempt> ret;
  while (std::unique_ptr<Row> r = q->NextRow()) {
    Attempt att;
    att.id = r->GetInt(0);
    att.polyhedron = r->GetString(1);
    att.method = r->GetInt(2);
    att.createdate = r->GetInt(3);
    att.best_error = r->GetFloat(4);
    att.iters = r->GetInt(5);
    att.evals = r->GetInt(6);
    att.source = r->GetInt(7);
    ret.push_back(std::move(att));
  }
  return ret;
}

void SolutionDB::AddAttempt(const std::string &poly, int method, int source,
                            double best,
                            int64_t iters, int64_t evals) {
  db->ExecuteAndPrint(StringPrintf(
      "insert into attempts (polyhedron, createdate, method, source, "
      "best, iters, evals) "
      "values ('%s', %lld, %d, %d, %.17g, %lld, %lld)",
      poly.c_str(), time(nullptr), method, source, best, iters, evals));
}

void SolutionDB::AddSolution(const std::string &polyhedron,
                             const frame3 &outer_frame,
                             const frame3 &inner_frame,
                             int method, int source,
                             double ratio, double clearance) {
  db->ExecuteAndPrint(
      StringPrintf(
          "insert into solutions "
          "(polyhedron, method, source, outerframe, innerframe, "
          "createdate, ratio, clearance, invalid) "
          "values ('%s', %d, %d, '%s', '%s', %lld, %.17g, %.17g, 0)",
          polyhedron.c_str(),
          method,
          source,
          FrameString(outer_frame).c_str(),
          FrameString(inner_frame).c_str(),
          time(nullptr),
          ratio, clearance));
}

static std::vector<SolutionDB::Nopert> GetNopertsForQuery(
    std::unique_ptr<Database::Query> q) {
  std::vector<SolutionDB::Nopert> ret;
  while (std::unique_ptr<Database::Row> r = q->NextRow()) {
    SolutionDB::Nopert nop;
    nop.id = r->GetInt(0);
    std::string vs = r->GetString(1);
    std::vector<std::string> parts = Util::Split(vs, ',');
    CHECK(parts.size() % 3 == 0);
    auto GetDouble = [](const std::string &s) {
        auto od = Util::ParseDoubleOpt(s);
        CHECK(od.has_value()) << "Expected double in nopert: " << s;
        return od.value();
      };
    for (int i = 0; i < parts.size(); i += 3) {
      nop.vertices.emplace_back(GetDouble(parts[i]),
                                GetDouble(parts[i + 1]),
                                GetDouble(parts[i + 2]));
    }
    nop.method = r->GetInt(2);
    nop.createdate = r->GetInt(3);
    ret.push_back(std::move(nop));
  }
  return ret;
}

std::vector<SolutionDB::Nopert> SolutionDB::GetAllNoperts() {
  return GetNopertsForQuery(
    db->ExecuteString(
        "select "
        "id, vertices, method, createdate "
        "from noperts"));
}

SolutionDB::Nopert SolutionDB::GetNopert(int id) {
  std::vector<Nopert> one = GetNopertsForQuery(
    db->ExecuteString(
        std::format(
            "select "
            "id, vertices, method, createdate "
            "from noperts where id = {}",
            id)));
  CHECK(one.size() == 1) << "Couldn't find nopert " << id;
  return std::move(one[0]);
}
