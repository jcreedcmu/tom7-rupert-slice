#include "solutions.h"

#include <cstdint>
#include <ctime>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "auto-histo.h"
#include "base/logging.h"
#include "base/stringprintf.h"
#include "database.h"
#include "polyhedra.h"
#include "util.h"

using frame3 = SolutionDB::frame3;

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
                      "createdate integer not null, "
                      // area of inner hull / outer hull
                      "ratio real not null"
                      ")");

  db->ExecuteAndPrint("create table "
                      "if not exists "
                      "best ("
                      "id integer primary key, "
                      "polyhedron string not null, "
                      "outerframe string not null, "
                      "innerframe string not null, "
                      "createdate integer not null, "
                      "ratio real not null"
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

static std::vector<SolutionDB::Solution> GetSolutionsForQuery(
    std::unique_ptr<Database::Query> q) {
  std::vector<SolutionDB::Solution> ret;
  while (std::unique_ptr<Database::Row> r = q->NextRow()) {
    SolutionDB::Solution sol;
    sol.polyhedron = r->GetString(0);
    sol.method = r->GetInt(1);
    auto oo = SolutionDB::StringFrame(r->GetString(2));
    auto io = SolutionDB::StringFrame(r->GetString(3));
    if (!oo.has_value() || !io.has_value()) continue;
    sol.outer_frame = oo.value();
    sol.inner_frame = io.value();
    sol.createdate = r->GetInt(4);
    sol.ratio = r->GetFloat(5);
    ret.push_back(std::move(sol));
  }
  return ret;
}

std::vector<SolutionDB::Solution> SolutionDB::GetAllSolutions() {
  return GetSolutionsForQuery(
    db->ExecuteString(
        "select "
        "polyhedron, method, outerframe, innerframe, "
        "createdate, ratio "
        "from solutions"));

}

std::vector<SolutionDB::Solution> SolutionDB::GetSolutionsFor(
    const std::string &name) {
  return GetSolutionsForQuery(
    db->ExecuteString(
        StringPrintf(
            "select "
            "polyhedron, method, outerframe, innerframe, "
            "createdate, ratio "
            "from solutions "
            "where polyhedron = '%s'",
            name.c_str())));
}

std::vector<SolutionDB::Attempt> SolutionDB::GetAllAttempts() {
  std::unique_ptr<Query> q =
    db->ExecuteString(
        "select "
        "polyhedron, method, createdate, best, iters, evals "
        "from attempts");

  std::vector<Attempt> ret;
  while (std::unique_ptr<Row> r = q->NextRow()) {
    Attempt att;
    att.polyhedron = r->GetString(0);
    att.method = r->GetInt(1);
    att.createdate = r->GetInt(2);
    att.best_error = r->GetFloat(3);
    att.iters = r->GetInt(4);
    att.evals = r->GetInt(5);
    ret.push_back(std::move(att));
  }
  return ret;
}

void SolutionDB::AddAttempt(const std::string &poly, int method,
                            double best,
                            int64_t iters, int64_t evals) {
  db->ExecuteAndPrint(StringPrintf(
      "insert into attempts (polyhedron, createdate, method, best, "
      "iters, evals) "
      "values ('%s', %lld, %d, %.17g, %lld, %lld)",
      poly.c_str(), time(nullptr), method, best, iters, evals));
}

void SolutionDB::AddSolution(const std::string &polyhedron,
                             const frame3 &outer_frame,
                             const frame3 &inner_frame,
                             int method,
                             double ratio) {
  db->ExecuteAndPrint(
      StringPrintf(
          "insert into solutions "
          "(polyhedron, method, outerframe, innerframe, createdate, ratio) "
          "values ('%s', %d, '%s', '%s', %lld, %.17g)",
          polyhedron.c_str(),
          method,
          FrameString(outer_frame).c_str(),
          FrameString(inner_frame).c_str(),
          time(nullptr),
          ratio));
}
