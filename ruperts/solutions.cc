#include "solutions.h"

#include <cstdint>
#include <ctime>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "database.h"
#include "base/logging.h"
#include "base/stringprintf.h"
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
                      // An attempt (generally descending
                      // until reaching a local minimum).
                      "count integer not null, "
                      "createdate integer not null"
                      ")");
}

std::vector<SolutionDB::Solution> SolutionDB::GetAllSolutions() {
  std::unique_ptr<Query> q =
    db->ExecuteString(
        "select "
        "polyhedron, method, outerframe, innerframe, "
        "createdate, ratio "
        "from solutions");

  std::vector<Solution> ret;
  while (std::unique_ptr<Row> r = q->NextRow()) {
    Solution sol;
    sol.polyhedron = r->GetString(0);
    sol.method = r->GetInt(1);
    auto oo = StringFrame(r->GetString(2));
    auto io = StringFrame(r->GetString(3));
    if (!oo.has_value() || !io.has_value()) continue;
    sol.outer_frame = oo.value();
    sol.inner_frame = io.value();
    sol.createdate = r->GetInt(4);
    sol.ratio = r->GetFloat(5);
    ret.push_back(std::move(sol));
  }
  return ret;
}

void SolutionDB::AddAttempt(const std::string &poly, int method,
                            int64_t count) {
  db->ExecuteAndPrint(StringPrintf(
      "insert into attempts (polyhedron, createdate, method, count) "
      "values ('%s', %lld, %d, %lld)",
      poly.c_str(), time(nullptr), method, count));
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
