#include "shrinklutions.h"

#include <cstdint>
#include <ctime>
#include <format>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "base/logging.h"
#include "base/stringprintf.h"
#include "database.h"
#include "util.h"

using frame3 = ShrinklutionDB::frame3;

std::string ShrinklutionDB::FrameString(const frame3 &frame) {
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


std::optional<frame3> ShrinklutionDB::StringFrame(std::string_view s) {
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

void ShrinklutionDB::Init() {
  db->ExecuteAndPrint("create table "
                      "if not exists "
                      "shrinksol ("
                      "id integer primary key, "
                      "num integer not null, "
                      "cubes string not null, "
                      "method integer not null, "
                      "source integer not null, "
                      "createdate integer not null, "
                      "radius real not null, "
                      "invalid boolean not null "
                      ")");
}

std::optional<std::vector<frame3>>
ShrinklutionDB::StringCubes(std::string_view s) {
  std::vector<frame3> ret;
  std::vector<std::string> scubes = Util::Split(s, '|');
  for (const std::string &scube : scubes) {
    auto co = ShrinklutionDB::StringFrame(scube);
    if (!co.has_value()) return std::nullopt;
    ret.push_back(co.value());
  }
  return {ret};
}

std::string ShrinklutionDB::CubesString(const std::vector<frame3> &cubes) {
  std::vector<std::string> scubes;
  for (const frame3 &cube : cubes) {
    scubes.push_back(FrameString(cube));
  }
  return Util::Join(scubes, "|");
}

// Expects a specific column order; see below.
static std::vector<ShrinklutionDB::Solution> GetSolutionsForQuery(
    std::unique_ptr<Database::Query> q) {
  std::vector<ShrinklutionDB::Solution> ret;
  while (std::unique_ptr<Database::Row> r = q->NextRow()) {
    ShrinklutionDB::Solution sol;
    sol.id = r->GetInt(0);
    sol.num = r->GetInt(1);
    std::optional<std::vector<frame3>> ocubes =
      ShrinklutionDB::StringCubes(r->GetString(2));
    if (!ocubes.has_value()) continue;
    sol.cubes = std::move(ocubes.value());
    sol.method = r->GetInt(3);
    sol.source = r->GetInt(4);
    sol.createdate = r->GetInt(5);
    sol.radius = r->GetFloat(6);
    ret.push_back(std::move(sol));
  }
  return ret;
}

std::vector<ShrinklutionDB::Solution> ShrinklutionDB::GetAllSolutions() {
  return GetSolutionsForQuery(
    db->ExecuteString(
        "select "
        "id, num, cubes, method, source, createdate, radius "
        "from shrinksol "
        "where invalid = 0"));
}

ShrinklutionDB::Solution ShrinklutionDB::GetBestSolutionFor(int num) {
  std::vector<Solution> sols = GetSolutionsForQuery(
    db->ExecuteString(
        std::format(
            "select "
            "id, num, cubes, method, source, createdate, radius "
            "from shrinksol "
            "where num = '{}' "
            "and invalid = 0 "
            "order by radius "
            "limit 1",
            num)));

  CHECK(!sols.empty()) << "No solution for " << num;
  return sols[0];
}

bool ShrinklutionDB::HasSolutionWithRadius(int num, double radius) {
  static constexpr double tol = 1.0e-10;
  std::vector<Solution> sols = GetSolutionsForQuery(
    db->ExecuteString(
        std::format(
            "select "
            "id, num, cubes, method, source, createdate, radius "
            "from shrinksol "
            "where num = '{}' "
            "and radius > {:17g} "
            "and radius < {:17g} "
            "and invalid = 0 "
            "limit 1",
            num, radius - tol, radius + tol)));
  return !sols.empty();
}

ShrinklutionDB::Solution ShrinklutionDB::GetSolution(int id) {
  std::vector<Solution> sols = GetSolutionsForQuery(
    db->ExecuteString(
        std::format(
            "select "
            "id, num, cubes, method, source, createdate, radius "
            "from shrinksol "
            "where id = {}", id)));
  CHECK(sols.size() == 1) << "Solution " << id << " not found";
  return sols[0];
}

std::vector<ShrinklutionDB::Solution>
ShrinklutionDB::GetSolutionsFor(int num) {
  return GetSolutionsForQuery(
    db->ExecuteString(
        std::format(
            "select "
            "id, num, cubes, method, source, createdate, radius "
            "from shrinksol "
            "where num = {} "
            "and invalid = 0", num)));
}

void ShrinklutionDB::AddSolution(int num,
                                 const std::vector<frame3> &cubes,
                                 int method, int source,
                                 double radius) {
  db->ExecuteAndPrint(
      StringPrintf(
          "insert into shrinksol "
          "(num, cubes, method, source, createdate, radius, invalid) "
          "values (%d, '%s', %d, %d, %lld, %.17g, 0)",
          num, CubesString(cubes).c_str(),
          method,
          source,
          time(nullptr),
          radius));
}

