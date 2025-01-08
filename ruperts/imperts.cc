#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <functional>
#include <limits>
#include <mutex>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "ansi.h"
#include "arcfour.h"
#include "atomic-util.h"
#include "auto-histo.h"
#include "base/stringprintf.h"
#include "opt/opt.h"
#include "periodically.h"
#include "polyhedra.h"
#include "randutil.h"
#include "rendering.h"
#include "solutions.h"
#include "status-bar.h"
#include "threadutil.h"
#include "timer.h"
#include "yocto_matht.h"

// Try to find counterexamples.

#define ABLOOD(s) AFGCOLOR(148, 0, 0, s)

DECLARE_COUNTERS(polyhedra, total_evals, u1_, u2_, u3_, u4_, u5_, u6_);

static StatusBar *status = nullptr;

using vec2 = yocto::vec<double, 2>;
using vec3 = yocto::vec<double, 3>;
using vec4 = yocto::vec<double, 4>;
using mat4 = yocto::mat<double, 4>;
using quat4 = yocto::quat<double, 4>;
using frame3 = yocto::frame<double, 3>;

using Solution = SolutionDB::Solution;
using Attempt = SolutionDB::Attempt;

static double InnerDistanceLoss(
    const Polyhedron &poly,
    const frame3 &outer_frame, const frame3 &inner_frame) {
  Mesh2D souter = Shadow(Rotate(poly, outer_frame));
  Mesh2D sinner = Shadow(Rotate(poly, inner_frame));

  // Although computing the convex hull is expensive, the tests
  // below are O(n*m), so it is helpful to significantly reduce
  // one of the factors.
  const std::vector<int> outer_hull = GrahamScan(souter.vertices);
  HullInscribedCircle circle(souter.vertices, outer_hull);

  // Does every vertex in inner fall inside the outer shadow?
  double error = 0.0;
  int errors = 0;
  for (const vec2 &iv : sinner.vertices) {
    if (circle.DefinitelyInside(iv) || InHull(souter, outer_hull, iv)) {
      // Further from the hull is better, when on the inside.
      error -= DistanceToHull(souter.vertices, outer_hull, iv);
    } else {
      // Include some gradient here as well, but the score must be
      // postive enough that it can never be overcome by summing
      // with some negative distances. The negative distances for
      // these polyhedra are bounded by 2 or so.
      error += 10000.0 + DistanceToHull(souter.vertices, outer_hull, iv);
      errors++;
    }
  }

  if (error <= 0.0 && errors > 0) [[unlikely]] {
    // If they are not in the mesh, don't return an actual zero.
    return std::numeric_limits<double>::min() * errors;
  } else {
    return error;
  }
}

static void SaveImprovement(SolutionDB *db,
                            const Polyhedron &poly,
                            const Solution &old,
                            const frame3 &outer_frame,
                            const frame3 &inner_frame) {

  std::optional<double> new_ratio = GetRatio(poly, outer_frame, inner_frame);

  if (!new_ratio.has_value()) {
    status->Printf(ARED("SOLUTION IS INVALID!?") "\n");
    return;
  }

  const double ratio = new_ratio.value();

  db->AddSolution(poly.name, outer_frame, inner_frame,
                  SolutionDB::METHOD_IMPROVE, old.id, ratio);

  Rendering rendering(poly, 3840, 2160);
  auto Render = [&rendering, &poly](const frame3 &outer_frame,
                                    const frame3 &inner_frame,
                                    uint32_t outer_color,
                                    uint32_t inner_color,
                                    bool bad) {
      Polyhedron outer = Rotate(poly, outer_frame);
      Polyhedron inner = Rotate(poly, inner_frame);
      Mesh2D souter = Shadow(outer);
      Mesh2D sinner = Shadow(inner);

      std::vector<int> outer_hull = QuickHull(souter.vertices);
      std::vector<int> inner_hull = QuickHull(sinner.vertices);

      rendering.RenderHull(souter, outer_hull, outer_color);
      rendering.RenderHull(sinner, inner_hull, inner_color);

      if (bad) {
        rendering.RenderBadPoints(sinner, souter);
      }
    };

  Render(old.outer_frame, old.inner_frame, 0x440000FF, 0x005500BB, false);
  Render(outer_frame, inner_frame, 0xAA0000FF, 0x00FF00AA, true);

  rendering.Save(StringPrintf("impert-%s-%lld.png", poly.name, time(nullptr)));

  std::string old_ratio_str = ARED("(invalid)");;
  {
    std::optional<double> old_ratio =
      GetRatio(poly, old.outer_frame, old.inner_frame);

    if (old_ratio.has_value()) {
      old_ratio_str = StringPrintf(APURPLE("%.7g"), old_ratio.value());
    }
  }

  printf("Added solution (" AYELLOW("%s") ") to db; "
         "ratio " "%s" AGREEN(" → ") ACYAN("%.7g") "\n",
         poly.name, old_ratio_str.c_str(), ratio);
}


#define ANSI_ORANGE ANSI_FG(247, 155, 57)

static constexpr int MIN_FLAT_ITERS = 100;
static constexpr int MAX_ITERS = 3000;
template<class UpdateStatus>
static std::optional<std::pair<frame3, frame3>> TryImprove(
    int thread_idx,
    ArcFour *rc,
    const Polyhedron &poly,
    int source,
    const frame3 &original_outer_frame,
    const frame3 &original_inner_frame,
    int64_t *iters_out, double *best_error_out, int64_t *evals_out,
    const UpdateStatus &update_status) {
  CHECK(!poly.faces->v.empty());

  Timer improve_timer;

  std::string short_name = PolyhedronShortName(poly.name);

  const auto &[original_outer_rot, otrans] =
    UnpackFrame(original_outer_frame);
  const auto &[original_inner_rot, itrans] =
    UnpackFrame(original_inner_frame);

  const double start_loss = InnerDistanceLoss(poly,
                                              original_outer_frame,
                                              original_inner_frame);
  status->Printf("[" AYELLOW("%d") "] "
                 "Starting loss: %s%.17g" ANSI_RESET "\n",
                 thread_idx,
                 start_loss < 0.999999 ? ANSI_GREEN :
                 start_loss < 0.0 ? ANSI_YELLOW :
                 start_loss == 0.0 ? ANSI_ORANGE : ANSI_RED,
                 start_loss);

  if (start_loss > 0.0) {
    status->Printf("[" AYELLOW("%d") "] "
                   ABLOOD("✘") " " ARED("Bogus") ".\n",
                   thread_idx);
    return std::nullopt;
  }

  double best_loss = start_loss;
  std::optional<std::pair<frame3, frame3>> best_result;
  int last_improved_iter = 0;

  AutoHisto histo(MAX_ITERS);

  // The outer translation should be 0, but we can just
  // make that be the case. The z component should also
  // be zero, but can be ignored.
  const vec2 original_trans = [&]() {
      vec3 o = itrans - otrans;
      return vec2(o.x, o.y);
    }();

  int64_t evals_here = 0;
  for (int iter = 0; iter < MAX_ITERS; iter++) {
    {
      ANSI::ProgressBarOptions options;
      options.include_frac = false;
      options.include_percent = false;

      std::string arrow;
      if (best_result.has_value()) {
        options.bar_filled = 0x177715FF;
        options.bar_empty  = 0x001a03FF;

        arrow =
          StringPrintf(AORANGE("%.7g") AGREEN(" → ") ACYAN("%.7g"),
                       start_loss, best_loss);
      } else {
        arrow =
          StringPrintf(ARED("%.3g") " " AYELLOW("%s"),
                       start_loss,
                       histo.UnlabeledHoriz(20).c_str());
      }

      int target_iters =
        best_result.has_value() ? last_improved_iter + MIN_FLAT_ITERS :
        MAX_ITERS;

      std::string bar = ANSI::ProgressBar(
          iter, target_iters,
          StringPrintf("%d " AWHITE("%s") "[%d] %s",
                       iter,
                       short_name.c_str(), source, arrow.c_str()),
          improve_timer.Seconds(),
          options);
      update_status(bar);
    }

    // four params for outer rotation, four params for inner
    // rotation, two for 2d translation of inner.
    static constexpr int D = 10;

    // Get the frames from the appropriate positions in the
    // argument.

    auto OuterFrame = [&original_outer_rot](
        const std::array<double, D> &args) {
        const auto &[o0, o1, o2, o3,
                     i0_, i1_, i2_, i3_, dx_, dy_] = args;
        quat4 tweaked_rot = normalize(quat4{
            .x = original_outer_rot.x + o0,
            .y = original_outer_rot.y + o1,
            .z = original_outer_rot.z + o2,
            .w = original_outer_rot.w + o3,
          });
        return yocto::rotation_frame(tweaked_rot);
      };

    auto InnerFrame = [&original_inner_rot, &original_trans](
        const std::array<double, D> &args) {
        const auto &[o0_, o1_, o2_, o3_,
                     i0, i1, i2, i3, dx, dy] = args;
        quat4 tweaked_rot = normalize(quat4{
            .x = original_inner_rot.x + i0,
            .y = original_inner_rot.y + i1,
            .z = original_inner_rot.z + i2,
            .w = original_inner_rot.w + i3,
          });
        frame3 rotate = yocto::rotation_frame(tweaked_rot);
        frame3 translate = yocto::translation_frame(
            vec3{
              .x = original_trans.x + dx,
              .y = original_trans.y + dy,
              .z = 0.0,
            });
        return rotate * translate;
      };

    std::function<double(const std::array<double, D> &)> Loss =
      [&poly, &evals_here, &OuterFrame, &InnerFrame](
          const std::array<double, D> &args) {
        total_evals++;
        evals_here++;
        return InnerDistanceLoss(poly, OuterFrame(args), InnerFrame(args));
      };

    constexpr double Q = 1;
    constexpr double T = 0.5;
    // constexpr double Q = 0.15;
    // constexpr double T = 0.25;
    // constexpr double Q = 0.000001;
    // constexpr double T = 0.000001;

    const std::array<double, D> lb =
      {-Q, -Q, -Q, -Q,
       -Q, -Q, -Q, -Q, -T, -T};
    const std::array<double, D> ub =
      {+Q, +Q, +Q, +Q,
       +Q, +Q, +Q, +Q, +T, +T};

    const int seed = RandTo(rc, 0x7FFFFFFE);
    const auto &[args, error] =
      Opt::Minimize<D>(Loss, lb, ub, 2000, 2, 100, seed);

    histo.Observe(error);

    if (error < 0.0 && error < best_loss) {
      status->Printf("[" AYELLOW("%d") "] Iter %d: " AGREEN("Success!") " "
                     ABLUE("%.17g") " → " ACYAN("%.17g") "\n",
                     thread_idx, iter,
                     best_loss, error);

      frame3 outer_frame = OuterFrame(args);
      frame3 inner_frame = InnerFrame(args);

      best_loss = error;
      best_result = {std::make_pair(outer_frame, inner_frame)};
      last_improved_iter = iter;
    } else {
      *best_error_out = std::min(*best_error_out, error);
    }

    if (best_result.has_value() &&
        iter - last_improved_iter >= MIN_FLAT_ITERS) {
      *iters_out = iter;
      *evals_out = evals_here;
      return best_result;
    }
  }

  *iters_out = MAX_ITERS;
  *evals_out = evals_here;
  return best_result;
}

struct Imperts {

  static constexpr int NUM_THREADS = 8;

  std::mutex m;
  ArcFour rc = ArcFour(StringPrintf("work.%lld", time(nullptr)));
  SolutionDB db;
  // map solution id to the count of attempts to improve
  std::unordered_map<int, int> num_attempted;
  std::unordered_set<int> already_improved;
  std::unordered_set<int> in_progress;
  Periodically refresh_solutions_per = Periodically(60.0);
  std::vector<std::string> threadstatus;

  bool should_die = false;
  Timer timer;
  AutoHisto histo = AutoHisto(10000);
  Periodically status_per = Periodically(5.0);
  double total_gen_sec = 0.0;
  double total_solve_sec = 0.0;
  bool success = false;
  int64_t improved = 0, notimproved = 0;

  Imperts() {
    polyhedra.Reset();
    total_evals.Reset();
    threadstatus.resize(NUM_THREADS, "init");
  }

  std::vector<Solution> work_queue;
  std::optional<Solution> GetWork() {
    MutexLock ml(&m);
    if (refresh_solutions_per.ShouldRun() || work_queue.empty()) {
      std::unordered_map<std::string, int> num_solutions;
      std::unordered_map<std::string, double> best_ratio;
      status->Printf("Refresh solution db.");
      std::vector<Attempt> attempts = db.GetAllAttempts();
      for (const Attempt &att : attempts) {
        if (att.method == SolutionDB::METHOD_IMPROVE) {
          CHECK(att.source > 0) << "Bad database: " << att.id;
          num_attempted[att.source]++;
        }
      }

      work_queue.clear();
      std::vector<Solution> all_solutions = db.GetAllSolutions();
      for (Solution &sol : all_solutions) {
        num_solutions[sol.polyhedron]++;
        auto it = best_ratio.find(sol.polyhedron);
        if (it == best_ratio.end()) {
          best_ratio[sol.polyhedron] = sol.ratio;
        } else {
          it->second = std::min(it->second, sol.ratio);
        }
        if (sol.method == SolutionDB::METHOD_IMPROVE) {
          already_improved.insert(sol.source);
        }
      }

      constexpr bool SEQUENTIAL_IMPROVEMENT = false;

      #define APINK(s) AFGCOLOR(230, 168, 225, s)
      for (const auto &[name, best] : best_ratio) {
        status->Printf(AWHITE("%s") " (%s) has " AGREEN("%d")
                       " sols, best ratio "
                       APINK("%.17g") "\n",
                       PolyhedronShortName(name).c_str(),
                       name.c_str(),
                       num_solutions[name],
                       best);
      }

      for (Solution &sol : all_solutions) {
        if (SEQUENTIAL_IMPROVEMENT || sol.method !=
            SolutionDB::METHOD_IMPROVE) {
          if (!already_improved.contains(sol.id)) {
            if (num_solutions[sol.polyhedron] < 4 ||
                best_ratio[sol.polyhedron] > 0.999 ||
                (num_attempted[sol.id] == 0 &&
                 !in_progress.contains(sol.id))) {
              work_queue.emplace_back(std::move(sol));
            }
          }
        }
      }

      Shuffle(&rc, &work_queue);
    }

    // Find work that has not yet been attempted.
    if (work_queue.empty()) {
      return std::nullopt;
    } else {
      Solution sol = work_queue.back();
      work_queue.pop_back();
      in_progress.insert(sol.id);
      return {sol};
    }
  }

  void MaybeStatus() {
    status_per.RunIf([this]() {
        MutexLock ml(&m);
        double total_time = timer.Seconds();
        const int64_t evals = total_evals.Read();
        const int64_t polys = polyhedra.Read();
        double eps = evals / total_time;

        std::string timing =
          StringPrintf(
              "%s "
              "[" AWHITE("%.1f") "/s] "
              " " AGREY("=") "  "
              "%s " APURPLE("gen") " + "
              "%s " ABLUE("sol"),
              ANSI::Time(total_time).c_str(),
              eps,
              ANSI::Time(total_gen_sec).c_str(),
              ANSI::Time(total_solve_sec).c_str());

        std::string msg =
          StringPrintf(
              ACYAN("%s") AWHITE("∫") " "
              ABLUE("%s") AWHITE("∎") " "
              ARED("%s") ABLOOD("=") " "
              AGREEN("%s") AWHITE("↓") " "
              "Queue: %d",
              FormatNum(evals).c_str(),
              FormatNum(polys).c_str(),
              FormatNum(notimproved).c_str(),
              FormatNum(improved).c_str(),
              (int)work_queue.size());


        std::vector<std::string> status_lines;
        status_lines.reserve(4 + NUM_THREADS);

        status_lines.push_back(
            "———————————————————————————————————————————"
            "————————————————————————————————");
        for (const std::string &line : threadstatus)
          status_lines.push_back(line);
        status_lines.push_back(
            "———————————————————————————————————————————"
            "————————————————————————————————");
        status_lines.push_back(timing);
        status_lines.push_back(msg);

        status->EmitStatus(status_lines);
      });
  }

  void Run() {
    ParallelFan(
      NUM_THREADS,
      [&](int thread_idx) {
        ArcFour rc(StringPrintf("noperts.%d.%lld\n",
                                thread_idx, time(nullptr)));

        for (;;) {
          {
            MutexLock ml(&m);
            if (should_die) return;
          }

          Timer gen_timer;
          std::optional<Solution> osol = GetWork();
          if (!osol.has_value()) {
            status->Printf("No more work.\n");
            MutexLock ml(&m);
            should_die = true;
            return;
          }
          const Solution &solution = osol.value();

          // PERF: Use singletons for this.
          Polyhedron poly = PolyhedronByName(solution.polyhedron);
          polyhedra++;
          const double gen_sec = gen_timer.Seconds();

          status->Printf("[" AYELLOW("%d") "] #%d. " AWHITE("%s") " "
                         "(via %s)\n",
                         thread_idx, solution.id, poly.name,
                         SolutionDB::MethodName(solution.method));

          Timer solve_timer;
          int64_t iters = 0, evals = 0;
          double best_error = 0.0;
          std::optional<std::pair<frame3, frame3>> oimproved =
            TryImprove(thread_idx, &rc, poly,
                       solution.id,
                       solution.outer_frame,
                       solution.inner_frame,
                       &iters, &best_error, &evals,
                       [this, thread_idx](const std::string &s) {
                         {
                           MutexLock ml(&m);
                           threadstatus[thread_idx] = s;
                         }
                         this->MaybeStatus();
                       });
          const double solve_sec = solve_timer.Seconds();

          {
            MutexLock ml(&m);

            total_gen_sec += gen_sec;
            total_solve_sec += solve_sec;

            if (oimproved.has_value()) {
              already_improved.insert(solution.id);
              const auto &[oframe, iframe] = oimproved.value();
              improved++;
              SaveImprovement(&db, poly, solution, oframe, iframe);
            } else {
              num_attempted[solution.id]++;
              notimproved++;
              if (iters > 0) {
                db.AddAttempt(poly.name, SolutionDB::METHOD_IMPROVE,
                              solution.id, best_error, iters, evals);
              }
            }
          }
          delete poly.faces;
        }
      });
  }
};

int main(int argc, char **argv) {
  ANSI::Init();
  printf("\n");

  status = new StatusBar(Imperts::NUM_THREADS + 4);

  Imperts imperts;
  imperts.Run();

  printf("OK\n");
  return 0;
}

