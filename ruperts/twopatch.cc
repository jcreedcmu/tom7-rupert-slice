
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <format>
#include <mutex>
#include <numbers>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

#include "ansi.h"
#include "arcfour.h"
#include "atomic-util.h"
#include "base/stringprintf.h"
#include "big-polyhedra.h"
#include "hashing.h"
#include "map-util.h"
#include "nd-solutions.h"
#include "opt/opt.h"
#include "patches.h"
#include "periodically.h"
#include "polyhedra.h"
#include "status-bar.h"
#include "threadutil.h"
#include "timer.h"
#include "util.h"
#include "yocto_matht.h"

DECLARE_COUNTERS(sols_done, pairs_done);

// Look for solutions that involve the two specific patches.
// Plot them on both outer and inner patches.
using namespace yocto;

static int NUM_THREADS = 16;

static bool CLOUD = false;

static int LIMIT = 99999999;

static constexpr bool SELF_CHECK = false;

// See howmanyrep.cc to plot how this affects the error.
// With the corrected loss function, over 1 million samples:
// After 23 reps we're within 1/2048 of the error found
// (in 200 rounds). So with a margin of safety we should
// be pretty confident in the optimization result.
static constexpr int ATTEMPTS = 32;

static constexpr int DIGITS = 24;

static constexpr int TARGET_SAMPLES = 1'000'000;

// XXX Specific to snub cube.
[[maybe_unused]]
static constexpr int TOTAL_PATCHES = 36;
static constexpr int TOTAL_PAIRS = TOTAL_PATCHES * TOTAL_PATCHES;

static constexpr std::string_view PATCH_INFO_FILE =
  "scube-patchinfo.txt";
static const char *PATCH_STATUS_FILE =
  "scube-patchstatus.txt";

static_assert(__cplusplus >= 202002L, "This code requires C++20.");
static std::atomic_flag terminated;
extern "C" void HandleSigTerm(int sig) {
  if (sig == SIGTERM) {
    terminated.test_and_set();
  }
}


std::string PolyString(const std::vector<vec2> &poly) {
  std::string ret;
  for (const vec2 &v : poly) {
    AppendFormat(&ret, "  ({:.17g}, {:.17g})\n", v.x, v.y);
  }
  return ret;
}

// Avoid redoing work by storing the completed set here.
struct PatchStatus {
  std::unordered_set<std::pair<int, int>,
                     Hashing<std::pair<int, int>>> reserved;
  std::unordered_set<std::pair<int, int>,
                     Hashing<std::pair<int, int>>> done;

  PatchStatus(std::string_view filename) {
    for (const std::string &original_line :
           Util::NormalizeLines(Util::ReadFileToLines(filename))) {
      std::string_view line = original_line;
      std::string_view cmd = Util::Chop(&line);
      std::string_view so = Util::Chop(&line);
      std::string_view si = Util::Chop(&line);
      CHECK(!so.empty() && !si.empty()) << "Bad line " << original_line;
      int o = atoi(std::string(so).c_str());
      int i = atoi(std::string(si).c_str());
      if (cmd == "reserved") {
        reserved.insert(std::make_pair(o, i));
      } else if (cmd == "done") {
        done.insert(std::make_pair(o, i));
      } else {
        CHECK(false) << "Bad command: " << original_line;
      }
    }
  }
};

struct TwoPatch {

  TwoPatch(std::string_view what,
           StatusBar *status,
           const BigPoly &big_poly,
           uint64_t outer_code, uint64_t inner_code,
           uint64_t outer_mask = 0, uint64_t inner_mask = 0) :
    what(what),
    status(status),
    boundaries(big_poly),
    small_poly(SmallPoly(big_poly)),
    outer_code(outer_code), inner_code(inner_code),
    outer_mask(outer_mask), inner_mask(inner_mask),
    sols(TwoPatchFilename(outer_code, inner_code)),
    diameter(Diameter(small_poly)) {

    if (sols.Size() >= TARGET_SAMPLES) {
      pairs_done++;
      status->Print(ACYAN("{}") ": Already have {} samples.",
                    TwoPatchFilename(outer_code, inner_code),
                    sols.Size());
      return;
    }

    if (outer_mask == 0) {
      outer_mask = GetCodeMask(boundaries, outer_code);
    }
    if (inner_mask == 0) {
      inner_mask = GetCodeMask(boundaries, inner_code);
    }
    CHECK(outer_mask != 0 && inner_mask != 0);

    // Just need to compute the hulls once.
    outer_hull = ComputeHullForPatch(boundaries, outer_code, outer_mask, {});
    inner_hull = ComputeHullForPatch(boundaries, inner_code, inner_mask, {});

    start_sols_size = sols.Size();
    {
      MutexLock ml(&mu);
      MaybeStatus();
    }
  }

  const std::string_view what;
  StatusBar *status = nullptr;
  Periodically status_per = Periodically(1);
  Periodically save_per = Periodically(60 * 5);

  std::mutex mu;
  bool should_die = false;
  const Boundaries boundaries;
  const Polyhedron small_poly;
  const uint64_t outer_code = 0, inner_code = 0;
  uint64_t outer_mask = 0, inner_mask = 0;
  double total_sample_sec = 0.0, total_opt_sec = 0.0, total_add_sec = 0.0;
  NDSolutions<6> sols;
  int64_t start_sols_size = 0;
  const double diameter = 0.0;
  Timer run_timer;

  std::vector<int> outer_hull, inner_hull;


  void WorkThread(int thread_idx) {
    ArcFour rc(std::format("{}.{}", time(nullptr), thread_idx));

    for (;;) {
      {
        MutexLock ml(&mu);
        if (sols.Size() >= TARGET_SAMPLES) {
          should_die = true;
        }

        if (terminated.test())
          should_die = true;

        if (should_die)
          return;
      }

      Timer sample_timer;
      // Uniformly random view positions in each patch.
      // PERF: This can be a lot smarter, but sample generation is
      // a tiny fraction of the overall time.
      const vec3 outer_view =
        GetVec3InPatch(&rc, boundaries, outer_code, outer_mask);
      const vec3 inner_view =
        GetVec3InPatch(&rc, boundaries, inner_code, inner_mask);
      double sample_sec = sample_timer.Seconds();

      const frame3 outer_frame = FrameFromViewPos(outer_view);
      const frame3 inner_frame = FrameFromViewPos(inner_view);

      // 2D convex polygons (the projected hulls in these view
      // positions).
      const std::vector<vec2> outer_poly =
        PlaceHull(outer_frame, outer_hull);
      const std::vector<vec2> inner_poly =
        PlaceHull(inner_frame, inner_hull);

      // We can precompute inscribed circle etc., although we
      // generally expect these hulls to be pretty close (we
      // have already removed the interior points). Probably
      // better would be to use the inscribed/circumscribed
      // circles to set bounds on the translation.

      // PERF unnecessary!
      if (SELF_CHECK) {
        CHECK(SignedAreaOfConvexPoly(outer_poly) > 0.0);
        CHECK(IsConvexAndScreenClockwise(outer_poly)) << PolyString(outer_poly);

        CHECK(SignedAreaOfConvexPoly(inner_poly) > 0.0);
        CHECK(IsConvexAndScreenClockwise(inner_poly)) << PolyString(inner_poly);
      }

      PolyTester2D outer_tester(outer_poly);
      if (SELF_CHECK) {
        CHECK(outer_tester.IsInside(vec2{0, 0}));
      }

      // we rotate the inner polygon around the origin by theta, and
      // translate it by dx,dy.
      //
      // The loss function finds the maximum distance among the points
      // that are outside the outer poly. Overall, we're trying to
      // minimize that difference.
      auto Loss = [&outer_tester, &inner_poly](
          const std::array<double, 3> &args) {
          const auto &[theta, dx, dy] = args;
          frame2 iframe = rotation_frame2(theta);
          iframe.o = {dx, dy};

          int outside = 0;
          double max_sqdistance = 0.0;
          for (const vec2 &v_in : inner_poly) {
            vec2 pt = transform_point(iframe, v_in);

            // Is the transformed point in the hull? If not,
            // compute its distance.

            std::optional<double> osqdist =
              outer_tester.SquaredDistanceOutside(pt);

            if (osqdist.has_value()) {
              outside++;
              max_sqdistance = std::max(max_sqdistance, osqdist.value());
            }
          }

          double min_dist = sqrt(max_sqdistance);

          if (outside > 0 && min_dist == 0.0) [[unlikely]] {
            return outside / 1.0e16;
          } else {
            return min_dist;
          }
        };

      // PERF: With better bounds on the OD and ID of the
      // two hulls, we could limit the translation a lot more.
      const double TMAX = diameter;

      static constexpr int D = 3;
      const std::array<double, D> lb =
        {0.0, -TMAX, -TMAX};
      const std::array<double, D> ub =
        {2.0 * std::numbers::pi, +TMAX, +TMAX};

      Timer opt_timer;
      const auto &[args, error] =
        Opt::Minimize<D>(Loss, lb, ub, 1000, 2, ATTEMPTS);
      [[maybe_unused]] const double opt_sec = opt_timer.Seconds();
      [[maybe_unused]] double aps = ATTEMPTS / opt_sec;

      Timer add_timer;
      // The "solution" is the same outer frame, but we need to
      // include the 2D rotation and translation in the inner frame.
      const auto &[theta, dx, dy] = args;
      const frame3 inner_sol_frame =
        translation_frame(vec3{dx, dy, 0}) *
        // around the z axis
        rotation_frame({0, 0, 1}, theta) *
        inner_frame;

      // solutions itself is synchronized.
      std::array<double, 6> key;
      key[0] = outer_view.x;
      key[1] = outer_view.y;
      key[2] = outer_view.z;
      key[3] = inner_view.x;
      key[4] = inner_view.y;
      key[5] = inner_view.z;
      sols.Add(key, error, outer_frame, inner_sol_frame);
      double add_sec = add_timer.Seconds();

      {
        MutexLock ml(&mu);
        total_sample_sec += sample_sec;
        total_opt_sec += opt_sec;
        total_add_sec += add_sec;
        MaybeStatus();
      }

      sols_done++;
    }
  }

  // With lock.
  void MaybeStatus() {
    status_per.RunIf([&]() {
        double tot_sec = total_sample_sec + total_opt_sec + total_add_sec;
        double wall_sec = run_timer.Seconds();
        int64_t numer = sols_done.Read();
        int64_t denom = std::max(TARGET_SAMPLES - start_sols_size, int64_t{0});

        std::string oline =
          std::format(AWHITE("outer") " {:016x} = {}",
                       outer_code,
                       boundaries.ColorMaskedBits(outer_code, outer_mask));
        std::string iline =
          std::format(AWHITE("inner") " {:016x} = {}",
                      inner_code,
                      boundaries.ColorMaskedBits(inner_code, inner_mask));

        std::string timing =
          std::format(AGREY("[") ACYAN("{}") "/" AWHITE("{}") AGREY("]") " "
                      "{} sample {} opt ({:.2f}%) {} add {:.2f}/sec",
                      pairs_done.Read(), TOTAL_PAIRS,
                      ANSI::Time(total_sample_sec),
                      ANSI::Time(total_opt_sec),
                      (100.0 * total_opt_sec) / tot_sec,
                      ANSI::Time(total_add_sec),
                      numer / wall_sec);

        std::string bar =
          ANSI::ProgressBar(numer, denom,
                            std::format(ACYAN("|")
                                        AWHITE("{}")
                                        ACYAN("|") " {} total. Save in {}",
                                        what,
                                        sols.Size(),
                                        ANSI::Time(save_per.SecondsLeft())),
                            wall_sec);

        status->EmitStatus({oline, iline, timing, bar});
      });

    save_per.RunIf([&]() {
        sols.Save();
      });
  }

  std::vector<vec2> PlaceHull(const frame3 &frame,
                              const std::vector<int> &hull) const {
    std::vector<vec2> out;
    out.resize(hull.size());
    for (int hidx = 0; hidx < hull.size(); hidx++) {
      int vidx = hull[hidx];
      const vec3 &v_in = small_poly.vertices[vidx];
      const vec3 v_out = transform_point(frame, v_in);
      out[hidx] = vec2{v_out.x, v_out.y};
    }
    return out;
  }

  void Solve() {
    sols_done.Reset();

    if (sols.Size() >= TARGET_SAMPLES) {
      // There's nothing to do, and we didn't even initialize
      // fully!
      return;
    }

    std::vector<std::thread> threads;
    for (int i = 0; i < NUM_THREADS; i++) {
      threads.emplace_back(&TwoPatch::WorkThread, this, i);
    }

    for (;;) {
      {
        MutexLock ml(&mu);
        if (sols.Size() >= TARGET_SAMPLES)
          break;

        if (terminated.test())
          break;
      }
      std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    {
      MutexLock ml(&mu);
      should_die = true;
    }

    for (std::thread &t : threads) t.join();
    threads.clear();

    sols.Save();
    std::string filename = TwoPatchFilename(outer_code, inner_code);
    status->Print("Done in {}: Saved {} sols to {}",
                  ANSI::Time(run_timer.Seconds()),
                  sols.Size(),
                  filename);
    pairs_done++;
  }

};

static void Info() {
  PatchInfo patchinfo = LoadPatchInfo(PATCH_INFO_FILE);
  std::vector<std::pair<uint64_t, PatchInfo::CanonicalPatch>> cc =
    MapToSortedVec(patchinfo.canonical);
  for (int idx = 0; idx < cc.size(); idx++) {
    const auto &[code, _] = cc[idx];
    printf("%d: %llx\n", idx, code);
  }
}

static void UpdateStatus() {
  StatusBar status(1);
  PatchInfo patchinfo = LoadPatchInfo(PATCH_INFO_FILE);

  std::vector<std::pair<uint64_t, PatchInfo::CanonicalPatch>> cc =
    MapToSortedVec(patchinfo.canonical);

  status.Print("Radix: {}", (int)cc.size());

  std::string matrix;

  std::string out;
  int done = 0, partial = 0;
  for (int outer = 0; outer < cc.size(); outer++) {
    const auto &[outer_code, canon1] = cc[outer];
    matrix += std::format(AGREY("{:02}") " ", outer);
    for (int inner = 0; inner < cc.size(); inner++) {
      const auto &[inner_code, canon2] = cc[inner];

      std::string filename = TwoPatchFilename(outer_code, inner_code);
      // NDSolutions<6> sols{filename};
      const std::size_t nsols =
        NDSolutions<6>::SolutionsInFile(filename).value_or(0);
      if (nsols == 0) {
        matrix += AGREY("-");
      } else {
        if (nsols >= TARGET_SAMPLES) {
          AppendFormat(&out, "done {} {}\n", outer, inner);
          status.Print("{} {} done.", outer, inner);
          done++;
          matrix += AGREEN("⏹");
        } else {
          AppendFormat(&out, "reserved {} {}\n", outer, inner);
          status.Print("{} {} {:.2f}%", outer, inner,
                       (nsols * 100.0) / TARGET_SAMPLES);
          partial++;
          matrix += AYELLOW("∘");
        }
      }
    }
    matrix += "\n";
    status.Progressf(outer, cc.size(), "Checking solutions...");
  }

  Util::WriteFile(PATCH_STATUS_FILE, out);
  printf("All done. %d/%d done, %d partial\n",
         done, TOTAL_PAIRS, partial);

  printf("%s\n", matrix.c_str());
}

static void RunWork(StatusBar *status, int start_outer, int start_inner) {
  PatchInfo patchinfo = LoadPatchInfo(PATCH_INFO_FILE);
  status->Print(
      "Total to run: {} * {} = {}",
      (int)patchinfo.canonical.size(),
      (int)patchinfo.canonical.size(),
      (int)(patchinfo.canonical.size() * patchinfo.canonical.size()));
  BigPoly scube = BigScube(DIGITS);

  ArcFour rc(std::format("{}", time(nullptr)));
  std::vector<std::pair<uint64_t, PatchInfo::CanonicalPatch>> cc =
    MapToSortedVec(patchinfo.canonical);
  // Shuffle(&rc, &cc);

  PatchStatus patch_status(PATCH_STATUS_FILE);

  for (int outer = start_outer; outer < cc.size(); outer++) {
    const auto &[code1, canon1] = cc[outer];
    for (int inner = start_inner; inner < cc.size(); inner++) {
      const auto &[code2, canon2] = cc[inner];
      // Skip it if it is registered as complete.
      if (patch_status.done.contains(std::make_pair(outer, inner))) {
        continue;
      }

      std::string filename = TwoPatchFilename(code1, code2);

      // If it is reserved, then only try it if we have
      // the file.
      if (patch_status.reserved.contains(std::make_pair(outer, inner))) {
        if (!Util::ExistsFile(filename)) {
          status->Print("{:x} {:x} is reserved but not by us.",
                        code1, code2);
          continue;
        }
      }

      std::string what = std::format("{}.{}", outer, inner);
      TwoPatch two_patch(what, status, scube,
                         code1, code2, canon1.mask, canon2.mask);
      two_patch.Solve();

      if (CLOUD) {
        status->Print("Copy to storage bucket.\n");
        std::string cmd = std::format(
            "gsutil -q cp {} gs://tom7-ruperts/ && "
            "rm -f {}",
            filename, filename);
        status->Print(AGREY("{}"), cmd.c_str());
        if (0 == std::system(cmd.c_str())) {
          status->Print(AGREEN("OK"));
        } else {
          status->Print(ARED("FAILED"));
        }

        // Either way, add it to the done set.
        patch_status.done.insert(std::make_pair(outer, inner));
        {
          FILE *f = fopen(PATCH_STATUS_FILE, "a");
          fprintf(f, "done %d %d\n", outer, inner);
          fclose(f);
        }
        LIMIT--;
        if (LIMIT == 0) {
          status->Print("Hit limit. Exit.\n");
          return;
        }
      }
    }

    // Only for the first row.
    start_inner = 0;
  }
}

int main(int argc, char **argv) {
  ANSI::Init();

  std::vector<std::string> args;
  for (int i = 1; i < argc; i++) {
    if (0 == strcmp(argv[i], "--threads")) {
      // consume arg
      i++;
      CHECK(i < argc);
      int t = atoi(argv[i]);
      CHECK(t > 0);
      NUM_THREADS = t;
    } else if (0 == strcmp(argv[i], "--cloud")) {
      CLOUD = true;
    } else if (0 == strcmp(argv[i], "--limit")) {
      i++;
      CHECK(i < argc);
      int t = atoi(argv[i]);
      CHECK(t > 0);
      LIMIT = t;
    } else {
      args.push_back(argv[i]);
    }
  }

  printf("Args:");
  for (const std::string &arg : args) {
    printf(" [" AWHITE("%s") "]", arg.c_str());
  }
  printf("\n");

  if (args.size() == 1 && args[0] == "info") {
    Info();
    return 0;
  } else if (args.size() == 1 && args[0] == "updatestatus") {
    UpdateStatus();
    return 0;
  }

  int start_outer = 0;
  int start_inner = 0;
  if (args.size() > 0) start_outer = atoi(args[0].c_str());
  if (args.size() > 1) start_inner = atoi(args[1].c_str());

  std::signal(SIGTERM, HandleSigTerm);

  StatusBar status = StatusBar(4);
  status.Print("{}With " AYELLOW("{}") " threads, run " ACYAN("{}")
               " " ACYAN("{}") APURPLE("+") " (limit " ABLUE("{}") ")",
               CLOUD ? AWHITE("☁️") " " : "",
               NUM_THREADS, start_outer, start_inner, LIMIT);
  RunWork(&status, start_outer, start_inner);

  // done:
  // 0b0000101000010101110101111011111,0b1101110011101000001011100000101
  // 0b0000101000010101110101111011111,0b0000101000010101110101111011111,

  return 0;
}
