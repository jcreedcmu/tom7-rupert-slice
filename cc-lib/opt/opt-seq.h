
// "Open loop" vesion of optimizer. It gives you a sequence of
// arguments to test on, using the score as guidance. This is
// accomplished by running the optimizer in another thread.

#ifndef _CC_LIB_OPT_OPT_SEQ_H
#define _CC_LIB_OPT_OPT_SEQ_H

#include <condition_variable>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <thread>
#include <utility>
#include <vector>

struct OptSeq {
  // Pair of [low, high] bounds for each parameter.
  OptSeq(const std::vector<std::pair<double, double>> &bounds);
  ~OptSeq();
  // TODO: Save/Load

  // The client should make a call to Next() to get an argument,
  // evaluate it, and then call Result with its score. Lower
  // scores are preferable.
  std::vector<double> Next();
  void Result(double d);

  // Get the overall best pair (lowest score) of argument and
  // score. Always has a value once there has been one call to
  // Result.
  std::optional<std::pair<std::vector<double>, double>> GetBest();

  size_t size() const { return bounds.size(); }

 private:
  void Observe(const std::vector<double> &arg, double d);
  void OptThread();
  double Eval(std::span<const double> args);

  std::mutex m;
  std::condition_variable c;
  std::unique_ptr<std::thread> th;

  // Lower and upper bound for each optimization parameter.
  std::vector<std::pair<double, double>> bounds;
  // An arg vector, pending for the next call to Next(). Always
  // the correct size, but only contains meaningful data if has_arg
  // is true.
  std::vector<double> arg;
  // A result, pending to be consumed by the optimization thread.
  std::optional<double> result;
  // Previous sequence of optimized values.
  std::vector<std::pair<std::vector<double>, double>> history;

  std::optional<std::pair<std::vector<double>, double>> best;
  bool has_arg = false;
  bool should_die = false;
};

#endif
