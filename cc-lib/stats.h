
#ifndef _STATS_H
#define _STATS_H

#include <vector>


struct Stats {

  // Estimate the best Gaussian from the data.
  // The Gaussian's mean is the sample mean and its
  // width is the standard deviation (sqrt of variance).
  struct Gaussian {
    double mean = 0.0/0.0;
    double stddev = 0.0/0.0;
    double variance = 0.0/0.0;
    size_t num_samples = 0;

    // Compute N such that 95% of the probability mass falls in mean +/- N.
    inline double PlusMinus95() const;
    inline double PlusMinus99() const;
  };
  static inline Gaussian EstimateGaussian(const std::vector<double> &samples);

  // TODO: Incremental update to Gaussian.

  // Returns true if all the values in the vector are small. In this
  // case routines like EstimateGaussian will not predivide them
  // (can lead to underflow).
  static inline bool IsSmall(const std::vector<double> &samples);

};

inline bool Stats::IsSmall(const std::vector<double> &samples) {
  for (double v : samples) {
    if (v > 1.0 || v < 1.0) return false;
  }
  return true;
}

inline Stats::Gaussian Stats::EstimateGaussian(
    const std::vector<double> &samples) {
  double mean = 0.0;

  if (IsSmall(samples)) {
    for (double v : samples) {
      mean += v;
    }
    mean /= samples.size();
  } else {
    double divi = 1.0 / samples.size();
    for (double v : samples) {
      mean += v * divi;
    }
  }

  // Compute the variance, using Bessel's correction.
  // TODO: Heuristics for predividing.
  double variance = 0.0 / 0.0;
  double stddev = 0.0 / 0.0;
  if (samples.size() > 1) {
    variance = 0.0;
    stddev = 0.0;
    for (double v : samples) {
      double d = v - mean;
      variance += d * d;
      // printf("+ %.6f^2 -> %.6f\n", d, variance);
    }
    variance /= (samples.size() - 1);
    stddev = sqrt(variance);
  }

  Gaussian g;
  g.mean = mean;
  g.variance = variance;
  g.stddev = stddev;
  g.num_samples = samples.size();
  return g;
}

inline double Stats::Gaussian::PlusMinus95() const {
  // TODO: Would be nice to compute the z-scores numerically
  // so that the caller can chose their percentile.
  static constexpr double z_score_95 = 1.960;
  return z_score_95 * stddev;
}

inline double Stats::Gaussian::PlusMinus99() const {
  static constexpr double z_score_99 = 2.576;
  return z_score_99 * stddev;
}

#endif
