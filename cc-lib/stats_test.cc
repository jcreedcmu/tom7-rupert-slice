
#include "stats.h"

#include "base/logging.h"
#include "ansi.h"

using Gaussian = Stats::Gaussian;

static void TestGaussian() {
  {
    Gaussian g = Stats::EstimateGaussian({});
    CHECK(std::isnan(g.mean));
    CHECK(std::isnan(g.variance));
    CHECK(std::isnan(g.stddev));
    CHECK(g.num_samples == 0);
  }

  {
    Gaussian g = Stats::EstimateGaussian({7.0});
    CHECK(7.0 == g.mean);
    CHECK(std::isnan(g.variance));
    CHECK(std::isnan(g.stddev));
    CHECK(g.num_samples == 1);
  }

  {
    Gaussian g = Stats::EstimateGaussian({7.0, 7.0});
    CHECK(7.0 == g.mean);
    CHECK(g.variance < 0.000000001) << g.variance;
    CHECK(g.stddev < 0.0000000001);
    CHECK(g.num_samples == 2);
  }

  {
    Gaussian g = Stats::EstimateGaussian({-7.0, 7.0});
    CHECK(g.mean < 0.000000001);
    CHECK(g.variance > 1.0);
    CHECK(g.stddev > 1.0);
    CHECK(g.num_samples == 2);
  }

  {
    Gaussian g = Stats::EstimateGaussian({-0.7, 0.7});
    CHECK(g.mean < 0.000000001);
    CHECK(g.variance < 1.0);
    CHECK(g.stddev < 1.0);
    CHECK(g.num_samples == 2);
  }

}

int main(int argc, char **argv) {
  ANSI::Init();

  TestGaussian();

  printf("OK\n");
  return 0;
}
