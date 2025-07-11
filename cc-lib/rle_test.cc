
#include "rle.h"

#include <cstdio>
#include <vector>
#include <string>
#include <cstdint>

#include "base/stringprintf.h"
#include "base/logging.h"
#include "arcfour.h"
#include "randutil.h"
#include "timer.h"
#include "ansi.h"

using uint8 = uint8_t;

using namespace std;

static string ShowVector(const vector<uint8> &v) {
  string s = "{";
  for (int i = 0; i < (int)v.size(); i++) {
    AppendFormat(&s, "{}, ", v[i]);
  }
  return s + "}";
}

static void CheckSameVector(const vector<uint8> &a,
                            const vector<uint8> &b) {
  CHECK_EQ(a.size(), b.size()) << "\n"
                               << ShowVector(a) << "\n" << ShowVector(b);
  for (int i = 0; i < (int)a.size(); i++) {
    CHECK_EQ(a[i], b[i]) << "\n" << ShowVector(a) << "\n" << ShowVector(b);
  }
}

static void DecoderTests() {
  vector<uint8> empty = {};
  vector<uint8> d_empty = RLE::Decompress(empty);
  CHECK(d_empty.empty());

  vector<uint8> simple_run = {
    // Run of 4
    3,
    // Value = 42
    42,
  };
  CheckSameVector({42, 42, 42, 42}, RLE::Decompress(simple_run));

  vector<uint8> simple_singletons = {
    // Singletons
    0, 1, 0, 2, 0, 3, 0, 4,
  };
  CheckSameVector({1, 2, 3, 4}, RLE::Decompress(simple_singletons));

  vector<uint8> small = {
    // Run of 4x42
    3, 42,
    // Then a singleton zero
    0, 0,
    // Run of 2x99
    1, 99,
  };
  CheckSameVector({42, 42, 42, 42, 0, 99, 99},
                  RLE::Decompress(small));
}

static void EncoderTests() {
  // These don't strictly need to encode this way.
  CheckSameVector({3, 42}, RLE::Compress({42, 42, 42, 42}));
  CheckSameVector({3, 42, 0, 0, 1, 99},
                  RLE::Compress({42, 42, 42, 42, 0, 99, 99}));
  CheckSameVector({3, 42, 0, 0, 1, 99, 0, 8},
                  RLE::Compress({42, 42, 42, 42, 0, 99, 99, 8}));
}

int main(int argc, char **argv) {
  ANSI::Init();
  ArcFour rc{"rle_test"};

  DecoderTests();
  EncoderTests();

  int64_t compressed_bytes = 0, uncompressed_bytes = 0;
  #define NUM_TESTS 2000
  Timer timer;
  for (int cutoff = 0; cutoff < 256; cutoff++) {
    if (cutoff % 10 == 0) {
      printf("%.1f%% ... ", (100.0 * cutoff) / 256);
    }

    const uint8 run_cutoff = cutoff;
    for (int test_num = 0; test_num < NUM_TESTS; test_num++) {
      int len = RandTo(&rc, 2048);
      CHECK_LT(len, 2048);
      vector<uint8> bytes;
      bytes.reserve(len);
      for (int j = 0; j < len; j++) {
        if (rc.Byte() < 10) {
          int runsize = rc.Byte() + rc.Byte();
          const uint8 target = rc.Byte();
          while (runsize--) {
            bytes.push_back(target);
            j++;
          }
        } else {
          bytes.push_back(rc.Byte());
        }
      }

      uncompressed_bytes += bytes.size();
      // fprintf(stderr, "Start: %s\n", ShowVector(bytes).c_str());
      vector<uint8> compressed = RLE::CompressEx(bytes, run_cutoff);
      // fprintf(stderr, "Compressed: %s\n", ShowVector(compressed).c_str());
      compressed_bytes += compressed.size();

      vector<uint8> uncompressed;
      CHECK(RLE::DecompressEx(compressed, run_cutoff, &uncompressed))
        << " test_num " << test_num;
      CHECK_EQ(uncompressed.size(), bytes.size());
      for (int i = 0; i < (int)uncompressed.size(); i++) {
        CHECK_EQ(uncompressed[i], bytes[i]) << " test_num "
                                            << test_num
                                            << " byte #" << i;
      }
    }
  }
  double elapsed = timer.Seconds();
  printf("\n"
         "Total uncompressed: %d\n"
         "Total compressed:   %d\n"
         "Average ratio: %.3f:1\n"
         "Time: %s\n"
         "kB/sec (round trip plus validation): %.1f\n",
         (int)uncompressed_bytes,
         (int)compressed_bytes,
         (double)uncompressed_bytes / compressed_bytes,
         ANSI::Time(elapsed).c_str(),
         (uncompressed_bytes / 1000.0) / elapsed);

  return 0;
}
