#ifndef _AUTO_HISTO_H
#define _AUTO_HISTO_H

#include <cstdint>
#include <algorithm>
#include <vector>
#include <string>
#include <cmath>

#include "utf8.h"
#include "ansi.h"
#include "base/logging.h"
#include "base/stringprintf.h"

// TODO: This is still experimental.
//
// Take one parameter, like "max amount of memory to use."
// Keep exact samples until we reach the memory budget;
// then use those samples to produce a bucketing. From then
// on, just accumulate into buckets.
struct AutoHisto {
  // Processed histogram for external rendering.
  struct Histo {
    std::vector<double> buckets;
    // The nominal left edge of the minimum bucket and right edge of the
    // maximum bucket, although these buckets actually contain data from
    // -infinity on the left and to +infinity on the right (i.e., these
    // are not the actual min and max samples). If the samples are
    // degenerate, we pick something so max > min.
    double min = 0.0, max = 0.0;
    // The width of each bucket (except the "open" buckets at the left
    // and right ends, which are infinite).
    double bucket_width = 0.0;
    // max - min. Always positive, even for degenerate data.
    double histo_width = 0.0;
    // The minimum and maximum value (count) of any bucket.
    // If the histogram is totally empty, this is set to [0, 1].
    double min_value = 0.0, max_value = 0.0;

    // Give the value of a bucket's left, right, or center.
    double BucketLeft(int idx) const { return min + bucket_width * idx; }
    double BucketRight(int idx) const { return BucketLeft(idx + 1); }
    double BucketCenter(int idx) const {
      return min + (bucket_width * (idx + 0.5));
    }
  };

  explicit AutoHisto(int64_t max_samples = 100000) :
    max_samples(max_samples) {
    CHECK(max_samples > 2);
  }

  // Passing count > 1 is the ~same as calling Observe that many times
  // with the same x, but is more efficient when the histogram is
  // bucketed.
  void Observe(double x, int64_t count = 1) {
    if (!std::isfinite(x))
      return;

    total_samples += count;

    if (static_cast<int64_t>(x) != x)
      integral = false;

    if (Bucketed()) {
      AddBucketed(x, count, &data);
    } else {
      if ((int64_t)data.size() + count < max_samples) {
        for (int i = 0; i < count; i++) {
          data.push_back(x);
        }

      } else {
        // Transition to bucketed mode and add the samples after.

        // Sort data ascending so that it's easy to compute quantiles.
        std::sort(data.begin(), data.end());

        // Skip 1% of data, shrinking both sides this amount.
        int64_t skip = max_samples * 0.005;

        // XXX compute bounds, number of actual buckets
        // TODO: Detect integral data better.
        min = data[skip];
        double max = data[data.size() - (1 + skip)];

        // Expand bounds if requested.
        if (flag_min.has_value())
          min = std::min(min, flag_min.value());

        if (flag_max.has_value())
          max = std::max(max, flag_max.value());

        CHECK(min <= max) << StringPrintf("%.17g - %.17g",
                                          min, max);

        width = max - min;

        // If the samples are degenerate, we can choose any
        // bucket width. We use 1.0.
        if (width <= 0.0) width = 1.0;

        num_buckets = max_samples;

        std::vector<double> bucketed(num_buckets, 0.0);
        for (double d : data) AddBucketed(d, 1, &bucketed);
        data = std::move(bucketed);

        AddBucketed(x, count, &data);
      }
    }
  }

  // Add a point to include in the domain of the histogram (e.g. 0.0) even
  // if we don't have samples that would require it.
  void AddFlag(double f) {
    if (!flag_min.has_value()) flag_min = {f};
    if (!flag_max.has_value()) flag_max = {f};

    flag_min = {std::min(flag_min.value(), f)};
    flag_max = {std::max(flag_max.value(), f)};
  }

  // Merge another histogram into this one. Note that when the data
  // are already bucketed, aliasing artifacts are unavoidable.
  void MergeOther(const AutoHisto &other) {
    if (other.Bucketed()) {
      for (int i = 0; i < other.data.size(); i++) {
        if (other.data[i] > 0.0) {
          // Treat it as being in the center of the bucket.
          double x = other.min + other.width * (i + 0.5);
          Observe(x, other.data[i]);
        }
      }
    } else {
      for (double d : other.data) {
        Observe(d);
      }
    }
  }

  // Recommended to use a number of buckets that divides max_samples;
  // otherwise we get aliasing.
  Histo GetHisto(int buckets) const {
    CHECK(buckets >= 1);
    Histo histo;
    histo.buckets.resize(buckets, 0.0);

    if (Bucketed()) {

      histo.min = Min();
      histo.max = Max();
      histo.histo_width = width;
      const double bucket_width = width / buckets;
      histo.bucket_width = bucket_width;

      // Resampling the pre-bucketed histogram.
      for (int64_t b = 0; b < (int64_t)data.size(); b++) {
        // Original data. Use the center of the bucket as its value.
        double v = data[b];
        double center = min + ((b + 0.5) * BucketWidth());
        AddToHisto(&histo, center, v);
      }
      SetHistoScale(&histo);

    } else if (data.empty()) {
      // Without data, the histogram is degenerate.
      // Set bounds of [0, 1] and "max value" of 1.0.
      histo.min = 0.0;
      histo.max = 1.0;
      histo.histo_width = 1.0;
      histo.bucket_width = 1.0 / buckets;
      histo.min_value = 0.0;
      histo.max_value = 1.0;

    } else {

      // Compute temporary histogram from data. We have the
      // number of buckets.
      // TODO: Better detect integer data.
      double minx = data[0], maxx = data[0];
      for (double x : data) {
        minx = std::min(x, minx);
        maxx = std::max(x, maxx);
      }

      if (maxx == minx) {
        // All samples are the same. We need the histogram to
        // have positive width, though.
        maxx = minx + 1;
      }

      CHECK(maxx > minx);

      // If the data are integral, use integer-size buckets.
      // We do this by rounding up to the next integer bucket
      // size, then expanding equally on the low and high ends.
      if (integral) {

        int64_t bucket_width = (int64_t)std::ceil((maxx - minx) / buckets);
        int64_t full_width = bucket_width * buckets;

        // XXX this isn't quite right...
        int64_t leftover = (int64_t)std::round(full_width - (maxx - minx));
        int64_t add_left = leftover / 2;
        int64_t add_right = leftover - add_left;

        histo.min = minx - add_left;
        histo.max = maxx + add_right;
        histo.histo_width = full_width;
        histo.bucket_width = bucket_width;
      } else {
        histo.min = minx;
        histo.max = maxx;
        histo.histo_width = maxx - minx;
        histo.bucket_width = histo.histo_width / buckets;
      }

      // Using the raw samples.
      for (double x : data) {
        AddToHisto(&histo, x, 1.0);
      }
      SetHistoScale(&histo);
    }

    return histo;
  }

  // Vertical layout.
  std::string SimpleANSI(int buckets) const {
    const Histo histo = GetHisto(buckets);

    // Figure out what scale to use for labels.
    // We need to fit these all within the same column length.
    int places = 1;
    // Maximum chararacter length of label.
    int max_label_length = 1;
    while (places < 17) {
      // Are there duplicate labels? There would always be consecutive
      // duplicates if so, since these appear in order.
      bool ok = true;
      max_label_length = 1;
      std::string prev_label = "";
      for (int bidx = 0; bidx < (int)histo.buckets.size(); bidx++) {
        std::string label =
          std::format("{:.{}f}", histo.BucketLeft(bidx), places);
        // We want to compute the max label length even if we fail,
        // since we might have reached max precision.
        if (label == prev_label)
          ok = false;
        max_label_length = std::max((int)label.size(), max_label_length);
        prev_label = std::move(label);
      }

      if (ok) {
        break;
      }
      places++;
    }

    const int label_column = max_label_length;

    std::string ret;
    for (int bidx = 0; bidx < (int)histo.buckets.size(); bidx++) {
      const std::string label =
        PadLeft(std::format("{:.{}f}", histo.BucketLeft(bidx), places),
                label_column);
      const std::string count =
        StringPrintf("%lld", (int64_t)histo.buckets[bidx]);
      const int BAR_CHARS = 70 - label_column;
      double f = histo.buckets[bidx] / histo.max_value;

      const bool before = f > 0.5;
      const uint32_t bar_color = (bidx & 1) ? 0xc8c880FF : 0xbebe76FF;

      StringAppendF(&ret, "%s " AFGCOLOR(32, 32, 23, "|"), label.c_str());
      if (before) {
        int chars_used = 2 + (int)count.size();
        std::string bar = FilledBar(BAR_CHARS, chars_used, f);
        StringAppendF(&ret, "%s%s" " %s " ANSI_RESET "%s%s" ANSI_RESET,
                      // But with black foreground.
                      ANSI::BackgroundRGB32(bar_color).c_str(),
                      ANSI::ForegroundRGB32(0x786800FF).c_str(),
                      count.c_str(),
                      ANSI::ForegroundRGB32(bar_color).c_str(),
                      bar.c_str());
      } else {
        std::string bar = FilledBar(BAR_CHARS, 0, f);
        StringAppendF(&ret, "%s%s" ANSI_RESET " " AGREY("%s"),
                      ANSI::ForegroundRGB32(bar_color).c_str(),
                      bar.c_str(), count.c_str());
      }

      StringAppendF(&ret, "\n");
    }
    return ret;
  }

  // Space-efficient horizontal layout (two lines).
  std::string SimpleHorizANSI(int buckets) const {
    // TODO: In most cases, the factor limiting resolution is
    // the width of the labels themselves. We should probably
    // render at native resolution and just label it like an
    // axis.
    const Histo histo = GetHisto(buckets);

    std::vector<std::string> labels;
    int max_label = 0;
    for (int bidx = 0; bidx < (int)histo.buckets.size(); bidx++) {
      const std::string label =
        integral ?
        StringPrintf("%lld", (int64_t)histo.BucketLeft(bidx)) :
        StringPrintf("%.1f", histo.BucketLeft(bidx));
      labels.emplace_back(label);
      max_label = std::max(max_label, (int)label.size());
    }

    const int bar_width = max_label + 1;

    // Column data.
    std::string ret;
    for (int bidx = 0; bidx < (int)histo.buckets.size(); bidx++) {
      std::string fcc =
        FilledColumnChar(histo.buckets[bidx] / histo.max_value);
      std::string bar;
      for (int i = 0; i < bar_width; i++) {
        bar += fcc;
      }

      if (bidx & 1) {
        StringAppendF(&ret, AFGCOLOR(200, 200, 128, "%s"),
                      bar.c_str());
      } else {
        StringAppendF(&ret, AFGCOLOR(190, 190, 118, "%s"),
                      bar.c_str());
      }
    }
    StringAppendF(&ret, "\n");

    // Column labels.
    for (int bidx = 0; bidx < (int)histo.buckets.size(); bidx++) {
      std::string label = PadLeft(labels[bidx], bar_width);
      if (bidx & 1) {
        StringAppendF(&ret, AFGCOLOR(170, 170, 170, "%s"), label.c_str());
      } else {
        StringAppendF(&ret, AFGCOLOR(150, 150, 150, "%s"), label.c_str());
      }
    }
    return ret;
  }

  // Single line, unlabeled, uncolored. Always returns a string of
  // length "buckets" (codepoints).
  std::string UnlabeledHoriz(int buckets) const {
    const Histo histo = GetHisto(buckets);
    // Column data.
    std::string ret;
    for (int bidx = 0; bidx < (int)histo.buckets.size(); bidx++) {
      ret += FilledColumnChar(histo.buckets[bidx] / histo.max_value);
    }
    return ret;
  }

  bool Empty() const { return total_samples == 0; }

  int64_t NumSamples() const { return total_samples; }

  bool IsIntegral() const { return integral; }

  // TODO: Simple one-line ANSI histo with colored bars.

  // Probably should have the caller do printing.
  void PrintSimpleANSI(int buckets) const {
    printf("%s", SimpleANSI(buckets).c_str());
  }

  std::string SimpleAsciiString(int buckets) const {
    std::string ret = StringPrintf("%lld samples in %d buckets. "
                                   "%.6f min. %.6f max\n",
                                   total_samples, buckets,
                                   Min(), Max());

    const Histo histo = GetHisto(buckets);


    for (int bidx = 0; bidx < (int)histo.buckets.size(); bidx++) {
      StringAppendF(&ret, "%.4f: %.4f (%.4f%%)\n",
                    histo.BucketLeft(bidx),
                    histo.buckets[bidx],
                    (histo.buckets[bidx] * 100.0) / total_samples);
    }

    return ret;
  }

 private:

  // Single character.
  static std::string FilledColumnChar(float f) {
    const int px = std::clamp((int)std::round(f * 8), 0, 8);
    if (px == 0) return " ";
    else return UTF8::Encode(0x2580 + px);
  }

  // Generate a bar using unicode characters that is
  // (f * chars * 8) - (used * 8) pixels long. (The 'used'
  // parameter is for when part of the bar is already
  // output, like with numbers over it.)
  // To unicode-utils? Or ANSIImage?
  static std::string FilledBar(int chars, int used, float f) {
    if (chars <= 0) return "";
    if (used >= chars) return "";
    // integer number of pixels
    f = std::clamp(f, 0.0f, 1.0f);
    int px = (int)std::round(f * (chars * 8));
    px = std::max(px - used * 8, 0);
    int full = px / 8;

    std::string ret;
    for (int i = 0; i < full; i++) {
      ret += UTF8::Encode(0x2588);
    }

    int remain = chars - full;
    if (remain > 0) {
      int partial = px % 8;
      if (partial) {
        // partial
        ret += UTF8::Encode(0x2590 - partial);
        remain--;
      }

      /*
      for (int i = 0; i < remain; i++) {
        ret.push_back(' ');
      }
      */
    }
    return ret;
  }

  static std::string PadLeft(std::string s, int n) {
    while ((int)s.size() < n) s = " " + s;
    return s;
  }

  bool Bucketed() const { return num_buckets != 0; }
  // only when Bucketed
  double Min() const { return min; }
  double Max() const { return min + width; }
  double BucketWidth() const { return width / num_buckets; }

  static void SetHistoScale(Histo *h) {
    CHECK(!h->buckets.empty());
    double minv = h->buckets[0];
    double maxv = minv;
    for (double v : h->buckets) {
      minv = std::min(v, minv);
      maxv = std::max(v, maxv);
    }
    h->min_value = minv;
    h->max_value = maxv;
  }

  static void AddToHisto(Histo *h, double x, double count) {
    // printf("Add v %.1f * ct %.1f: ", x, count);
    double f = (x - h->min) / h->histo_width;
    int64_t bucket = std::clamp((int64_t)(f * h->buckets.size()),
                                (int64_t)0,
                                (int64_t)h->buckets.size() - 1);
    // printf("frac = %.3f  bucket = %lld\n", f, bucket);
    h->buckets[bucket] += count;
  }

  void AddBucketed(double x, int64_t count, std::vector<double> *v) {
    double f = (x - min) / width;
    int64_t bucket = std::clamp((int64_t)(f * num_buckets),
                                (int64_t)0,
                                num_buckets - 1);
    (*v)[bucket] += count;
  }


  // This represents points that we want to include in the final histogram
  // (e.g. 0.0) even if there are no data points that would require them.
  std::optional<double> flag_min, flag_max;

  // This either represents the exact data (until we exceed max_samples)
  // or the bucketed data (once we've decided on min, max, buckets).
  std::vector<double> data;
  double min = 0.0, width = 0.0;
  int64_t max_samples = 0;
  // If 0, then we're still collecting samples.
  int64_t num_buckets = 0;
  int64_t total_samples = 0;
  // True if we've only ever observed integer samples.
  bool integral = true;
};

#endif
