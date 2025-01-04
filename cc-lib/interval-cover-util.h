

#ifndef _CC_LIB_INTERVAL_COVER_UTIL_H
#define _CC_LIB_INTERVAL_COVER_UTIL_H

#include <charconv>
#include <functional>
#include <string>
#include <string_view>

#include "base/stringprintf.h"
#include "interval-cover.h"
#include "util.h"

struct IntervalCoverUtil {

  // Routines to serialize common IntervalCover formats.

  // Using "t" and "f".
  static std::string ToString(const IntervalCover<bool> &ic);
  static IntervalCover<bool> ParseBool(std::string_view s);


  // Generic Serialization. The t-to-string function cannot output
  // newlines and should not care about leading or trailing
  // whitespace.
  template<class T>
  inline static std::string Serialize(
      const IntervalCover<T> &ic,
      const std::function<std::string(const T&)> &ttos);

  template<class T>
  inline static IntervalCover<T> Parse(
      std::string_view contents,
      const std::function<T(std::string_view)> &stof,
      T default_value = T{});
};


// Template implementations follow.

template<class T>
inline std::string IntervalCoverUtil::Serialize(
    const IntervalCover<T> &ic,
    const std::function<std::string(const T&)> &ttos) {
  std::string out;
  for (uint64_t p = 0; !IntervalCover<T>::IsAfterLast(p); p = ic.Next(p)) {
    typename IntervalCover<T>::Span s = ic.GetPoint(p);
    StringAppendF(&out, "%llu %llu %s\n",
                  s.start, s.end, ttos(s.data).c_str());
  }

  return out;
}

template<class T>
inline IntervalCover<T> IntervalCoverUtil::Parse(
    std::string_view contents,
    const std::function<T(std::string_view)> &stof,
    T default_value) {

  auto ParseU64 = [](std::string_view s, uint64_t &u) -> bool {
      const auto &[ptr_, err] =
        std::from_chars(s.data(), s.data() + s.size(), u);
      // i.e., success
      return err == std::errc();
    };

  IntervalCover<T> ret(std::move(default_value));
  std::vector<std::string> lines = Util::SplitToLines(contents);
  for (const std::string &full_line : lines) {
    std::string norml_line = Util::LoseWhiteL(Util::LoseWhiteR(full_line));
    std::string_view line(full_line);
    if (line.empty()) continue;

    std::string_view start_tok = Util::NextToken(&line, ' ');
    std::string_view end_tok = Util::NextToken(&line, ' ');

    uint64_t start = 0, end = 0;
    CHECK(ParseU64(start_tok, start));
    CHECK(ParseU64(end_tok, end));
    ret.SetSpan(start, end, stof(line));
  }
  return ret;
}

#endif
