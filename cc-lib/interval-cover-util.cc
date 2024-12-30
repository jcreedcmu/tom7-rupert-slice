
#include "interval-cover-util.h"

#include <string>
#include <functional>
#include <string_view>

#include "interval-cover.h"

static void TrimSpace(std::string_view *s) {
  while (!s->empty() && (*s)[0] == ' ') s->remove_prefix(1);
  while (!s->empty() && s->back() == ' ') s->remove_suffix(1);
}

static std::string BoolToString(bool b) {
  return b ? "t" : "f";
}

static bool BoolFromString(std::string_view s) {
  TrimSpace(&s);
  CHECK(s.size() == 1 && (s[0] == 't' || s[0] == 'f')) << s;
  return s[0] == 't';
}

std::string IntervalCoverUtil::ToString(const IntervalCover<bool> &ic) {
  return Serialize<bool>(ic, BoolToString);
}

IntervalCover<bool> IntervalCoverUtil::ParseBool(std::string_view s) {
  return Parse<bool>(s, BoolFromString, false);
}
