
#include "byte-set.h"

#include <compare>
#include <cstdint>
#include <cstdio>
#include <string>

#include "base/stringprintf.h"
#include "base/logging.h"

static constexpr bool VERBOSE = false;

bool ByteSet64::Contains(uint8_t b) const {
  switch (type) {
    case EMPTY:
      return false;
    case VALUES:
      for (int i = 0; i < 7; i++)
        if (payload[i] == b)
          return true;
      return false;
    case RANGES:
      for (int i = 0; i < 3; i++) {
        uint8_t start = payload[i * 2];
        uint8_t len = payload[i * 2 + 1];
        if (len == 0) continue;
        uint8_t end = start + len - 1;
        // Handle wrap-around:
        if (start <= end) {
          if (b >= start && b <= end) return true;
        } else {
          if (b >= start || b <= end) return true;
        }
      }
      return false;
    default:
      LOG(FATAL) << "Bad ByteSet64 type: " << type;
  }
}

int ByteSet64::Size() const {
  int count = 0;
  // PERF: We can probably make a faster specialized
  // routine for each representation.
  for (int i = 0; i < 256; i++) {
    if (Contains(i)) count++;
  }
  return count;
}

// PERF: It's possible to represent empty with zero-length
// intervals. If we normalized, we could just test for the EMPTY
// type here.
bool ByteSet64::Empty() const {
  return Size() == 0;
}

ByteSet64 ByteSet64::Union(const ByteSet64 &a, const ByteSet64 &b) {
  ByteSet64 ret = a;
  for (int i = 0; i < 256; i++)
    if (b.Contains(i))
      ret.Add(i);
  return ret;
}

void ByteSet64::Clear() {
  Set(EMPTY);
}

void ByteSet64::Add(uint8_t v) {
  auto AddToRanges = [this](uint8_t v) {
      if (VERBOSE) {
        printf("AddToRanges %02x. Currently: "
               "%02x %02x %02x %02x %02x %02x %02x\n",
               v,
               payload[0],
               payload[1],
               payload[2],
               payload[3],
               payload[4],
               payload[5],
               payload[6]);
      }
      CHECK(type == RANGES);
      // PERF: Better heuristics can produce better intervals here.
      // PERF: We should also merge intervals once they overlap.
      int bestscore = 9999;
      int besti = -1;
      uint8_t beststart = 0;
      uint8_t bestlen = 0;
      for (int i = 0; i < 3; i++) {
        int start = payload[i * 2];
        int length = payload[i * 2 + 1];
        if (InInterval(start, length, v)) {
          // Already included.
          return;
        }

        // We can always include the point unless
        // the length is already 255.
        if (length < 255) {
          // One past end. So 0 is the very end.
          const uint8_t end = start + length;
          // consider wrapping around.
          if (length == 0) {
            // Can claim an empty interval. But
            // this is not better than extending
            // an existing one.
            if (2 < bestscore) {
              besti = i;
              bestscore = 2;
              beststart = v;
              bestlen = 1;
            }
          } else if (start == 0 && v == 255) {
            besti = i;
            bestscore = 1;
            beststart = 255;
            bestlen = length + 1;
          } else if (v == 0 && end == 255) {
            besti = i;
            bestscore = 1;
            beststart = start;
            bestlen = length + 1;
          } else if (v < start) {
            // Expand downward.
            int dist = start - v;
            if (dist < bestscore) {
              besti = i;
              bestscore = dist;
              beststart = v;
              bestlen = length + dist;
            }
          } else if (v >= end) {
            int dist = (int)(v - end) + 1;
            if (dist < bestscore) {
              besti = i;
              bestscore = dist;
              beststart = start;
              bestlen = length + dist;
            }
          } else {
            // TODO: Handle extending wraparound intervals.
          }
        }
      }

      if (besti == -1) {
        Set(RANGES, 0, 255, 0xFF, 1);
      } else {
        payload[besti * 2] = beststart;
        payload[besti * 2 + 1] = bestlen;
      }
    };

  switch (type) {
  case EMPTY: {
    type = VALUES;
    payload[0] = v;
    for (int i = 1; i < 7; i++) {
      payload[i] = 0;
    }
    break;
  }

  case VALUES: {
    for (int i = 0; i < 7; i++) {
      if (payload[i] == v) return;
      if (i > 0 && payload[i] == payload[i - i]) {
        // If there's a duplicate, then we have space
        // to replace it.
        payload[i] = v;
        return;
      }
    }

    // Otherwise, we need to create intervals.
    // std::sort(payload.data(), payload.data() + 7);
    uint8_t tmp[7];
    for (int i = 0; i < 7; i++) {
      tmp[i] = payload[i];
      payload[i] = 0;
    }
    type = RANGES;
    for (int i = 0; i < 7; i++) {
      AddToRanges(tmp[i]);
    }
    // And the one we are trying to add.
    AddToRanges(v);
    break;
  }

  case RANGES:
    AddToRanges(v);
    break;

  default:
    LOG(FATAL) << "Unknown type";
  }
}

void ByteSet64::AddSet(const ByteSet64 &s) {
  // PERF: We could have efficient iteration if we knew that
  // intervals don't overlap and duplicates in the value
  // case were all at the end. We could accomplish this
  // pretty straightforwardly with a Normalize method.
  for (int i = 0; i < 256; i++) {
    if (s.Contains(i)) {
      Add(i);
    }
  }
}

void ByteSet64::AddSet(const ByteSet &s) {
  for (uint8_t v : s) {
    Add(v);
  }
}

ByteSet64::ByteSet64(const ByteSet &s) {
  if (s.Empty()) {
    type = EMPTY;
    return;
  }

  int count = s.Size();
  if (count <= 7) {
    type = VALUES;
    uint8_t last = 0;
    int i = 0;
    for (uint8_t v : s) {
      payload[i++] = v;
      last = v;
    }
    while (i < 7) payload[i++] = last;
    return;
  }

  // Can't represent a length of 256 for the
  // universal set, so we need two ranges:
  // [0, 255) and [255, 256).
  if (count == 256) {
    type = RANGES;
    payload[0] = 0;
    payload[1] = 255;
    payload[2] = 255;
    payload[3] = 1;
    payload[4] = 0;
    payload[5] = 0;
    return;
  }

  // Otherwise, use nontrivial ranges.
  type = RANGES;
  // This could be smarter; when we fail we don't
  // necessarily fail the best way.

  int range_idx = 0;

  // The code below is conceptually removing elements from the
  // set, but it always does this at the beginning and end.
  // So the working set is s intersected with [lower_bound, upper_bound).
  int lower_bound = 0;
  int upper_bound = 256;

  // First, handle wrap-around. Only one interval
  // needs to use this.
  if (s.Contains(0) && s.Contains(255)) {
    int lo = 255;
    int hi = 0;
    while (s.Contains(hi)) hi++;
    while (s.Contains(lo)) lo--;
    CHECK(hi < 255 && lo > 0 && hi <= lo) << "Since we know "
      "this is not the universal set, there must be some "
      "element between 0 and 255 that is not contained. " <<
      StringPrintf("lo: %02x hi: %02x", lo, hi);
    payload[range_idx * 2] = lo + 1;
    payload[range_idx * 2 + 1] = (hi + 256) - (lo + 1);
    range_idx++;
    lower_bound = hi;
    upper_bound = lo;
  }

  while (range_idx < 3) {
    // Find the next interval, greedily.
    while (lower_bound < upper_bound && !s.Contains(lower_bound))
      lower_bound++;

    // Empty set.
    if (lower_bound >= upper_bound) {
      // Remainder of payload will be zero, which is correct.
      return;
    }

    uint8_t start = lower_bound;
    uint8_t length = 0;
    while (lower_bound < upper_bound && s.Contains(lower_bound)) {
      lower_bound++;
      length++;
    }

    payload[range_idx * 2] = start;
    payload[range_idx * 2 + 1] = length;
    range_idx++;
  }

  // If we ran out of space for ranges, we have to approximate the
  // set. This is the place where we should look for heuristics,
  // but for now, just greedily expand the last interval:
  int last_start = payload[4];
  // Find the largest element in the set. The upper bound is
  // the number *after* that.
  while (!s.Contains(upper_bound - 1)) upper_bound--;
  // Something is wrong if we emitted a range that starts after
  // the largest element??
  CHECK(upper_bound > last_start);
  // Now expand the last range to include the largest number.
  // We already handled the universal set, so the length
  // must be in range.
  payload[5] = upper_bound - last_start;
}

bool ByteSet64::operator ==(const ByteSet64 &other) const {
  return (*this <=> other) == std::strong_ordering::equal;
}

std::strong_ordering ByteSet64::operator <=>(const ByteSet64 &other) const {
  // Not much we can do to make this faster unless we normalize?
  for (int i = 0; i < 256; i++) {
    bool a = Contains(i);
    bool b = other.Contains(i);
    if (a != b) {
      return a ? std::strong_ordering::greater :
        std::strong_ordering::less;
    }
  }
  return std::strong_ordering::equal;
}

ByteSet ByteSet64::ToByteSet() const {
  ByteSet s;
  // PERF Easy to do this with fewer steps.
  for (int i = 0; i < 256; i++)
    if (Contains(i))
      s.Add(i);
  return s;
}

uint8_t ByteSet64::GetSingleton() const {
  switch (type) {
  case EMPTY:
    LOG(FATAL) << "GetSingleton on empty.";
    break;
  case VALUES:
    return payload[0];
  case RANGES:
    for (int i = 0; i < 3; i++) {
      // As long as it's not empty, we can use the first
      // element of the range.
      if (payload[i * 2 + 1] > 0) {
        return payload[i * 2];
      }
    }
  default:
    LOG(FATAL) << "Invalid type";
  }
}

std::string ByteSet64::DebugString() const {
  return ToByteSet().DebugString();
}

std::string ByteSet::DebugString() const {
  std::string ret;

  int range_start = 0;
  bool in_range = false;
  auto EndRange = [&](int b) {
      if (b - 1 == range_start) {
        StringAppendF(&ret, " %02x", range_start);
      } else {
        StringAppendF(&ret, " %02x-%02x", range_start, b - 1);
      }
      in_range = false;
    };

  for (int b = 0; b < 256; b++) {
    if (Contains(b)) {
      if (in_range) {
        continue;
      } else {
        range_start = b;
        in_range = true;
      }
    } else {
      if (in_range) {
        EndRange(b);
      } else {
        continue;
      }
    }
  }

  if (in_range) {
    EndRange(256);
  }
  return ret;
}
