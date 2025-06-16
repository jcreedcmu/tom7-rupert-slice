// Various representations of a set of bytes, for program
// analysis applications.

#ifndef _CC_LIB_BYTE_SET_H
#define _CC_LIB_BYTE_SET_H

#include <bit>
#include <cstddef>
#include <initializer_list>
#include <iterator>
#include <cstdint>
#include <compare>
#include <string>

#include "base/logging.h"
#include "base/macros.h"

// Each of the following is a lattice (i.e. has subset, union, and
// intersection operations) representing a set of bytes.

// Represents the set of bytes exactly (the "free lattice").
struct ByteSet {
  ByteSet() { u.a = u.b = u.c = u.d = 0; }
  inline ByteSet(const std::initializer_list<uint8_t> &values);

  inline bool Empty() const;
  inline bool Contains(uint8_t v) const;
  inline void Add(uint8_t v);
  inline void AddSet(const ByteSet &other);
  inline int Size() const;

  std::string DebugString() const;

  inline static ByteSet Top();
  inline static ByteSet Bottom();

  inline static ByteSet Singleton(uint8_t v);

  inline static ByteSet Union(const ByteSet &s, const ByteSet &t);
  inline static ByteSet Intersection(const ByteSet &s, const ByteSet &t);

  // true if a <= b.
  static bool Subset(const ByteSet &a, const ByteSet &b) {
    return a == Intersection(a, b);
  }

  // Get one element from the set; intended for uses where
  // the set has size 1. Aborts if the set is empty.
  uint8_t GetSingleton() const {
    auto GetSingleBit = [](int offset, uint64_t w) {
        return offset + std::countl_zero<uint64_t>(w);
      };
    if (u.a) return GetSingleBit(0, u.a);
    if (u.b) return GetSingleBit(64, u.b);
    if (u.c) return GetSingleBit(128, u.c);
    if (u.d) return GetSingleBit(192, u.d);
    LOG(FATAL) << "GetSingleton on empty ByteSet.";
  }

  template<class F>
  ByteSet Map(const F &f) const {
    ByteSet ret;
    for (uint8_t v : *this) {
      ret.Add(f(v));
    }
    return ret;
  }

  void Clear() {
    u.a = u.b = u.c = u.d = 0;
  }

  // Iterates over bytes that are set, in ascending order.
  class const_iterator {
   public:
    using value_type = uint8_t;
    using difference_type = std::ptrdiff_t;
    using iterator_category = std::input_iterator_tag;

    const_iterator(const ByteSet *bs, int idx_in) : bs(bs), idx(idx_in) {
      idx = NextFrom(idx);
    }

    uint8_t operator*() const {
      // Invalid to dereference the end pointer.
      ASSUME(idx >= 0 && idx < 256);
      return (uint8_t)idx;
    }
    const_iterator& operator++() {
      // Invalid to increment past the end.
      ASSUME(idx >= 0 && idx < 256);
      idx = NextFrom(idx + 1);
      return *this;
    }
    bool operator==(const const_iterator& other) const {
      return idx == other.idx;
    }
    bool operator!=(const const_iterator& other) const {
      return idx != other.idx;
    }

   private:
    // Find the next iterator position at idx or later:
    // While idx is not end, and idx is not in the set,
    // increment it.
    int NextFrom(int idx) {
      ASSUME(idx >= 0 && idx <= 256);
      while (idx != 256) {
        uint8_t v = idx;
        int i = v >> 6;
        int bit = v & 0b00111111;
        // Mask zeroes for bits we've already passed; we can use
        // countl_zero to quickly find the first set bit, but
        // we don't want to count ones behind the iterator.
        uint64_t masked_word = bs->u.words[i] & ((~uint64_t{0}) >> bit);
        if (masked_word == 0) {
          // No bits left in this word; skip to the next.
          i++;
          idx = i * 64;
        } else {
          // Then we have another bit here.
          int one_bit = std::countl_zero<uint64_t>(masked_word);
          return i * 64 + one_bit;
        }
      }
      return idx;
    }

    const ByteSet *bs = nullptr;
    // 0-256, where 256 represents the end iterator.
    // When 0-255, it is a byte in the set.
    int idx = 0;
  };

  const_iterator begin() const { return const_iterator(this, 0); }
  const_iterator end() const { return const_iterator(this, 256); }

  // Weirdly, we need both operator== and operator<=>.
  bool operator ==(const ByteSet &other) const {
    for (int i = 0; i < 4; i++)
      if (u.words[i] != other.u.words[i])
        return false;
    return true;
  }

  std::strong_ordering operator <=>(const ByteSet &other) const {
    for (int i = 0; i < 4; i++) {
      auto ord = u.words[i] <=> other.u.words[i];
      if (ord != std::strong_ordering::equal) return ord;
    }
    return std::strong_ordering::equal;
  }

  // Details exposed for benchmarking.
  inline int SizeSIMD() const;
  [[deprecated]]
  inline int SizeASM() const;

 private:
  friend class const_iterator;
  union {
    struct {
      uint64_t a;
      uint64_t b;
      uint64_t c;
      uint64_t d;
    };
    uint64_t words[4];
  } u;
  static_assert(sizeof(u) == 32);
};


// Packed 64-bit representation. Of course no such representation can
// represent all sets accurately, so this always represents a
// *superset* of what has been inserted. For small counts it is
// exact, but I would not rely on any specific behavior.
//
// Note the only advantage of this is its compactness. The ByteSet
// implementation is way faster.
struct ByteSet64 {
  // Deterministic. But note that there are multiple ways of
  // representing a set, and this must be lossy for cardinality
  // reasons. Always represents a superset of the input (and
  // ideally the same set).
  explicit ByteSet64(const ByteSet &s);
  ByteSet64() : type(EMPTY) {
    for (int i = 0; i < 7; i++) payload[i] = 0;
  }

  static ByteSet64 Singleton(uint8_t b) {
    ByteSet64 s;
    s.Set(VALUES, b, b, b, b, b, b, b);
    return s;
  }

  ByteSet ToByteSet() const;
  std::string DebugString() const;

  // imperative union
  void AddSet(const ByteSet64 &other);
  void AddSet(const ByteSet &other);

  static ByteSet64 Union(const ByteSet64 &a, const ByteSet64 &b);

  // Get one element from the set; intended for uses where
  // the set has size 1. Aborts if the set is empty.
  uint8_t GetSingleton() const;

  bool Contains(uint8_t) const;
  // Not cheap because of the possibility of duplicate
  // values or overlapping ranges.
  int Size() const;

  bool Empty() const;
  void Clear();
  void Add(uint8_t b);

  // Allows iterating over bytes that are set. Order is deterministic
  // but arbitrary. Note that this internally iterates over all 256
  // values, even if it is sparse. (PERF: This can be improved a lot,
  // but it probably requires normalizing the representation.)
  class const_iterator {
   public:
    using value_type = uint8_t;
    using difference_type = std::ptrdiff_t;
    // using reference = uint8_t;
    using iterator_category = std::input_iterator_tag;

    const_iterator(const ByteSet64 *bs, int idx_in) : bs(bs), idx(idx_in) {
      // Advance to the next set bit:
      while (idx < 256 && !bs->Contains(idx)) ++idx;
    }

    uint8_t operator*() const { return (uint8_t)idx; }
    const_iterator& operator++() {
      ++idx;
      while (idx < 256 && !bs->Contains(idx)) ++idx;
      return *this;
    }
    bool operator==(const const_iterator& other) const {
      return idx == other.idx;
    }
    bool operator!=(const const_iterator& other) const {
      return idx != other.idx;
    }

   private:
    const ByteSet64 *bs = nullptr;
    // 0-256, where 256 represents the end iterator.
    // When 0-255, it is a byte in the set.
    int idx = 0;
  };

  const_iterator begin() const { return const_iterator(this, 0); }
  const_iterator end() const { return const_iterator(this, 256); }

  bool operator ==(const ByteSet64 &other) const;
  std::strong_ordering operator <=>(const ByteSet64 &other) const;

  template<class F>
  ByteSet64 Map(const F &f) const {
    ByteSet64 ret;
    for (uint8_t v : *this) {
      ret.Add(f(v));
    }
    return ret;
  }

  // below is nominally private, but used in testing.
  friend class const_iterator;

  // The representation always fits in 8 bytes, but uses
  // different formats.
  //
  // TODO: So many possible tricks here. But since we use
  // so few types, we could consider allocating some type
  // bits to something else (like "has zero"). Another
  // thing we could do is give bitmasks (known-1 and
  // known-0).
  enum Type : uint8_t {
    EMPTY = 0,
    // 7 specific values. To store fewer than 7 values,
    // repeat the last value.
    VALUES = 1,
    // 3 ranges, of the form (start,length). Order by
    // start byte and use ranges with length=0 if there.
    // are fewer than 3. Note that the ranges can wrap
    // around!
    RANGES = 2,
  };
  uint8_t type = 0;
  uint8_t payload[7] = {};

  static bool InInterval(uint8_t start, uint8_t len, uint8_t v) {
    int end = (int)start + (int)len;
    if (v >= start && v < end) return true;
    if (v < start && v + 256 < end) return true;
    return false;
  }
  void Set(uint8_t typ, uint8_t p0 = 0, uint8_t p1 = 0, uint8_t p2 = 0,
           uint8_t p3 = 0, uint8_t p4 = 0, uint8_t p5 = 0, uint8_t p6 = 0) {
    type = typ;
    payload[0] = p0;
    payload[1] = p1;
    payload[2] = p2;
    payload[3] = p3;
    payload[4] = p4;
    payload[5] = p5;
    payload[6] = p6;
  }
};
static_assert(sizeof(ByteSet64) == 8);


// Inline implementations follow.

ByteSet::ByteSet(const std::initializer_list<uint8_t> &values) : ByteSet() {
  for (uint8_t v : values) {
    Add(v);
  }
}

bool ByteSet::Contains(uint8_t v) const {
  int i = v >> 6;
  int bit = v & 0b00111111;
  return !!(1 & (u.words[i] >> (63 - bit)));
}

void ByteSet::Add(uint8_t v) {
  int i = v >> 6;
  int bit = v & 0b00111111;
  u.words[i] |= uint64_t{1} << (63 - bit);
}

void ByteSet::AddSet(const ByteSet &other) {
  u.a |= other.u.a;
  u.b |= other.u.b;
  u.c |= other.u.c;
  u.d |= other.u.d;
}

ByteSet ByteSet::Top() {
  ByteSet s;
  s.u.a = s.u.b = s.u.c = s.u.d = ~uint64_t{0};
  return s;
}

ByteSet ByteSet::Bottom() {
  return ByteSet();
}

ByteSet ByteSet::Singleton(uint8_t v) {
  ByteSet s;
  s.Add(v);
  return s;
}

ByteSet ByteSet::Union(const ByteSet &s, const ByteSet &t) {
  ByteSet ret;
  ret.u.a = s.u.a | t.u.a;
  ret.u.b = s.u.b | t.u.b;
  ret.u.c = s.u.c | t.u.c;
  ret.u.d = s.u.d | t.u.d;
  return ret;
}

ByteSet ByteSet::Intersection(const ByteSet &s, const ByteSet &t) {
  ByteSet ret;
  ret.u.a = s.u.a & t.u.a;
  ret.u.b = s.u.b & t.u.b;
  ret.u.c = s.u.c & t.u.c;
  ret.u.d = s.u.d & t.u.d;
  return ret;
}

bool ByteSet::Empty() const {
  return (u.a | u.b | u.c | u.d) == 0;
}

int ByteSet::Size() const {
  // TODO PERF: This version with std::popcount compiles into
  // SIMD instructions with clang, which takes about twice as
  // much time as using the POPCNT instruction. The assembly
  // version below is straightforward, but I had problems where
  // inline assembly would work but only in the presence of
  // printfs. So I'm not confident that I am using the
  // register constraints correctly.
  return SizeSIMD();
}

// Details exposed for benchmarking.
int ByteSet::SizeSIMD() const {
  return std::popcount<uint64_t>(u.a) +
    std::popcount<uint64_t>(u.b) +
    std::popcount<uint64_t>(u.c) +
    std::popcount<uint64_t>(u.d);
}

[[deprecated]]
int ByteSet::SizeASM() const {
  uint64_t a = u.a, b = u.b, c = u.c, d = u.d;
  uint64_t a_count = 0, b_count = 0, c_count = 0, d_count = 0;

  __asm__ volatile ("popcnt %1, %0\n\t" : "=r"(a_count) : "r"(a) :);
  __asm__ volatile ("popcnt %1, %0\n\t" : "=r"(b_count) : "r"(b) :);
  __asm__ volatile ("popcnt %1, %0\n\t" : "=r"(c_count) : "r"(c) :);
  __asm__ volatile ("popcnt %1, %0\n\t" : "=r"(d_count) : "r"(d) :);

  return a_count + b_count + c_count + d_count;
}

#endif
