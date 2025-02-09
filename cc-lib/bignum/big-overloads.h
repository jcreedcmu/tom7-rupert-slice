// Operator overloads for big.h. These can be very nice if doing
// a lot of math, but they also can obscure what's really going
// on (or pollute the namespace), so they are kept separate.

#ifndef _CCLIB_BIGNUM_BIG_OVERLOADS_H
#define _CCLIB_BIGNUM_BIG_OVERLOADS_H

#include "bignum/big.h"

#include <cstdint>
#include <cstddef>

// TODO modifying operators
// TODO spaceship
// TODO << >> | & ^
// TODO more operators that take int64 on one side
// TODO bigq operators, maybe in a separate file?
// TODO assignment from int

inline BigInt operator ""_b(const char *s) {
  return BigInt(s);
}

inline BigInt operator *(const BigInt &a, const BigInt &b) {
  return BigInt::Times(a, b);
}

inline BigInt operator *(const BigInt &a, int64_t b) {
  return BigInt::Times(a, b);
}

inline BigInt operator *(int64_t a, const BigInt &b) {
  return BigInt::Times(b, a);
}

inline BigInt operator +(const BigInt &a, const BigInt &b) {
  return BigInt::Plus(a, b);
}

inline BigInt operator +(const BigInt &a, int64_t b) {
  return BigInt::Plus(a, b);
}

inline BigInt operator +(int64_t a, const BigInt &b) {
  return BigInt::Plus(b, a);
}

inline BigInt operator /(const BigInt &a, const BigInt &b) {
  return BigInt::Div(a, b);
}

inline BigInt operator /(const BigInt &a, int64_t b) {
  return BigInt::Div(a, b);
}

// Note: To avoid confusion, this is CMod (same semantics as % on
// signed integers in C), not the number theoretic modulus oprator.
inline BigInt operator %(const BigInt &a, const BigInt &b) {
  return BigInt::CMod(a, b);
}

inline int64_t operator %(const BigInt &a, int64_t b) {
  return BigInt::CMod(a, b);
}

inline BigInt operator &(const BigInt &a, const BigInt &b) {
  return BigInt::BitwiseAnd(a, b);
}

inline uint64_t operator &(uint64_t a, const BigInt &b) {
  return BigInt::BitwiseAnd(b, a);
}

inline uint64_t operator &(const BigInt &a, uint64_t b) {
  return BigInt::BitwiseAnd(a, b);
}

inline BigInt operator -(const BigInt &a, const BigInt &b) {
  return BigInt::Minus(a, b);
}

inline BigInt operator -(const BigInt &a, int64_t b) {
  return BigInt::Minus(a, b);
}

inline BigInt operator -(int64_t a, const BigInt &b) {
  return BigInt::Minus(a, b);
}

inline BigInt operator -(const BigInt &a) {
  return BigInt::Negate(a);
}

inline BigInt operator -(BigInt &&a) {
  return BigInt::Negate(std::move(a));
}

inline BigInt operator +(const BigInt &a) {
  return a;
}

inline BigInt operator <<(const BigInt &a, uint64_t bits) {
  return BigInt::LeftShift(a, bits);
}

inline BigInt operator >>(const BigInt &a, uint64_t bits) {
  return BigInt::RightShift(a, bits);
}

// prefix increment
inline BigInt &operator ++(BigInt &a) {
  a = std::move(a) + 1;
  return a;
}

// PERF: Use versions that modify a in place.
// PERF: Versions with int64_t rhs.

inline BigInt &operator<<=(BigInt &a, uint64_t bits) {
  a = a << bits;
  return a;
}

inline BigInt &operator>>=(BigInt &a, uint64_t bits) {
  a = a >> bits;
  return a;
}

inline BigInt &operator+=(BigInt &a, const BigInt &b) {
  a = a + b;
  return a;
}

inline BigInt &operator+=(BigInt &a, int64_t b) {
  a = a + b;
  return a;
}

inline BigInt &operator-=(BigInt &a, const BigInt &b) {
  a = a - b;
  return a;
}

inline BigInt &operator-=(BigInt &a, int64_t b) {
  a = a - b;
  return a;
}

inline BigInt &operator*=(BigInt &a, const BigInt &b) {
  a = a * b;
  return a;
}

inline BigInt &operator/=(BigInt &a, const BigInt &b) {
  a = a / b;
  return a;
}

inline BigInt &operator/=(BigInt &a, int64_t b) {
  a = a / b;
  return a;
}

inline BigInt &operator%=(BigInt &a, const BigInt &b) {
  a = a % b;
  return a;
}

inline BigInt &operator&=(BigInt &a, const BigInt &b) {
  a = a & b;
  return a;
}

inline BigInt &operator&=(BigInt &a, uint64_t b) {
  // Can't make use of the fact that a & b returns uint64_t
  // in the assignment operator.
  a = BigInt(a & b);
  return a;
}

inline bool operator <(const BigInt &a, const BigInt &b) {
  return BigInt::Less(a, b);
}

inline bool operator <(const BigInt &a, int64_t b) {
  return BigInt::Less(a, b);
}

inline bool operator <=(const BigInt &a, const BigInt &b) {
  return BigInt::LessEq(a, b);
}

inline bool operator <=(const BigInt &a, int64_t b) {
  return BigInt::LessEq(a, b);
}

inline bool operator >(const BigInt &a, const BigInt &b) {
  return BigInt::Greater(a, b);
}

inline bool operator >(const BigInt &a, int64_t b) {
  return BigInt::Greater(a, b);
}

inline bool operator >=(const BigInt &a, const BigInt &b) {
  return BigInt::GreaterEq(a, b);
}

inline bool operator >=(const BigInt &a, int64_t b) {
  return BigInt::GreaterEq(a, b);
}

inline bool operator ==(const BigInt &a, const BigInt &b) {
  return BigInt::Eq(a, b);
}

inline bool operator !=(const BigInt &a, const BigInt &b) {
  return !BigInt::Eq(a, b);
}

inline bool operator ==(const BigInt &a, int64_t b) {
  return BigInt::Eq(a, b);
}

inline bool operator ==(int64_t a, const BigInt &b) {
  return BigInt::Eq(b, a);
}

inline bool operator !=(const BigInt &a, int64_t b) {
  return !BigInt::Eq(a, b);
}

inline bool operator !=(int64_t a, const BigInt &b) {
  return !BigInt::Eq(b, a);
}

template<>
struct std::hash<BigInt> {
  std::size_t operator()(const BigInt &k) const {
    // XXX: This just uses the low word, but it would be less
    // surprising to depend on the whole thing (even if it
    // might not actually be more efficient for many uses).
    return (size_t)BigInt::HashCode(k);
  }
};


// Big rational overloads.

inline std::strong_ordering operator<=>(const BigRat &a, const BigRat &b) {
  int c = BigRat::Compare(a, b);
  if (c < 0) return std::strong_ordering::less;
  else if (c == 0) return std::strong_ordering::equal;
  return std::strong_ordering::greater;
}

inline bool operator==(const BigRat &a, const BigRat &b) {
  return (a <=> b) == std::strong_ordering::equal;
}

inline BigRat operator+(const BigRat &a, const BigRat &b) {
  return BigRat::Plus(a, b);
}

inline BigRat operator-(const BigRat &a, const BigRat &b) {
  return BigRat::Minus(a, b);
}

inline BigRat operator*(const BigRat &a, const BigRat &b) {
  return BigRat::Times(a, b);
}

inline BigRat operator/(const BigRat &a, const BigRat &b) {
  return BigRat::Div(a, b);
}

inline BigRat operator-(const BigRat &a) {
  return BigRat::Negate(a);
}

inline BigRat &operator+=(BigRat &a, const BigRat &b) {
  a = a + b;
  return a;
}

inline BigRat &operator-=(BigRat &a, const BigRat &b) {
  a = a - b;
  return a;
}

inline BigRat &operator*=(BigRat &a, const BigRat &b) {
  a = a * b;
  return a;
}

inline BigRat &operator/=(BigRat &a, const BigRat &b) {
  a = a / b;
  return a;
}

template<>
struct std::hash<BigRat> {
  std::size_t operator()(const BigRat &k) const {
    // As above.
    return (size_t)BigRat::HashCode(k);
  }
};


// TODO: Combinations that take integers or bigints.

#endif
