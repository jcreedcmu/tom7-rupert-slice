
// Defines mathematical constants and functions with arbitrary
// precision.

#ifndef _CC_LIB_BIGNUM_BIG_NUMBERS_H
#define _CC_LIB_BIGNUM_BIG_NUMBERS_H

#include <cstdint>

#include "bignum/big.h"

struct BigNumbers {

  // Returns 1 / 10^digits, for use as an epsilon value.
  static BigRat Digits(int64_t digits);

  // Returns a rational approximation of π that is no more than
  // epsilon from the correct value.
  static BigRat Pi(const BigRat &epsilon);

};

#endif
