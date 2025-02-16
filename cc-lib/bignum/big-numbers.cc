#include "bignum/big-numbers.h"

#include <cstdio>
#include <cstdint>
#include <deque>

#include "bignum/big.h"
#include "bignum/big-overloads.h"

BigRat BigNumbers::Digits(int64_t digits) {
  return BigRat(BigInt(1), BigInt::Pow(BigInt(10), digits));
}

BigInt BigNumbers::Pow10(int64_t digits) {
  return BigInt::Pow(BigInt(10), digits);
}

// TODO: Might make sense to expose this too?
//
// Represents the tail of the Taylor series expansion of
// arctan(1/x).
namespace {
struct ArctanSeries {
  // for 1/x.
  explicit ArctanSeries(int x) : xx(x * x), xpow(x) {}

  // Return the 0-based index of the current term (i.e. the term that
  // Peek or Pop returns).
  int N() const { return n - (int)terms.size(); }

  // This pops (and removes) the first element of the series tail,
  // computing more if necessary.
  BigRat Pop() {
    if (terms.empty()) Push();
    assert(!terms.empty());
    BigRat r = std::move(terms.front());
    terms.pop_front();
    return r;
  }

  // Peek at the next element without removing it.
  const BigRat &Peek() {
    if (terms.empty()) Push();
    assert(!terms.empty());
    return terms.front();
  }

  // Bound on the sum of the tail, which is also a bound on the
  // error of of the series so far (against arctan(1/x)).
  // This sequence qualifies for the "alternating series test"
  // aka "Leibniz's rule", which says that the first term bounds
  // the sum.
  BigRat Bound() {
    return BigRat::Abs(Peek());
  }

 private:
  void Push() {
    //         (1/x)^(2n + 1)
    // (-1)^n --------------
    //         2n + 1

    // computed as
    //
    //           (-1)^n
    //          --------
    //          x^(2n+1)
    //        ------------
    //          2n + 1

    const int v = 2 * n + 1;
    // Signs alternate positive and negative.
    BigRat numer = BigRat(BigInt((n & 1) ? -1 : 1), xpow);
    terms.push_back(BigRat::Div(numer, BigInt(v)));

    // Increase exponent by 2.
    xpow = xpow * xx;
    n++;
  }

  // x^2
  const int64_t xx = 0;
  // Next term to be computed.
  int n = 0;
  // the power of x for term n.
  BigInt xpow;
  // PERF: We don't need a deque here any more. It can just
  // be the single next term.
  std::deque<BigRat> terms;
};
}  // namespace

BigRat BigNumbers::Pi(const BigRat &epsilon) {
  static constexpr bool VERBOSE = false;

  // We approximate π / 4, so rescale epsilon to that.
  const BigRat four(4);
  const BigRat target_epsilon = epsilon / four;

  if (VERBOSE) {
    printf("Compute pi with epsilon = %s\n",
           target_epsilon.ToString().c_str());
  }

  // https://en.wikipedia.org/wiki/Machin-like_formula
  // π / 4 = 4 * arctan(1/5) - arctan(1/239)
  // arctan(x) = Σ (-1)^n * (x^(2n + 1)) / (2n + 1)
  //    = x^1 / 1 - x^3 / 3 + x^5 / 5 - x^7 / 7 + ...

  ArctanSeries a(5), b(239);

  BigRat sum = BigRat(0);
  for (;;) {
    if (VERBOSE) printf("Enter with sum = %s\n", sum.ToString().c_str());
    // Total error could be more than each one on its own.
    BigRat err_bound = (four * a.Bound()) + b.Bound();
    if (err_bound < target_epsilon) {
      // We computed π / 4.
      return sum * four;
    }

    // Add the larger of the two terms.
    const BigRat &terma = a.Peek();
    const BigRat &termb = b.Peek();

    BigRat terma4 = terma * four;

    // Add the term that has the larger magnitude.
    if (BigRat::Abs(terma4) > BigRat::Abs(termb)) {
      if (VERBOSE)
        printf("Add term #%d of a: %s * 4\n",
               a.N(),
               terma.ToString().c_str());
      sum = sum + terma4;
      a.Pop();
    } else {
      if (VERBOSE)
        printf("Subtract term #%d of b: %s\n",
               b.N(),
               termb.ToString().c_str());

      sum = sum - termb;
      b.Pop();
    }
  }
}
