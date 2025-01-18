
#ifndef _POLYNOMIAL_H
#define _POLYNOMIAL_H

#include <string>
#include <map>
#include <utility>

#include "big.h"

// In a polynomial like 3x^2y - 4y + 7, this represents one of the
// "x^2y", "y", or "" (multiplicative unit).
// Each variable in the term is mapped to its power (> 0).
// The empty term is unit, used to represent a constant term like 7
// above.
struct Term {
  Term() {}
  // returns "" for the unit term.
  std::string ToString() const;

  static std::pair<Term, int> Divide(const Term &n, const Term &d);
  static Term Times(const Term &a, const Term &b);

  std::map<std::string, int> product;
};

// Represents a rational polynomial. Open domain of variables, just
// represented as strings.
struct Polynomial {
  // zero polynomial
  Polynomial() {}

  // Create a 0-degree polynomial representing just the number.
  explicit Polynomial(int a) : Polynomial(BigInt(a)) {}

  explicit Polynomial(const BigInt &a) : Polynomial(BigRat(a)) {}

  explicit Polynomial(const BigRat &a) {
    if (BigRat::Sign(a) != 0) {
      // empty set of variables
      Term constant;
      sum = {{constant, a}};
    }
  }

  explicit Polynomial(const std::string &x, int exponent = 1) {
    Term t;
    if (exponent != 0) {
      t.product = {{x, exponent}};
    }
    sum = {{t, BigRat(1)}};
  }

  // XXX swap order since we usually write this as "ct".
  explicit Polynomial(const Term &t, const BigRat &c) {
    if (BigRat::Sign(c) != 0) {
      sum = {{t, c}};
    }
  }

  // XXX document
  static Polynomial SubstTerm(const Term &t, const Term &src,
                              const Polynomial &dst);

  static Polynomial Exp(const Polynomial &a, int n);

  // For a source term like "x^2" and a polynomial like "3yz",
  // replace all ocurrences of x^2 (including in higher powers like x^3)
  // with "3yz".
  static Polynomial Subst(const Polynomial &p,
                          const Term &src, const Polynomial &dst);

  std::string ToString() const;

  // The partial derivative of p with respect to the variable x.
  static Polynomial PartialDerivative(const Polynomial &p,
                                      const std::string &x);

  static Polynomial Scale(const Polynomial &p, const BigRat &scale);
  static Polynomial Times(const Polynomial &a, const Polynomial &b);
  static Polynomial Plus(const Polynomial &a, const Polynomial &b);

  static bool Eq(const Polynomial &a, const Polynomial &b);

  // This belongs in polynomial-util or needs to be made more
  // general.
  static std::string ToCode(const std::string &type, const Polynomial &p);

  // Each term mapped to its rational coefficient. The coefficient
  // may not be zero.
  std::map<Term, BigRat> sum;
};


// Inline operators follow.

inline bool operator ==(const Term &a, const Term &b) {
  return a.product == b.product;
}

inline bool operator <(const Term &a, const Term &b) {
  return a.product < b.product;
}


inline Polynomial operator -(const Polynomial &p) {
  Polynomial np = p;
  for (auto &tc : np.sum)
    tc.second = BigRat::Negate(std::move(tc.second));
  return np;
}

inline Polynomial operator +(const Polynomial &a, const Polynomial &b) {
  return Polynomial::Plus(a, b);
}

inline Polynomial operator +(const Polynomial &a, int b) {
  return Polynomial::Plus(a, Polynomial(b));
}

inline Term operator *(const Term &a, const Term &b) {
  return Term::Times(a, b);
}

inline Polynomial operator *(const Polynomial &a, const Polynomial &b) {
  return Polynomial::Times(a, b);
}

inline Polynomial operator *(const Polynomial &a, const BigRat &s) {
  return Polynomial::Scale(a, s);
}

inline Polynomial operator *(const BigRat &s, const Polynomial &a) {
  return Polynomial::Scale(a, s);
}

inline Polynomial operator *(int64_t s, const Polynomial &a) {
  return Polynomial::Scale(a, BigRat(s));
}

inline Polynomial operator *(const Polynomial &a, int64_t s) {
  return Polynomial::Scale(a, BigRat(s));
}

inline Polynomial operator -(const Polynomial &a, const Polynomial &b) {
  return a + -b;
}

inline Polynomial operator -(const Polynomial &a, const BigRat &b) {
  return Polynomial::Plus(a, Polynomial(BigRat::Negate(b)));
}

inline bool operator ==(const Polynomial &a, const Polynomial &b) {
  return Polynomial::Eq(a, b);
}

inline Term operator ""_t(const char *s, size_t sz) {
  Term t;
  t.product[std::string(s, sz)] = 1;
  return t;
}

inline Polynomial operator ""_p(unsigned long long a) {
  // XXX check bounds
  return Polynomial((int)a);
}

inline Polynomial operator ""_p(const char *s, size_t sz) {
  // XXX allow leading ints in the string?
  // XXX allow parsing ^2 etc.?
  return Polynomial(std::string(s, sz));
}


#endif
