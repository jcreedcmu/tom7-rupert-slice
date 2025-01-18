
#ifndef _POLYNOMIAL_H
#define _POLYNOMIAL_H

#include <string>
#include <map>
#include <utility>
#include <optional>

#include "base/logging.h"
#include "base/stringprintf.h"
#include "big.h"

// In a polynomial like 3x^2y - 4y + 7, this represents one of the
// "x^2y", "y", or "" (multiplicative unit).
// Each variable in the term is mapped to its power (> 0).
// The empty term is unit, used to represent a constant term like 7
// above.
struct Term {
  Term() {}
  // returns "" for the unit term.
  std::string ToString() const {
    std::string ret;
    for (const auto &[x, e] : product) {
      if (e == 1) ret += x;
      else StringAppendF(&ret, "%s^%d", x.c_str(), e);
    }
    return ret;
  }

  static std::pair<Term, int> Divide(const Term &n, const Term &d) {
    // we have eg. t = x^3y^2z^4 and src = xz^2. Then
    // we to factor this to xy^2 * (xz^2)^2. xy^2 is the "remainder"
    // and the outer ^2 on (xz^2)^2 is the power.

    // Find the highest power.
    std::optional<int> max_power = std::nullopt;
    for (const auto &[x, e] : d.product) {
      // XXX check if this makes sense for negative powers?
      CHECK(e > 0);
      auto it = n.product.find(x);
      // If it's not present, we can exit early.
      if (it == n.product.end()) return std::make_pair(n, 0);
      int f = it->second;
      CHECK(f > 0);
      int q = f / e;
      if (q == 0) return std::make_pair(n, 0);
      if (!max_power.has_value() || max_power.value() > q) {
        max_power = {q};
      }
    }

    if (!max_power.has_value()) {
      // This means d is 1. The result is ambiguous so we choose 0.
      CHECK(d.product.empty()) << "bug";
      return std::make_pair(n, 0);
    }

    int p = max_power.value();
    CHECK(p > 0) << "bug";

    // Now we know that d^q divides the term. Compute the
    // remainder.
    Term rem;
    for (const auto &[x, e] : n.product) {
      auto it = d.product.find(x);
      if (it == d.product.end()) {
        // Unaffected term (as though f = 0).
        rem.product[x] = e;
      } else {
        int f = it->second;
        CHECK(f * p > 0) << f << " " << p;
        CHECK((e / (f * p)) > 0);
        int r = e % (f * p);
        if (r != 0) {
          rem.product[x] = r;
        }
      }
    }

    return std::make_pair(rem, p);
  }

  static Term Times(const Term &a, const Term &b) {
    // Same kind of thing as adding polynomials. Find common terms and
    // add their exponents.
    Term r;

    auto ait = a.product.begin();
    auto bit = b.product.begin();

    while (ait != a.product.end() && bit != b.product.end()) {
      if (ait->first == bit->first) {
        // Terms share a variable. Add the exponents. They might
        // completely cancel.
        int new_e = ait->second + bit->second;
        if (new_e != 0) {
          r.product[ait->first] = ait->second + bit->second;
        }
        ++ait;
        ++bit;
      } else if (ait->first < bit->first) {
        r.product[ait->first] = ait->second;
        ++ait;
      } else {
        r.product[bit->first] = bit->second;
        ++bit;
      }
    }

    // Add any leftovers.
    while (ait != a.product.end()) {
      r.product[ait->first] = ait->second;
      ++ait;
    }

    while (bit != b.product.end()) {
      r.product[bit->first] = bit->second;
      ++bit;
    }

    return r;
  }

  std::map<std::string, int> product;
};

static inline bool operator ==(const Term &a, const Term &b) {
  return a.product == b.product;
}

static inline bool operator <(const Term &a, const Term &b) {
  return a.product < b.product;
}

// Represents a rational polynomial. Open domain of variables, just
// represented as strings.
struct Polynomial {
  // zero polynomial
  Polynomial() {}

  explicit Polynomial(int a) {
    if (a != 0) {
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
    sum = {{t, 1}};
  }

  explicit Polynomial(const Term &t, int c) {
    if (c != 0) {
      sum = {{t, c}};
    }
  }

  static Polynomial SubstTerm(const Term &t, const Term &src,
                              const Polynomial &dst) {
    // we have eg. t = x^3y^2z^4 and src = xz^2. Then
    // we to factor this to xy^2 * (xz^2)^2. xy^2 is the "remainder"
    // and the outer ^2 on (xz^2)^2 is the power.
    const auto [rem, power] = Term::Divide(t, src);
    Polynomial out;
    // Could just be 1.
    out.sum = {{rem, 1}};

    return Times(out, Exp(dst, power));
  }

  static Polynomial Exp(const Polynomial &a, int n) {
    CHECK(n >= 0);
    // PERF can use repeated squaring.
    Polynomial ret{1};
    for (int i = 0; i < n; i++) ret = Times(ret, a);
    return ret;
  }

  // For a source term like "x^2" and a polynomial like "3yz",
  // replace all ocurrences of x^2 (including in higher powers like x^3)
  // with "3yz".
  static Polynomial Subst(const Polynomial &p,
                          const Term &src, const Polynomial &dst) {
    Polynomial out;
    for (const auto &[t, c] : p.sum) {
      Polynomial one = SubstTerm(t, src, dst);
      out = Plus(out, Scale(one, c));
    }
    return out;
  }

  std::string ToString() const {
    if (sum.empty()) return "0";
    // normally these would be sorted by descending power; we could
    // do that here?
    std::string ret;
    for (const auto &[t, c] : sum) {
      if (!ret.empty()) ret += " + ";
      if (c == 1) {
        std::string ts = t.ToString();
        // Term can be empty; need to explicitly represent the unit.
        if (ts.empty()) ret += "1";
        ret += ts;
      } else {
        StringAppendF(&ret, "%d%s", c, t.ToString().c_str());
      }
    }
    return ret;
  }

  static Polynomial PartialDerivative(const Polynomial &p,
                                      const std::string &x) {
    Polynomial r;
    for (const auto &[t, c] : p.sum) {
      auto it = t.product.find(x);
      if (it == t.product.end()) {
        // a term that doesn't depend on x; so it is dropped from
        // the derivative
      } else {
        int exponent = it->second;
        Term rt = t;
        // reduce its exponent
        if (exponent == 1) {
          rt.product.erase(x);
        } else {
          rt.product[x] = exponent - 1;
        }
        // e.g. 3x^7 yields (7 * 3)x^6
        r.sum[rt] = c * exponent;
      }
    }

    return r;
  }

  static Polynomial Scale(const Polynomial &p, int s) {
    Polynomial out;
    if (s != 0) {
      for (const auto &[t, c] : p.sum) {
        out.sum[t] = s * c;
      }
    }
    return out;
  }

  static Polynomial Times(const Polynomial &a, const Polynomial &b) {
    Polynomial r;

    for (const auto &[t1, c1] : a.sum) {
      for (const auto &[t2, c2] : b.sum) {
        Term rt = Term::Times(t1, t2);

        // PERF we could have an in-place addition of a single term.
        r = Plus(r, Polynomial(rt, c1 * c2));
      }
    }

    return r;
  }

  static Polynomial Plus(const Polynomial &a, const Polynomial &b) {
    Polynomial r;

    auto ait = a.sum.begin();
    auto bit = b.sum.begin();

    while (ait != a.sum.end() && bit != b.sum.end()) {
      if (ait->first == bit->first) {
        // Polynomials share a term; add the coefficients. They could
        // completely cancel.
        int coeff = ait->second + bit->second;
        if (coeff != 0) {
          r.sum[ait->first] = coeff;
        }
        ++ait;
        ++bit;
      } else if (ait->first < bit->first) {
        r.sum[ait->first] = ait->second;
        ++ait;
      } else {
        r.sum[bit->first] = bit->second;
        ++bit;
      }
    }

    // Add any leftovers.
    while (ait != a.sum.end()) {
      r.sum[ait->first] = ait->second;
      ++ait;
    }

    while (bit != b.sum.end()) {
      r.sum[bit->first] = bit->second;
      ++bit;
    }

    return r;
  }

  static bool Eq(const Polynomial &a, const Polynomial &b) {
    if (a.sum.size() != b.sum.size()) return false;
    for (const auto &[t, c] : a.sum) {
      auto it = b.sum.find(t);
      if (it == b.sum.end()) return false;
      if (it->second != c) return false;
    }
    return true;
  }

  // This belongs in polynomial-util or needs to be made more
  // general.
  static std::string ToCode(const std::string &type, const Polynomial &p);

  // Each term mapped to its integer coefficient. The coefficient
  // may not be zero.
  std::map<Term, int> sum;
};

inline Polynomial operator -(const Polynomial &p) {
  Polynomial np = p;
  for (auto &tc : np.sum)
    tc.second = -tc.second;
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

inline Polynomial operator *(const Polynomial &a, int s) {
  return Polynomial::Scale(a, s);
}

inline Polynomial operator *(int s, const Polynomial &a) {
  return Polynomial::Scale(a, s);
}

inline Polynomial operator -(const Polynomial &a, const Polynomial &b) {
  return a + -b;
}

inline Polynomial operator -(const Polynomial &a, int b) {
  return Polynomial::Plus(a, Polynomial(-b));
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
