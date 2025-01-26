#include "bignum/polynomial.h"

#include <string>
#include <map>
#include <cmath>
#include <set>
#include <vector>
#include <utility>

#include "base/logging.h"
#include "base/stringprintf.h"
#include "bignum/big-overloads.h"
#include "bignum/big.h"

using namespace std;

static string Join(const vector<string> &parts,
                   const string &sep) {
  if (parts.empty()) return "";
  if (parts.size() == 1) return parts[0];
  size_t result_len = 0;
  for (const string &part : parts)
    result_len += part.size();
  result_len += sep.size() * (parts.size() - 1);

  string out;
  out.reserve(result_len);
  out += parts[0];
  for (size_t i = 1; i < parts.size(); i++) {
    out += sep;
    out += parts[i];
  }
  return out;
}


// For e > 1; es must contain 1.
// Find (or insert) positive values e1, e2 in es, such that e1+e2 = e.
static std::pair<int, int> ClosePowers(std::set<int> *es,
                                       int e) {
  CHECK(e > 1);
  CHECK(es->find(1) != es->end()) << "Must contain the 1st power.";
  // i.e., the square root
  const int halfe = e >> 1;

  // This can obviously be more efficient!

  // This is the exponent closest to (and below) half
  // the power.
  int ecs = 0;
  for (const int e1 : *es) {
    // since we are going in ascending order.
    // The exactly-half case will be found by
    // the nested loop when e1=e2, and we need
    // not do any insertion in that case.
    if (e1 < halfe) ecs = e1;
    for (const int e2 : *es) {
      if (e1 + e2 == e) {
        return make_pair(e1, e2);
      }
    }
  }

  CHECK(ecs != 0) << "Impossible since set must contain 1.";
  int ecs_other = e - ecs;
  CHECK(es->find(ecs_other) == es->end()) << "Otherwise we would "
    "have found it in the loop above.";
  es->insert(ecs_other);
  return make_pair(ecs, ecs_other);
}

std::string Term::ToString() const {
  std::string ret;
  for (const auto &[x, e] : product) {
    if (e == 1) ret += x;
    else StringAppendF(&ret, "%s^%d", x.c_str(), e);
  }
  return ret;
}

std::pair<Term, int> Term::Divide(const Term &n, const Term &d) {
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


Term Term::Times(const Term &a, const Term &b) {
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


string Polynomial::ToCode(const string &type, const Polynomial &p) {
  // Collect all powers of variables >1 and <-1.
  // Note negative powers will just be in a divisor, so we
  // look at the absolute value of the exponent.
  std::map<string, std::set<int>> xpowers;
  for (const auto &[t, c_] : p.sum) {
    for (const auto &[x, signed_e] : t.product) {
      int e = abs(signed_e);
      if (e > 1) {
        xpowers[x].insert(e);
      }
    }
  }

  string code;

  auto PowVar = [](const std::string &x, int e) {
    CHECK(e > 0);
    if (e == 1) return x;
    return StringPrintf("%s_e%d", x.c_str(), e);
  };


  // Now generate them in a reasonably efficient way.
  for (auto &[x, es] : xpowers) {

    // Always include 1, since we already have that.
    es.insert(1);

    // Compute the powers from largest to smallest, so
    // that we know what prerequisites we also need to
    // generate.

    std::vector<string> rev_lines;
    while (!es.empty()) {
      const int e = *es.rbegin();
      es.erase(e);
      CHECK(e >= 1);
      if (e > 1) {
        const auto &[e1, e2] = ClosePowers(&es, e);
        rev_lines.push_back(
            StringPrintf("const %s %s = %s * %s;",
                         type.c_str(),
                         PowVar(x, e).c_str(),
                         PowVar(x, e1).c_str(),
                         PowVar(x, e2).c_str()));
      }
    }

    for (int i = rev_lines.size() - 1; i >= 0; i--) {
      StringAppendF(&code, "  %s\n", rev_lines[i].c_str());
    }
  }

  // PERF: We should factor out common terms (e.g. x^2y) and
  // compute them once.

  auto SummandVar = [](int n) {
    return StringPrintf("ps_%d", n);
  };

  StringAppendF(&code, "  // %d summands\n", p.sum.size());

  auto TermCode = [&PowVar](int64_t coeff, const Term &term) {
      CHECK(coeff != 0);
      // collect numerator, denominator
      std::vector<std::string> numer, denom;
      for (const auto &[x, e] : term.product) {
        CHECK(e != 0);
        if (e < 0) {
          denom.push_back(PowVar(x, -e));
        } else {
          numer.push_back(PowVar(x, e));
        }
      }

      string ns;
      if (numer.empty()) {
        ns = StringPrintf("%d", coeff);
      } else {
        ns = Join(numer, " * ");
        if (coeff != 1) {
          ns = StringPrintf("%d * %s", coeff, ns.c_str());
        }
      }

      if (denom.empty()) {
        return ns;
      } else {
        string ds = Join(denom, " * ");
        if (denom.size() != 1)
          ds = StringPrintf("(%s)", ds.c_str());
        return StringPrintf("(%s) / %s", ns.c_str(), ds.c_str());
      }

    };

  {
    int sidx = 0;
    for (const auto &[t, c] : p.sum) {
      const auto &[a, b] = c.Parts();
      const auto numero = a.ToInt();
      CHECK(numero.has_value() && b == BigInt(1)) << "For ToCode, the "
        "coefficients must be integers that fit in the radix. Got: " <<
        c.ToString();

      string tc = TermCode(numero.value(), t);
      string var = SummandVar(sidx);
      StringAppendF(&code, "  %s %s = %s;\n",
                    type.c_str(),
                    var.c_str(),
                    tc.c_str());
      sidx++;
    }
  }

  // And the final sum.

  StringAppendF(&code,
                "\n"
                "  return ");
  {
    for (int sidx = 0; sidx < (int)p.sum.size(); sidx++) {
      if (sidx != 0)
        StringAppendF(&code, " + ");
      StringAppendF(&code, "%s", SummandVar(sidx).c_str());
    }
  }

  StringAppendF(&code, ";\n");

  return code;
}

Polynomial Polynomial::Exp(const Polynomial &a, int n) {
  CHECK(n >= 0);
  // PERF can use repeated squaring.
  Polynomial ret{1};
  for (int i = 0; i < n; i++) ret = Times(ret, a);
  return ret;
}

Polynomial Polynomial::Plus(const Polynomial &a, const Polynomial &b) {
  Polynomial r;

  auto ait = a.sum.begin();
  auto bit = b.sum.begin();

  while (ait != a.sum.end() && bit != b.sum.end()) {
    if (ait->first == bit->first) {
      // Polynomials share a term; add the coefficients. They could
      // completely cancel.
      BigRat coeff = ait->second + bit->second;
      if (BigRat::Sign(coeff) != 0) {
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

Polynomial Polynomial::Scale(const Polynomial &p, const BigRat &s) {
  Polynomial out;
  if (BigRat::Sign(s) != 0) {
    for (const auto &[t, c] : p.sum) {
      out.sum[t] = s * c;
    }
  }
  return out;
}

Polynomial Polynomial::Times(const Polynomial &a, const Polynomial &b) {
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

Polynomial Polynomial::PartialDerivative(const Polynomial &p,
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
      r.sum[rt] = c * BigRat(exponent);
    }
  }

  return r;
}


bool Polynomial::Eq(const Polynomial &a, const Polynomial &b) {
  if (a.sum.size() != b.sum.size()) return false;
  for (const auto &[t, c] : a.sum) {
    auto it = b.sum.find(t);
    if (it == b.sum.end()) return false;
    if (it->second != c) return false;
  }
  return true;
}

Polynomial Polynomial::SubstTerm(const Term &t, const Term &src,
                                 const Polynomial &dst) {
  // we have eg. t = x^3y^2z^4 and src = xz^2. Then
  // we to factor this to xy^2 * (xz^2)^2. xy^2 is the "remainder"
  // and the outer ^2 on (xz^2)^2 is the power.
  const auto [rem, power] = Term::Divide(t, src);
  Polynomial out;
  // Could just be 1.
  out.sum = {{rem, BigInt(1)}};

  return Times(out, Exp(dst, power));
}

Polynomial Polynomial::Subst(const Polynomial &p, const Term &src,
                             const Polynomial &dst) {
  Polynomial out;
  for (const auto &[t, c] : p.sum) {
    Polynomial one = SubstTerm(t, src, dst);
    out = Plus(out, Scale(one, c));
  }
  return out;
}

std::string Polynomial::ToString() const {
  if (sum.empty()) return "0";
  // normally these would be sorted by descending power; we could
  // do that here?
  std::string ret;
  for (const auto &[t, c] : sum) {
    if (!ret.empty()) ret += " + ";
    if (c == BigRat(1)) {
      std::string ts = t.ToString();
      // Term can be empty; need to explicitly represent the unit.
      if (ts.empty()) ret += "1";
      ret += ts;
    } else {
      StringAppendF(&ret, "%s%s", c.ToString().c_str(),
                    t.ToString().c_str());
    }
  }
  return ret;
}
