#include "polynomial.h"

#include <string>
#include <map>
#include <cmath>
#include <set>
#include <vector>
#include <utility>

#include "base/logging.h"

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

  auto TermCode = [&PowVar](int coeff, const Term &term) {
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
      string tc = TermCode(c, t);
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
    for (int sidx = 0; sidx < p.sum.size(); sidx++) {
      if (sidx != 0)
        StringAppendF(&code, " + ");
      StringAppendF(&code, "%s", SummandVar(sidx).c_str());
    }
  }

  StringAppendF(&code, ";\n");

  return code;
}
