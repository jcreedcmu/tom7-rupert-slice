
#include "polynomial.h"

#include <string>

#include "base/logging.h"

#define CHECK_SEQ(a, b) do {                            \
    const std::string aa = (a), bb = (b);               \
    CHECK(aa == bb) << "Comparing " #a " and " #b "\n"  \
                    << aa << "\nvs\n" << bb;            \
  } while (false)

#define CHECK_PEQ(a, b) do {                              \
    const Polynomial aa = (a), bb = (b);                  \
    CHECK(aa == bb) << "Comparing " #a " and " #b "\n"    \
                    << aa.ToString() << "\nvs\n"          \
                    << bb.ToString();                     \
  } while (false)

static void Simple() {
  Polynomial seven = 7_p;
  Polynomial x = "x"_p;

  {
    Polynomial p = x + seven;
    CHECK_SEQ(p.ToString(), "7 + x");
    p = p + p;
    CHECK_SEQ(p.ToString(), "14 + 2x");
  }

  {
    Polynomial p = x * x + seven * x;
    CHECK_SEQ(p.ToString(), "7x + x^2");

    Polynomial pp = Polynomial::PartialDerivative(p, "x");
    CHECK_SEQ(pp.ToString(), "7 + 2x");
  }

  {
    Polynomial p = x * x * x + seven;
    Polynomial pp = Polynomial::PartialDerivative(p, "x");
    CHECK_SEQ(pp.ToString(), "3x^2");

    // invariant in y, so the derivative is zero
    Polynomial ppy = Polynomial::PartialDerivative(p, "y");
    CHECK_SEQ(ppy.ToString(), "0");
  }
}

static void Canceling() {
  {
    Polynomial k("k", 3);
    Polynomial invk("k", -3);

    Polynomial one = k * invk;
    CHECK_SEQ(one.ToString(), "1");
  }

  {
    Polynomial m("m", 2);
    Polynomial negm = -m;

    Polynomial zero = m + negm;
    CHECK_SEQ(zero.ToString(), "0");
  }
}

static void DivideTerm() {
  Term x = "x"_t;
  Term y = "y"_t;
  Term z = "z"_t;

  {
    const auto &[rem, pow] = Term::Divide(x, x);
    CHECK(rem == Term());
    CHECK(pow == 1);
  }

  {
    const auto &[rem, pow] = Term::Divide(x * y, x);
    CHECK(rem == y);
    CHECK(pow == 1);
  }
}


static void Subst() {
  Polynomial x("x");
  Polynomial y("y");
  Polynomial z("z");

  Polynomial p1 = Polynomial::Subst(x, "x"_t, y);
  CHECK(p1 == y);

  Polynomial p2 = 2 * x * x * x + 8 * x * x * y - 7 * y * y;
  Polynomial p2s = Polynomial::Subst(p2, "x"_t * "x"_t, z * y);

  CHECK(p2s == 2 * x * z * y + 8 * y * y * z - 7 * y * y) << p2s.ToString();
}

static void ZeroPower() {
  Polynomial p = "s"_p;
  Polynomial q = "q"_p;
  Polynomial r = "r"_p;
  Polynomial s = "s"_p;

  Polynomial b = "b"_p;
  Polynomial c = "c"_p;

  Polynomial t1 = s * b + r * c;
  Polynomial sp("s", 0);
  Polynomial t2 = s * c + q * b;
  Polynomial qp("q", 0);

  Polynomial res = t1 * sp + t2 * qp;
  CHECK_PEQ(res, t1 + t2);
}

int main(int argc, char **argv) {

  Simple();
  Canceling();
  DivideTerm();
  Subst();

  ZeroPower();

  printf("OK\n");
  return 0;
}
