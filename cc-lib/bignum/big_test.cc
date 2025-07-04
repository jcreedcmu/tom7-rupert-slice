
#include "bignum/big.h"
#include <numbers>
#ifndef BIG_USE_GMP
#include "bignum/bign.h"
#endif

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>
#include <unordered_set>

#include "base/logging.h"
#include "base/stringprintf.h"
#include "timer.h"

using int64 = int64_t;
using namespace std;

static constexpr bool RUN_BENCHMARKS = false;

#define CHECK_SEQ(a, b) do {                                            \
    auto aa = (a);                                                      \
    auto bb = (b);                                                      \
    CHECK(aa == bb) << "Expected equal strings:\n" << #a << "\n" << #b  \
                    << "\nValues:\n" << aa << "\n" << bb << "\n";       \
  } while (false)

static void TestToString() {
  for (int i = -100000; i < 100000; i++) {
    BigInt bi(i);
    string s = bi.ToString();
    CHECK(!s.empty());
    CHECK(BigInt::Eq(BigInt(s), bi));
    // We do some resizing tricks in the GMP version, which
    // can inadvertently leave nul bytes.
    for (char c : s) CHECK(c != 0) << i;
  }
}

static void TestSign() {
  CHECK(BigInt::Sign(BigInt(-3)) == -1);
  CHECK(BigInt::Sign(BigInt(0)) == 0);
  CHECK(BigInt::Sign(BigInt("19823749283749817")) == 1);
  CHECK(BigInt::Sign(BigInt("-999999999999999999999")) == -1);
}

static void CopyAndAssign() {
  BigInt a{1234};
  BigInt b{5678};
  BigInt c = a;
  a = b;
  CHECK(BigInt::Eq(a, b));
  CHECK(!BigInt::Eq(a, c));
  {
    BigInt z{8888};
    a = z;
  }
  CHECK(a.ToInt() == 8888);

  string s = "190237849028374901872390876190872349817230948719023874190827349817239048712903847190283740918273490817230948798767676767676738347482712341";
  BigInt big(s);
  CHECK(big.ToString() == s);
  big = a;
  CHECK(big.ToInt() == 8888);
  CHECK(big.ToU64() == 8888u);
}

static void TestToU64() {

  {
    auto uo = BigInt(0).ToU64();
    CHECK(uo.has_value() && uo.value() == 0);
  }

  CHECK(!BigInt(-1).ToU64().has_value());

  {
    auto uo =
      BigInt::Minus(
          BigInt::LeftShift(BigInt(1), 64),
          1).ToU64();
    CHECK(uo.has_value() && uo.value() == (uint64_t)-1);
  }

}

static void HashCode() {
  BigInt a{1234};
  BigInt b{5678};
  // Technically this can be anything, but it would probably be a bug if
  // small integers don't get different values.
  CHECK(BigInt::HashCode(a) != BigInt::HashCode(b));
  CHECK(BigInt::HashCode(a) == BigInt::HashCode(a));

  string s = "190237849028374901872390876190872349817230948719023874190827349817239048712903847190283740918273490817230948798767676767676738347482712341";
  BigInt big(s);
  CHECK(BigInt::HashCode(a) != BigInt::HashCode(big));
  CHECK(BigInt::HashCode(big) == BigInt::HashCode(big));
}

static void TestRatString() {
  {
    BigRat a("123");
    CHECK_SEQ(a.ToString(), "123");
  }

  {
    BigRat b("-4");
    CHECK_SEQ(b.ToString(), "-4");
  }

  {
    BigRat c("41/17");
    CHECK_SEQ(c.ToString(), "41/17");
  }

  {
    BigRat d("-41/17");
    CHECK_SEQ(d.ToString(), "-41/17");
  }
}

static void TestRatFromDouble() {
  // XXX test this more!
  {
    BigRat z = BigRat::FromDouble(0.0);
    std::string s = z.ToString();
    CHECK(s == "0") << s;
  }

  {
    BigRat one_half = BigRat::FromDouble(0.5);
    std::string s = one_half.ToString();
    CHECK(s == "1/2") << s;
  }

  {
    double ud = 0.4999999999999431;
    CHECK(ud < 0.5);
    BigRat u = BigRat::FromDouble(ud);
    printf("%s\n", u.ToString().c_str());
    CHECK(BigRat::Less(u, BigRat(1, 2))) << u.ToString();
  }

  {
    BigRat r = BigRat::FromDouble(65536);
    std::string s = r.ToString();
    CHECK(s == "65536") << s;
  }

  {
    BigRat r = BigRat::FromDouble(65537);
    std::string s = r.ToString();
    CHECK(s == "65537") << s;
  }

  {
    BigRat r = BigRat::FromDouble(-65535);
    std::string s = r.ToString();
    CHECK(s == "-65535") << s;
  }

  {
    BigRat r = BigRat::FromDouble(1.0 / -65536.0);
    std::string s = r.ToString();
    CHECK(s == "-1/65536") << s;
  }

  BigRat huge = BigRat::FromDouble(1.0e300);
  CHECK(BigRat::Greater(huge, BigRat(0)));
  BigRat almost_huge = BigRat::FromDouble(1.0e299);
  CHECK(BigRat::Greater(huge, almost_huge));

  // Note: This number is not exactly representable as a double
  // (not just 1/3, but 0.3333333). GMP gives slightly different
  // approximation, which I think is more correct.
  BigRat thirdish = BigRat::FromDouble(0.3333333);
  double t = thirdish.ToDouble();
  CHECK(std::abs(t - 0.3333333) < 0.0000001) << thirdish.ToString();

  printf("Rat FromDouble: OK\n");
}

static void TestRatToDouble() {
  {
    BigRat one_half(1, 2);
    double h = one_half.ToDouble();
    CHECK(std::abs(h - 0.5) < 1e-30) << h;
    printf("OK: %.17g\n", h);
  }

  {
    BigRat about_one_half(BigInt("100000000001"),
                          BigInt("200000000000"));
    double expected = 100000000001.0 / 200000000000.0;
    double actual = about_one_half.ToDouble();
    CHECK(std::abs(actual - expected) < 1e-10) <<
      std::format("Expected: {:.17g}\n"
                   "Actual: {:.17g}\n", expected, actual);
  }

  {
    BigRat about_one_half(
        BigInt("10000000000000000000000000000000000000000001"),
        BigInt("20000000000000000000000000000000000000000000"));
    CHECK(std::abs(about_one_half.ToDouble() - 0.5) < 1e-20);
  }
}

static void TestToDouble() {
  BigInt a{1234};
  BigInt b{-5678};

  CHECK((BigInt(0)).ToDouble() == 0.0);
  CHECK(a.ToDouble() == 1234.0);
  CHECK(b.ToDouble() == -5678.0);

  {
    BigInt big("7000000000000000000000000000"
               "000000000000000000000000000000000");
    double d = big.ToDouble();
    CHECK(d > 0.0);
    CHECK(std::isfinite(d));
    CHECK(d > 7e59 && d < 7e61) << std::format("{:.11g}\n", d);
  }

  {
    BigInt big("-7000000000000000000000000000"
               "000000000000000000000000000000000");
    double d = big.ToDouble();
    CHECK(d < 0.0);
    CHECK(std::isfinite(d));
    CHECK(d < -7e59 && d > -7e61) << std::format("{:.11g}\n", d);
  }

  // TODO: Test infinite cases
}

static void TestMod() {
  CHECK(BigInt::Eq(BigInt::Mod(BigInt{3}, BigInt{5}), BigInt{3}));
  CHECK(BigInt::Eq(BigInt::Mod(BigInt{7}, BigInt{5}), BigInt{2}));
  CHECK(BigInt::Eq(BigInt::Mod(BigInt{10}, BigInt{5}), BigInt{0}));
  CHECK(BigInt::Eq(BigInt::Mod(BigInt{-1}, BigInt{5}), BigInt{4}));
}

static void TestEq() {
  BigInt a{1234};
  BigInt b{5678};

  CHECK(BigInt::Eq(BigInt::Times(a, b), 7006652));
}

static void TestPow() {
  BigRat q(11,15);

  BigRat qqq = BigRat::Times(q, BigRat::Times(q, q));
  BigRat qcubed = BigRat::Pow(q, 3);
  printf("%s vs %s\n", qqq.ToString().c_str(),
         qcubed.ToString().c_str());
  CHECK(BigRat::Eq(qqq, qcubed));
}

// Sloooow
static BigInt ReferencePowMod(const BigInt &base, const BigInt &exp,
                              const BigInt &mod) {
  auto eo = exp.ToInt();
  CHECK(eo.has_value());
  const int64_t e = eo.value();
  BigInt res(1);

  for (int i = 0; i < e; i++) {
    res = BigInt::CMod(BigInt::Times(res, base), mod);
  }

  return res;
}

static void TestPowMod() {
  for (const std::string &aa : {"0", "1", "2", "131", "1751", "31337"}) {
    BigInt a(aa);
    for (const std::string &bb : {"0", "1", "2", "17", "1331", "101017"}) {
      BigInt b(bb);

      for (const std::string &mm : {"2", "3", "5", "8", "65537"}) {
        BigInt m(mm);

        BigInt r = BigInt::PowMod(a, b, m);
        BigInt rr = ReferencePowMod(a, b, m);
        CHECK_SEQ(r.ToString(), rr.ToString());
      }
    }
  }
}

// TODO: Test/document behavior on negative inputs
static void TestQuotRem() {
  BigInt a(37);
  BigInt b(5);

  const auto [q, r] = BigInt::QuotRem(a, b);
  CHECK(BigRat::Eq(q, BigInt(7)));
  CHECK(BigRat::Eq(r, BigInt(2)));
}

static void TestPrimeFactors() {
  printf("Prime factors..\n");
  auto FTOS = [](const std::vector<std::pair<BigInt, int>> &fs) {
      string s;
      for (const auto &[b, i] : fs) {
        AppendFormat(&s, "{}^{} ", b.ToString(), i);
      }
      return s;
    };

  BigInt bi31337(31337);
  {
    std::vector<std::pair<BigInt, int>> factors =
      BigInt::PrimeFactorization(bi31337);

    CHECK(factors.size() == 1);
    CHECK(factors[0].second == 1);
    CHECK(BigInt::Eq(factors[0].first, bi31337));
  }

  {
    BigInt x(31337 * 71);
    std::vector<std::pair<BigInt, int>> factors =
      BigInt::PrimeFactorization(x);

    CHECK(factors.size() == 2) << FTOS(factors);
    CHECK(BigInt::Eq(factors[0].first, BigInt(71)));
    CHECK(factors[0].second == 1) << factors[0].second;
    CHECK(BigInt::Eq(factors[1].first, bi31337));
    CHECK(factors[1].second == 1) << factors[0].second;
  }

  {
    BigInt bi31337sq(31337 * 31337);
    std::vector<std::pair<BigInt, int>> factors =
      BigInt::PrimeFactorization(bi31337sq);

    CHECK(factors.size() == 1) << FTOS(factors);
    CHECK(BigInt::Eq(factors[0].first, bi31337));
    CHECK(factors[0].second == 2) << factors[0].second;
  }

  {
    // Largest 64-bit prime.
    BigInt p("18446744073709551557");
    std::vector<std::pair<BigInt, int>> factors =
      BigInt::PrimeFactorization(p);

    CHECK(BigInt::IsPrime(p));

    CHECK(factors.size() == 1) << FTOS(factors);
    CHECK(BigInt::Eq(factors[0].first, p)) << factors[0].first.ToString();
    CHECK(factors[0].second == 1) << factors[0].second;
  }

  {
    BigInt x("11682658198262752314377738154934272");
    std::vector<std::pair<BigInt, int>> factors =
      BigInt::PrimeFactorization(x);

    CHECK(factors.size() == 3) << FTOS(factors);
    CHECK(BigInt::Eq(factors[0].first, 2));
    CHECK(factors[0].second == 18);

    CHECK(BigInt::Eq(factors[1].first, 3));
    CHECK(factors[1].second == 1);

    CHECK(BigInt::Eq(factors[2].first,
                     BigInt("14855268094714803459647799371")));
    CHECK(factors[2].second == 1);
  }

  {
    BigInt x(1);
    // Must all be distinct and prime
    const array f = {
      2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 37, 41, 43, 47, 419,
      541, 547, 31337,
    };
    for (int factor : f) {
      x = BigInt::Times(x, BigInt(factor));
    }

    std::vector<std::pair<BigInt, int>> factors =
      BigInt::PrimeFactorization(x);

    CHECK(factors.size() == f.size());
    for (int i = 0; i < (int)f.size(); i++) {
      CHECK(factors[i].second == 1);
      CHECK(BigInt::IsPrime(factors[i].first));
      CHECK(BigInt::Eq(factors[i].first, BigInt(f[i])));
    }
  }

  CHECK(BigInt::IsPrime(BigInt("2847516444895613959067741")));
  CHECK(!BigInt::IsPrime(BigInt("934765193746581762467777777777777777777777")));
  CHECK(!BigInt::IsPrime(BigInt(
                             "934765193746581762467777777777777777777777"
                             "99999999999999999999999999")));

  // This is 7907^8, bigger than 2^64 and only with factors in the
  // trial division list.
  CHECK(!BigInt::IsPrime(BigInt("15278964097073166593290816202401")));

  CHECK(BigInt::IsPrime(BigInt("351287497542881756276856991")));
  CHECK(!BigInt::IsPrime(BigInt("351287497542881756276856997")));

  /*
    too hard for current implementation

    CHECK(BigInt::IsPrime(BigInt("36810898113683429059692379139738721323647376409491014157")));

    CHECK(BigInt::IsPrime(BigInt("2871856218107427379960936944425882732846010327626613179095159878800893591143935691173601490558081638031076860171")));

    CHECK(BigInt::IsPrime(BigInt("23778489029892836999435984213197969543147208"
                               "12182515510434297800338407156682609569285213"
                               "24829276602215887604926261909612695238233550"
                               "161061361")));
  */

  // Primorials
  {
    BigInt prim(1);
    for (uint64_t n = 2; n <= 128; n++) {
      if (!BigInt::IsPrime(BigInt(n)))
        continue;

      prim = BigInt::Times(prim, n);

      std::vector<std::pair<BigInt, int>> factors =
        BigInt::PrimeFactorization(prim);

      BigInt product(1);
      for (const auto &[p, e] : factors) {
        CHECK(e == 1) << n;
        for (int r = 0; r < e; r++) {
          product = BigInt::Times(product, p);
        }
      }

      CHECK(BigInt::Eq(product, prim)) << n;
    }
  }

  printf("Prime factorization OK.\n");
}


static void BenchDiv2() {
  double total_sec = 0.0;
  const int iters = 100;
  const BigInt two(2);
  for (int i = 0; i < iters; i++) {
    BigInt x = BigInt::Pow(two, 30000);
    Timer t;

    for (;;) {
      const auto [q, r] = BigInt::QuotRem(x, two);
      if (BigInt::Eq(r, 0)) {
        x = q;
      } else {
        break;
      }
    }

    CHECK(BigInt::Eq(x, 1));

    total_sec += t.Seconds();
    if (i % 10 == 0) {
      printf("%d/%d\n", i, iters);
    }
  }

  printf("%d iters in %.5fs = %.3f/s\n",
         iters, total_sec, iters / total_sec);
}

static void TestPi() {
  printf("----\n");
  {
    BigInt i{int64_t{1234567}};
    BigInt j{int64_t{33}};
    BigInt k = BigInt::Times(i, j);
    BigInt m("102030405060708090987654321");

    printf("Integer: %s %s %s\n%s\n",
           i.ToString().c_str(),
           j.ToString().c_str(),
           k.ToString().c_str(),
           m.ToString().c_str());
    fflush(stdout);
  }

  BigRat sum;
  #if BIG_USE_GMP
  static constexpr int LIMIT = 10000;
  #else
  static constexpr int LIMIT = 1000;
  #endif

  for (int i = 0; i < LIMIT; i++) {
    // + 1/1, - 1/3, + 1/5
    BigRat term{(i & 1) ? -1 : 1,
        i * 2 + 1};
    sum = BigRat::Plus(sum, term);
    if (i < 50) {
      BigRat tpi = BigRat::Times(sum, BigRat{4,1});
      printf("Approx pi: %s = %f\n",
             tpi.ToString().c_str(),
             tpi.ToDouble());
      fflush(stdout);
    } else if (i % 1000 == 0) {
      printf("%d...\n", i);
      fflush(stdout);
    }
  }

  BigRat res = BigRat::Times(sum, BigRat(4, 1));
  printf("Final approx pi: %s\n",
         res.ToString().c_str());
  fflush(stdout);


  // This sequence converges REALLY slow!
  BigRat pi_lb(314, 100);
  BigRat pi_ub(315, 100);

  CHECK(BigRat::Compare(pi_lb, pi_ub) == -1);
  CHECK(BigRat::Compare(pi_lb, res) == -1);
  CHECK(BigRat::Compare(res, pi_ub) == -1);
}

static void TestLeadingZero() {
  // TODO: Test this through BigNum interface. Ideally
  // the test should not care about the implementation.
#ifndef BIG_USE_GMP
  CHECK(BnnNumLeadingZeroBitsInDigit(1ULL) == 63);
  CHECK(BnnNumLeadingZeroBitsInDigit(0b1000ULL) == 60);
  CHECK(BnnNumLeadingZeroBitsInDigit(~0ULL) == 0);
  CHECK(BnnNumLeadingZeroBitsInDigit(0ULL) == 64);

  CHECK(std::countl_zero<BigNumDigit>(1ULL) == 63);
  CHECK(std::countl_zero<BigNumDigit>(0b1000ULL) == 60);
  CHECK(std::countl_zero<BigNumDigit>(~0ULL) == 0);
  CHECK(std::countl_zero<BigNumDigit>(0ULL) == 64);
#endif
}

static void TestToInt() {
# define ROUNDTRIP(x) do {                                              \
  BigInt bi((int64_t)(x));                                              \
  std::optional<int64_t> io = bi.ToInt();                               \
  CHECK(io.has_value()) << std::format("{:x}", x) << "("                \
                        << bi.ToString(16) << ")";                      \
  CHECK((x) == io.value()) << std::format("{:x}", x) << " vs "          \
                           << bi.ToString(16);                          \
} while (0)

  ROUNDTRIP(0);
  ROUNDTRIP(1);
  ROUNDTRIP(-1);
  ROUNDTRIP(0x7FFFFFFEL);
  ROUNDTRIP(0x7FFFFFFFL);
  ROUNDTRIP(0x80000000LL);
  ROUNDTRIP(0x80000000LL);
  ROUNDTRIP(0x7FFFFFFFFFFFFFFELL);
  ROUNDTRIP(0x7FFFFFFFFFFFFFFFLL);

# define NOROUNDTRIP(bi) do {                                     \
    std::optional<int64_t> io = (bi).ToInt();                     \
    CHECK(!io.has_value()) << #bi << " =\n" << \
      bi.ToString() << " " << io.value();     \
  } while (false)
  NOROUNDTRIP(BigInt::Plus(BigInt(int64_t{0x7FFFFFFFFFFFFFFF}),
                           BigInt(1)));
  NOROUNDTRIP(BigInt::Minus(
                  BigInt::Negate(BigInt(int64_t{0x7FFFFFFFFFFFFFFF})),
                  BigInt(1)));
  NOROUNDTRIP(BigInt::Times(BigInt(int64_t{0x7FFFFFFFFFFFFFFF}),
                            BigInt(10000)));

# undef ROUNDTRIP
# undef NOROUNDTRIP
}

static uint64_t Sqrt64Nuprl(uint64_t xx) {
  // printf("SqrtNuprl(%llu)\n", xx);
  if (xx <= 1) return xx;
  // z = xx / 4
  uint64_t z = xx >> 2;
  uint64_t r2 = 2 * Sqrt64Nuprl(z);
  uint64_t r3 = r2 + 1;
  // printf("r2 = %llu, r3 = %llu\n", r2, r3);
  return (xx < r3 * r3) ? r2 : r3;
}

static void TestSqrt() {
  for (const uint64_t u : std::initializer_list<uint64_t>{
        1234567, 0x7FFFFFFFFFFFFFFFULL, 121,
        0, 1, 2, 3, 4, 5, 6, 9999999999999ULL, 31337 * 31337ULL}) {
    BigInt ub(u);

    // Sqare root of squares should be equal
    BigInt uu = BigInt::Times(ub, ub);
    CHECK(BigInt::Eq(BigInt::Sqrt(uu), ub)) << u;

    {
      const auto [v, vrem] = BigInt::SqrtRem(uu);
      CHECK(BigInt::Eq(v, ub));
      CHECK(BigInt::Eq(vrem, BigInt{0}));
    }

    uint64_t us64 = Sqrt64Nuprl(u);
    BigInt usb = BigInt::Sqrt(ub);
    CHECK(BigInt::Eq(usb, BigInt(us64))) << u << " " << us64;
    const auto [vsb, vrem] = BigInt::SqrtRem(ub);
    CHECK(BigInt::Eq(vsb, BigInt(us64)));
    BigInt back = BigInt::Plus(BigInt::Times(vsb, vsb), vrem);
    CHECK(BigInt::Eq(back, ub));
  }
}

static void TestGCD() {
  BigInt g = BigInt::GCD(BigInt(8), BigInt(12));
  CHECK(BigInt::Eq(g, BigInt(4)));
}

static void TestLog() {
#if BIG_USE_GMP
  BigInt x = BigInt::LeftShift(BigInt(1), 301);
  double y = BigInt::LogBase2(x);
  CHECK(y >= 300.99 && y <= 301.01) << y;

  double z = BigInt::NaturalLog(x);
  CHECK(z >= 208.6 && z <= 208.7) << z;
#else
  printf("Warning: LogBase2 not implemented\n");
#endif
}

static void TestShift() {
  {
    BigInt a{"12398471982735675717171221"};
    BigInt b = BigInt::LeftShift(a, 18);
    BigInt c = BigInt::Times(a, BigInt{262144});
    CHECK(BigInt::Eq(b, c));
  }

  {
    BigInt a{"1293472907097860173485720741"};
    BigInt c = BigInt::RightShift(a, 51);
    CHECK(BigInt::Eq(c, BigInt("574417361275"))) << c.ToString();
  }
}

static void TestAnd() {
  {
    BigInt a{"19827348723"};
    BigInt b{"123908472983749187767123045885812"};
    BigInt c = BigInt::BitwiseAnd(a, b);
    CHECK(BigInt::Eq(c, BigInt("227083376"))) << c.ToString();
  }

  {
    BigInt a{"-19827348723"};
    BigInt b{"123908472983749187767123045885812"};
    BigInt c = BigInt::BitwiseAnd(a, b);
    CHECK(BigInt::Eq(c, BigInt("123908472983749187767122818802436")))
      << c.ToString();
  }

  {
    BigInt a("11122233344487293847298734827");
    uint64_t b = 0xFEFFFFFFFFFEDCBAULL;
    uint64_t c = BigInt::BitwiseAnd(a, b);
    CHECK(c == 0xbecb0d895d7858aa) << c;
  }

  {
    BigInt z(0);
    CHECK(0 == BigInt::BitwiseAnd(z, 1234));
  }

}

static void TestXor() {
  CHECK(BigInt::Eq(BigInt{0},
                   BigInt::BitwiseXor(BigInt{12345}, BigInt{12345})));
  CHECK(BigInt::Eq(BigInt{5},
                   BigInt::BitwiseXor(BigInt{4}, BigInt{1})));
}

static void TestOr() {
  CHECK(BigInt::Eq(BigInt{12345},
                   BigInt::BitwiseOr(BigInt{12345}, BigInt{12345})));
  CHECK(BigInt::Eq(BigInt{5}, BigInt::BitwiseOr(BigInt{4}, BigInt{1})));
  CHECK(BigInt::Eq(BigInt{5}, BigInt::BitwiseOr(BigInt{5}, BigInt{1})));
}

static void TestDivExact() {
  BigInt a{"23984727341"};
  BigInt b{"12737177354116809923874293874113"};
  BigInt c = BigInt::Times(a, b);
  BigInt d = BigInt::DivExact(c, a);
  BigInt e = BigInt::DivExact(c, b);
  CHECK(BigInt::Eq(b, d));
  CHECK(BigInt::Eq(a, e));
}

static void TestDivFloor() {
  // No rounding.
  CHECK(BigInt::Eq(
            BigInt::DivFloor(BigInt(-4), BigInt(-2)),
            2));
  CHECK(BigInt::Eq(
            BigInt::DivFloor(BigInt(40), BigInt(-20)),
            -2));
  CHECK(BigInt::Eq(
            BigInt::DivFloor(BigInt(-400), 200),
            -2));
  CHECK(BigInt::Eq(
            BigInt::DivFloor(BigInt(-4000), -2000),
            2));

  // Round towards negative infinity.
  // This is the same for positive numbers.
  CHECK(BigInt::Eq(
            BigInt::Div(BigInt(1234567), 175),
            BigInt::DivFloor(BigInt(1234567), 175)));

  CHECK(BigInt::Eq(
            BigInt::DivFloor(BigInt(10), -3),
            -4));

  CHECK(BigInt::Eq(
            BigInt::DivFloor(BigInt(-10001),
                             BigInt(10)),
            -1001));
}

static void TestJacobi() {
  #if BIG_USE_GMP
  // XXX It's broken outside of GMP?

  CHECK(BigInt::Jacobi(BigInt(11), BigInt(17)) == -1);
  CHECK(BigInt::Jacobi(BigInt(1), BigInt(1)) == 1);
  CHECK(BigInt::Jacobi(BigInt(6), BigInt(15)) == 0);

  CHECK(BigInt::Jacobi(BigInt(30), BigInt(59)) == -1);
  #endif
}

static void TestModRem() {
  BigInt neg1{-1};
  BigInt four{4};

  CHECK(BigInt::Eq(BigInt::Mod(neg1, four), 3));
  const auto [q, r] = BigInt::QuotRem(neg1, four);
  CHECK(BigInt::Eq(q, 0)) << q.ToString();
  CHECK(BigInt::Eq(r, -1)) << r.ToString();
}

static void TestDiv() {
  // Check that Div has the same rounding behavior as C++11.
  for (int x = -14; x < 7; x++) {
    for (int y = -7; y < 14; y++) {
      if (y != 0) {
        int z = x / y;
        BigInt bz = BigInt::Div(BigInt(x), BigInt(y));
        BigInt bq = BigInt::QuotRem(BigInt(x), BigInt(y)).first;

        CHECK(BigInt::Eq(bz, z)) <<
          std::format("{} / {} = {} (got {})\n",
                       x, y, z, bz.ToString());

        CHECK(BigInt::Eq(bq, z)) <<
          std::format("{} / {} = {} (got {})\n",
                      x, y, z, bq.ToString());
      }
    }
  }
}

static void TestCMod() {
  // Check that CMod has the same rounding behavior as C++11.
  for (int x = -14; x < 7; x++) {
    for (int y = -7; y < 14; y++) {
      if (y != 0) {
        int z = x % y;
        BigInt bz = BigInt::CMod(BigInt(x), BigInt(y));
        BigInt br = BigInt::QuotRem(BigInt(x), BigInt(y)).second;

        CHECK(BigInt::Eq(bz, z)) <<
          std::format("{} % {} = {} (got {})\n",
                       x, y, z, bz.ToString());
        CHECK(BigInt::Eq(br, z)) <<
          std::format("{} % {} = {} (got {})\n",
                      x, y, z, br.ToString());

        // Also check int64 version.
        int64_t ibz = BigInt::CMod(BigInt(x), (int64_t)y);
        CHECK(BigInt::Eq(bz, ibz));
      }
    }
  }
}

#define CHECK_HAS_EQ(aopt, b) do {                        \
    std::optional<BigInt> aopt_ = (aopt);                 \
    BigInt b_ = (b);                                      \
    CHECK(aopt_.has_value()) << #aopt << " vs " << #b;    \
    CHECK(BigInt::Eq(aopt_.value(), b_)) <<               \
      #aopt << " vs " << #b << "\n"                       \
      "Got: " << aopt_.value().ToString() << "\n"         \
      "Want: " << b_.ToString();                          \
  } while (0)

static void TestInvert() {
  for (const char *bs : {
      "1", "2", "3", "4", "5", "100", "10001", "31337",
      // 2^64
      "18446744073709551616",
      // 2^64 - 1
      "18446744073709551615",
      // 2^64 + 1
      "18446744073709551617",
      // 2^127
      "170141183460469231731687303715884105728"}) {
    const BigInt modulus(bs);

    if (BigInt::Eq(modulus, 1)) {
      // Degenerate case.
      CHECK_HAS_EQ(BigInt::ModInverse(BigInt(1), modulus),
                   BigInt(0));
    } else {
      // Otherwise, modular inverse of 1 always exists and is 1.
      CHECK_HAS_EQ(BigInt::ModInverse(BigInt(1), modulus),
                   BigInt(1));
    }

    if (BigInt::Less(modulus, BigInt(100000)) &&
        BigInt::IsPrime(modulus)) {

      for (BigInt a(1); BigInt::Less(a, modulus); a = BigInt::Plus(a, 1)) {
        auto ao = BigInt::ModInverse(a, modulus);
        CHECK(ao.has_value()) << "Every nonzero value has an inverse "
          "mod a prime p: " << a.ToString() << "^-1 mod " <<
          modulus.ToString();
        BigInt inv = ao.value();
        CHECK(BigInt::Greater(inv, 0) &&
              BigInt::Less(inv, modulus)) << inv.ToString();
        BigInt product = BigInt::CMod(BigInt::Times(inv, a), modulus);
        CHECK(BigInt::Eq(product, 1)) << product.ToString();
      }

    } else {

      for (const char *as : {
          "2", "3", "5", "6", "11", "27", "51",
          "120", "121",
          "15232", "90210",
          // 2^64 - 1
          "18446744073709551615"}) {
        const BigInt a(as);

        if (BigInt::Less(a, modulus)) {
          std::optional<BigInt> ao = BigInt::ModInverse(a, modulus);
          if (ao.has_value()) {
            BigInt inv = ao.value();
            CHECK(BigInt::Greater(inv, 0) &&
                  BigInt::Less(inv, modulus)) << inv.ToString();
            BigInt product = BigInt::CMod(BigInt::Times(inv, a), modulus);
            CHECK(BigInt::Eq(product, 1)) << product.ToString();
          }
        }
      }
    }
  }

  printf("Modular inverse OK.\n");
}

static void TestSwap() {
  BigInt a("11223344556677889900");
  BigInt b("55555555555555555555");

  std::swap(a, b);
  CHECK(a.ToString() == "55555555555555555555");
  CHECK(b.ToString() == "11223344556677889900");
}


static void TestRatSwap() {
  BigRat a(BigInt("11111111111111111117"), BigInt("555555555555555555555"));
  BigRat b(BigInt("11111111111111111111"), BigInt("259259259259259259259"));

  std::swap(a, b);
  CHECK(a.ToString() == "11111111111111111111/259259259259259259259");
  CHECK(b.ToString() == "11111111111111111117/555555555555555555555");
}

static void TestRatMove() {
  BigRat a(BigInt("11111111111111111117"), BigInt("555555555555555555555"));
  BigRat b(BigInt("11111111111111111111"), BigInt("259259259259259259259"));

  BigRat c{std::move(a)};
  CHECK(c.ToString() == "11111111111111111117/555555555555555555555");

  a = std::move(b);
  CHECK(a.ToString() == "11111111111111111111/259259259259259259259");

  b = BigRat(7);
  CHECK(b.ToString() == "7");
}

static void TestRatSqrt() {
  printf("Test sqrt:\n");
  {
    BigInt inv_epsilon{1000000000};
    BigRat half = BigRat::Sqrt(BigRat(1, 4), inv_epsilon);

    BigRat err = BigRat::Abs(BigRat::Minus(half, BigRat(1, 2)));

    BigRat epsilon{BigInt(1), inv_epsilon};
    CHECK(BigRat::LessEq(err, epsilon));
  }

  {
    #if BIG_USE_GMP
    const char *INV_EPS =
      "1000000000000000000000000000000000000000000000000000"
      "00000000000000000000000000000000000000000000000000000"
      "00000000000000000000000000000000000000000000000000000";
    const char *INV_EPS2 =
      "10000000000000000000000000000000000000000000000000000";
    #else
    const char *INV_EPS = "10000000000000000000000000000";
    const char *INV_EPS2 = "10000000000";
    #endif

    BigInt inv_epsilon{INV_EPS};

    BigRat approx_pi =
      BigRat::FromDecimal(
          "3.141592653589793238462643383279502884197169399375105820974"
          "94459230781640628620899862803482534211706798214808651328230"
          "66470938446095505822317253594081284811174502841027019385211"
          "05559644622948954930381964428810975665933446128475648233786"
          "78316527120190914564856692346034861045432664821339360726024"
          "9141273724587006");

    BigRat root_pi = BigRat::Sqrt(approx_pi, inv_epsilon);

    BigRat expected =
      BigRat::FromDecimal(
          "1.772453850905516027298167483341145182797549456122387128213"
          "80778985291128459103218137495065673854466541622682362428257"
          "06662361528657244226025250937096027870684620376986531051228"
          "49925173028950826228932095379267962800174639015351479720516"
          "70019018523401858544697449491264031392177552590621640541933"
          "25009063984076137334774751534336679897893658518364087954511"
          "65161738760059067393431791332809854846248184902054654852195");

    BigRat err = BigRat::Abs(BigRat::Minus(root_pi, expected));

    BigRat epsilon{BigInt{1}, inv_epsilon};
    CHECK(BigRat::LessEq(err, epsilon));

    BigRat sq = BigRat::Times(expected, expected);

    BigRat err2 = BigRat::Abs(BigRat::Minus(approx_pi, sq));
    BigRat eps2 = BigRat(BigInt{1}, BigInt{INV_EPS2});
    CHECK(BigRat::LessEq(err2, eps2));
  }
  printf("Sqrt OK\n");
}

static void TestRatCbrt() {
  printf("Test cbrt:\n");
  {
    BigInt inv_epsilon = BigInt(int64_t{1000000000});
    BigRat half = BigRat::Cbrt(BigRat(1, 8), inv_epsilon);

    BigRat err = BigRat::Abs(BigRat::Minus(half, BigRat(1, 2)));

    CHECK(BigRat::LessEq(err, BigRat(BigInt(1), inv_epsilon))) <<
      half.ToString() << "\n = \n" << half.ToDouble();
  }

  {
    #if BIG_USE_GMP
    const char *INV_EPS =
      "1000000000000000000000000000000000000000000000000000"
      "00000000000000000000000000000000000000000000000000000"
      "00000000000000000000000000000000000000000000000000000";
    const char *INV_EPS2 =
      "10000000000000000000000000000000000000000000000000000";
    #else
    const char *INV_EPS = "10000000000000000000000000000";
    const char *INV_EPS2 = "10000000000";
    #endif

    BigInt inv_epsilon{INV_EPS};

    BigRat approx_pi =
      BigRat::FromDecimal(
          "3.141592653589793238462643383279502884197169399375105820974"
          "94459230781640628620899862803482534211706798214808651328230"
          "66470938446095505822317253594081284811174502841027019385211"
          "05559644622948954930381964428810975665933446128475648233786"
          "78316527120190914564856692346034861045432664821339360726024"
          "9141273724587006");

    BigRat cbrt_pi = BigRat::Cbrt(approx_pi, inv_epsilon);

    BigRat expected =
      BigRat::FromDecimal(
          "1.464591887561523263020142527263790391738596855627937174357"
          "25593713839364979828626614568206782035382089750397001521899"
          "32809245750259902118221943750432362715959855331660754443485"
          "09903392045088050903776065494919699342658172267942315230283"
          "41629182121187298054296401930582483384853899500923551707960"
          "75783189347980366547850256686536710438778249536690301754859"
          "12120284965639908171248396095188032464249826403243390244800"
          "31842046360937876480565369407819683548409250105232633503176"
          "35399902875649696425049942464679589940958825254252541166052");

    BigRat err = BigRat::Abs(BigRat::Minus(cbrt_pi, expected));

    BigRat epsilon(BigInt(1), inv_epsilon);
    CHECK(BigRat::LessEq(err, epsilon));

    BigRat cube = BigRat::Times(expected,
                                BigRat::Times(expected, expected));

    BigRat err2 = BigRat::Abs(BigRat::Minus(approx_pi, cube));
    BigRat eps2 = BigRat{BigInt(1), BigInt{INV_EPS2}};
    CHECK(BigRat::LessEq(err2, eps2)) << cube.ToDouble();
  }
  printf("Cbrt OK\n");
}

static void TestRatTruncate() {
  printf("Test rat truncate:\n");

  {
    BigRat r = BigRat::Truncate(BigRat{1, 2}, BigInt(1000));
    CHECK(r.ToString() == "1/2") << r.ToString();
  }


  BigRat pi = BigRat::FromDouble(std::numbers::pi);

  {
    BigRat r = BigRat::Truncate(pi, BigInt(10));
    CHECK(r.ToString() == "22/7") << r.ToString();
  }

  {
    BigRat r = BigRat::Truncate(BigRat::Negate(pi), BigInt(1000));
    CHECK(r.ToString() == "-355/113") << r.ToString();
  }

  {
    BigRat r = BigRat::Truncate(pi, BigInt(2000000));
    CHECK(r.ToString() == "5419351/1725033") << r.ToString();
  }


  BigRat phi = BigRat::FromDouble(std::numbers::phi);
  {
    // Successive Fibonacci numbers approximate phi. Also, in some
    // sense this is the hardest real number to approximate rationally.
    BigRat r = BigRat::Truncate(phi, BigInt(40000000));
    CHECK(r.ToString() == "63245986/39088169") << r.ToString();
  }
}

static void TestCtz() {
  CHECK(BigInt::BitwiseCtz(BigInt(0)) == 0);
  CHECK(BigInt::BitwiseCtz(BigInt(-1)) == 0);
  CHECK(BigInt::BitwiseCtz(BigInt(1)) == 0);
  CHECK(BigInt::BitwiseCtz(BigInt(2)) == 1);
  CHECK(BigInt::BitwiseCtz(BigInt(-2)) == 1);

  CHECK(BigInt::BitwiseCtz(
            BigInt::Times(BigInt::Pow(BigInt{2}, 389), 33)) == 389);
}

static void TestDivisibleBy() {
  BigInt threes("3333333333333333333333333333333333333333333333");
  CHECK(BigInt::DivisibleBy(threes, 3));
  CHECK(BigInt::DivisibleBy(threes, BigInt(3)));

  BigInt p(31337);
  BigInt xp = BigInt::Times(threes, p);
  CHECK(!BigInt::DivisibleBy(p, 3));
  CHECK(!BigInt::DivisibleBy(p, BigInt(3)));
  CHECK(BigInt::DivisibleBy(xp, 3));
  CHECK(BigInt::DivisibleBy(xp, BigInt(3)));

  CHECK(BigInt::DivisibleBy(xp, xp));
  CHECK(BigInt::DivisibleBy(xp, threes));
  CHECK(BigInt::DivisibleBy(p, 1));
}

static void TestCompare() {
  BigInt zero{0};
  BigInt one{1};
  BigInt two{2};
  BigInt neg_one{-1};
  BigInt neg_two{-2};

  BigInt large_pos("33333333333333333333333333333333333333333");
  BigInt large_neg("-888888888888888888833333333333333333333333");

  for (const auto &[a, b] :
         std::initializer_list<std::pair<BigInt, BigInt>>{
         {one, two},
         {zero, one},
         {zero, two},
         {neg_one, zero},
         {large_neg, zero},
         {neg_one, two},
         {neg_one, one},
         {neg_two, neg_one},
         {large_neg, large_pos},
         {large_neg, neg_two},
         {two, large_pos},
       }) {
      CHECK(BigInt::Less(a, b));
      CHECK(!BigInt::Less(b, a));
      CHECK(BigInt::LessEq(a, b));
      CHECK(!BigInt::LessEq(b, a));
      CHECK(BigInt::LessEq(a, a));
      CHECK(BigInt::LessEq(b, b));
      CHECK(!BigInt::Eq(a, b));
      CHECK(!BigInt::Eq(b, a));

      CHECK(BigInt::Greater(b, a));
      CHECK(BigInt::GreaterEq(b, a));
      CHECK(!BigInt::Greater(a, b));
      CHECK(!BigInt::GreaterEq(a, b));
      CHECK(BigInt::GreaterEq(a, a));
      CHECK(BigInt::GreaterEq(b, b));
  }
}

static void TestRatCompare() {

  CHECK(BigRat::Compare(BigRat(1, 5), BigRat(2, 5)) == -1);
  CHECK(BigRat::Compare(BigRat(1, 5), BigRat(-2, 5)) == 1);
  CHECK(BigRat::Compare(BigRat(5, 10), BigRat(1, 2)) == 0);

  CHECK(BigRat::Eq(BigRat(1, 2),
                   BigRat::Max(BigRat(5, 10), BigRat(1, 2))));
  CHECK(BigRat::Eq(BigRat(1, 2),
                   BigRat::Max(BigRat(1, 10), BigRat(1, 2))));

  CHECK(BigRat::Eq(BigRat(1, 2),
                   BigRat::Max(BigRat(5, 10), BigRat(-1, 3))));

  CHECK(BigRat::Less(BigRat(BigInt(9007199254739967),
                            BigInt(18014398509481984)),
                     BigRat(1, 2)));

  // 9007199254739967/18014398509481984 is 0.4999999999999431010
  CHECK(BigRat::Compare(BigRat(BigInt(9007199254739967),
                               BigInt(18014398509481984)),
                        BigRat(1, 2)) == -1);
}

static void TestRatHashCode() {
  // Technically this can be anything, but it would probably be a bug if
  // small integers don't get different values.
  std::unordered_set<uint64_t> distinct;
  for (const auto &r : {
        BigRat{123, 1},
        BigRat{567, 1},
        BigRat{123, 2},
        BigRat{567, 2}
    }) {
    distinct.insert(BigRat::HashCode(r));
  }
  CHECK(distinct.size() == 4);

  BigInt n{"190237849028374901872390876190872349817230948719023874190827349817239048712903847190283740918273490817230948798767676767676738347482712341"};
  BigInt d{"2398198723048348182374029384190823049810881688181811116747484774117777777777777777777777772039488118060668834743080000273410875486776387"};

  distinct.insert(BigRat::HashCode(BigRat(n, d)));
  CHECK(distinct.size() == 5);
  distinct.insert(BigRat::HashCode(BigRat(d, n)));
  CHECK(distinct.size() == 6);
  distinct.insert(BigRat::HashCode(BigRat(d, n)));
  CHECK(distinct.size() == 6);

  distinct.insert(BigRat::HashCode(BigRat(BigInt::Negate(n), d)));
  CHECK(distinct.size() == 7) << "Is it ignoring the sign bit?";
}

static void TestRatSign() {
  CHECK(BigRat::Sign(BigRat(BigInt(-3))) == -1);
  CHECK(BigRat::Sign(BigRat(BigInt(0))) == 0);
  CHECK(BigRat::Sign(BigRat(BigInt("19823749283749817"))) == 1);
  CHECK(BigRat::Sign(BigRat(BigInt("-999999999999999999999"))) == -1);

  CHECK(BigRat::Sign(BigRat(-3, 2)) == -1);
  CHECK(BigRat::Sign(BigRat(3, -2)) == -1);
  CHECK(BigRat::Sign(BigRat(0, -1)) == 0);
  CHECK(BigRat::Sign(BigRat(3215, 75)) == 1);
  CHECK(BigRat::Sign(BigRat(-3, -4)) == 1);
}

int main(int argc, char **argv) {
  printf("Start.\n");
  fflush(stdout);

  CopyAndAssign();
  TestToString();
  TestSign();
  TestToU64();

  TestCompare();
  TestRatCompare();

  TestRatString();
  TestRatFromDouble();
  TestToDouble();
  TestRatToDouble();

  TestDiv();
  TestCMod();
  TestModRem();

  TestEq();
  HashCode();
  TestMod();
  TestToInt();
  TestGCD();

  TestLog();

  TestDivisibleBy();
  TestDivExact();
  TestDivFloor();

  TestShift();
  TestAnd();
  TestXor();
  TestOr();
  TestCtz();

  TestLeadingZero();

  TestPow();
  TestQuotRem();
  TestPowMod();

  TestRatTruncate();

  TestPrimeFactors();

  TestPi();

  // Slow
  if (RUN_BENCHMARKS) {
    BenchDiv2();
  }

  TestSqrt();
  TestJacobi();
  TestInvert();

  TestSwap();
  TestRatSwap();
  TestRatMove();
  TestRatSqrt();
  TestRatCbrt();
  TestRatSign();

  TestRatHashCode();

  printf("OK\n");
}
