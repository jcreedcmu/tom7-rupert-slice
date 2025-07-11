/* PLEASE KEEP THIS LINE */
// (The previous line is used by the test, which opens this test file
// as text.)

#include "util.h"

#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "base/logging.h"

using namespace std;

#define CHECK_SEQ(a, b) do { \
  auto aa = (a); \
  auto bb = (b); \
  CHECK(aa == bb) << "Expected equal strings:\n" << #a << "\n" << #b \
                  << "\nValues:\n" << aa << "\n" << bb << "\n";      \
  } while (false)

#define CHECK_SVEQ(a, b) do { \
  auto aa = (a); \
  auto bb = (b); \
  CHECK(aa == bb) << "Expected equal vectors:\n" << #a << "\n" << #b \
                  << "\nValues:\n" << Util::Join(aa, "|") << "\n"    \
                  << Util::Join(bb, "|") << "\n";              \
  } while (false)

#define CHECK_OPT_VALUE(a, b) do {                            \
    auto aa = (a);                                            \
    auto bb = (b);                                            \
    CHECK(aa.has_value()) << "Expected optional " << #a       \
                          << " to have a value.";             \
    const auto &aav = aa.value();                             \
    CHECK_EQ(aav, bb) << "For " << #a << " got " << aav       \
                      << " but wanted " << #b " << = " << bb; \
  } while (false)


static string SlowReadFile(const string &filename) {
  FILE *f = fopen(filename.c_str(), "rb");
  if (!f) return "";
  string ret;
  ret.reserve(128);
  int c = 0;
  while (EOF != (c = getc(f))) {
    ret += (char)c;
  }
  fclose(f);
  return ret;
}

static constexpr char NONEXISTENT_FILE[] =
  "util_test_DOESNT_EXIST.cc";


// This test uses its source code as test data, so don't
// mess with the lines that tell you to keep them, duh.
static void TestReadFiles(const char *argv0) {
  /*
  printf("argv[0] = [%s]. BinaryDir: %s\n",
         argv0, Util::BinaryDir(argv0).c_str());
  */
  std::string self = Util::DirPlus(Util::BinaryDir(argv0), "util_test.cc");

  const string reference = SlowReadFile(self);
  CHECK(!reference.empty()) << self;
  // Self-check.
  CHECK(reference.find("KEEP THIS LINE TOO */") != string::npos);
  CHECK(reference.find("/* PLEASE KEEP THIS LINE */") == 0);
  const string s1 = Util::ReadFile(self);
  CHECK_EQ(reference, s1);
  const string s2 = Util::ReadFileMagic(self,
                                        "/* PLEASE KEEP THIS");
  CHECK_EQ(reference, s2);
  CHECK_EQ("", Util::ReadFileMagic(self, "WRONG"));
  CHECK_EQ("", Util::ReadFile(NONEXISTENT_FILE));
  CHECK_EQ("", Util::ReadFileMagic(NONEXISTENT_FILE, "/"));
  CHECK(Util::HasMagic(self, "/* PLEASE KEEP THIS"));
  CHECK(!Util::HasMagic(self, "* PLEASE KEEP"));
  // Would be nice to test files larger than 2^31 and 2^32, since these
  // have caused problems in the past.

  std::optional<string> os3 = Util::ReadFileOpt(self);
  CHECK(os3.has_value());
  CHECK_EQ(reference, *os3);

  std::optional<string> os4 = Util::ReadFileOpt(NONEXISTENT_FILE);
  CHECK(!os4.has_value());
}

static void TestWhitespace() {
  CHECK_EQ("", Util::LoseWhiteR(""));
  CHECK_EQ("", Util::LoseWhiteR(" \n\r \n"));
  CHECK_EQ(" p", Util::LoseWhiteR(" p \n\r \n"));
  CHECK_EQ("  p\nq", Util::LoseWhiteR("  p\nq\n\r \n"));
  CHECK_EQ("rrr", Util::LoseWhiteR("rrr"));

  CHECK_EQ("", Util::NormalizeWhitespace(""));
  CHECK_EQ("", Util::NormalizeWhitespace(" "));
  CHECK_EQ("", Util::NormalizeWhitespace(" \r\n \r \r"));
  CHECK_EQ("hello world", Util::NormalizeWhitespace("  \nhello \r\n \r "
                                                    "\rworld  \r\r\n"));
  CHECK_EQ("hello world", Util::NormalizeWhitespace("hello world"));
  CHECK_EQ("hello world", Util::NormalizeWhitespace("\thello\tworld\t"));
  CHECK_EQ("hello world", Util::NormalizeWhitespace("hello\nworld"));
  string s;
  CHECK_EQ("", Util::NormalizeWhitespace(s));

  CHECK(Util::IsWhitespace(' '));
  CHECK(Util::IsWhitespace('\r'));
  CHECK(Util::IsWhitespace('\n'));
  CHECK(Util::IsWhitespace('\t'));
  CHECK(!Util::IsWhitespace('_'));
  CHECK(!Util::IsWhitespace('\0'));
}

static void TestNormalizeLines() {
  std::vector<std::string> v = {"", "  ", " he  y ", "   yes"};
  std::vector<std::string> vv = Util::NormalizeLines(v);
  CHECK(vv.size() == 2);
  CHECK(vv[0] == "he y");
  CHECK(vv[1] == "yes");
}

static void TestPad() {
  CHECK_EQ("", Util::Pad(0, ""));
  CHECK_EQ("hello  ", Util::Pad(7, "hello"));
  CHECK_EQ("  hello", Util::Pad(-7, "hello"));
  CHECK_EQ("hello", Util::Pad(4, "hello"));
  CHECK_EQ("hello", Util::Pad(-4, "hello"));

  CHECK_EQ("", Util::PadEx(0, "", '_'));
  CHECK_EQ("hello__", Util::PadEx(7, "hello", '_'));
  CHECK_EQ("__hello", Util::PadEx(-7, "hello", '_'));
  CHECK_EQ("hello", Util::PadEx(4, "hello", '_'));
  CHECK_EQ("hello", Util::PadEx(-4, "hello", '_'));
}

static void TestPadEx() {
  // Test with zero padding.
  CHECK_EQ("hello", Util::PadEx(5, "hello", '\0'));
  CHECK_EQ("hello", Util::PadEx(-5, "hello", '\0'));
  CHECK_EQ("hello\0\0"sv, Util::PadEx(7, "hello", '\0'));
  CHECK_EQ("\0\0hello"sv, Util::PadEx(-7, "hello", '\0'));
}

static void TestJoin() {
  CHECK_EQ("", Util::Join({}, "X"));
  CHECK_EQ("aYYbYYcde", Util::Join({"a", "b", "cde"}, "YY"));
  CHECK_EQ("abcde", Util::Join({"a", "b", "cde"}, ""));
}

static void TestSplit() {
  CHECK_SVEQ((vector<string>{"hello", "world"}),
             Util::Split("hello world", ' '));

  CHECK_SVEQ((vector<string>{"hello", "", "world"}),
             Util::Split("hello  world", ' '));
  CHECK_SVEQ((vector<string>{"", ""}), Util::Split(" ", ' '));
  CHECK_SVEQ(vector<string>{""}, Util::Split("", 'x'));

  CHECK_SVEQ((vector<string>{"a", "", "b", "", "c"}),
             Util::Split("a||b||c", '|'));
}

static void TestSplitWith() {
  CHECK_SVEQ((vector<string>{"hello", "", "world"}),
             Util::SplitWith("hello  world", " "));

  CHECK_SVEQ((vector<string>{"hello", ".world"}),
             Util::SplitWith("hello...world", ".."));

  CHECK_SVEQ((vector<string>{"", "face", "first"}),
             Util::SplitWith("\\nface\\nfirst", "\\n"));

  CHECK_SVEQ((vector<string>{"face", "last", ""}),
             Util::SplitWith("face\\nlast\\n", "\\n"));
}

static void TestSplitToLines() {
  CHECK_SVEQ((vector<string>{"hello", "world"}),
             Util::SplitToLines("hello\nworld"));
  CHECK_SVEQ((vector<string>{"hello", "world"}),
             Util::SplitToLines("hello\nworld\n"));
  CHECK_SVEQ((vector<string>{"hello"}),
             Util::SplitToLines("hello\n"));
  CHECK_SVEQ((vector<string>{"hello"}),
             Util::SplitToLines("hello"));
}

static void TestTokens() {
  auto IsSpace = [](char c) { return c == ' ' || c == '\n'; };
  /*
  for (const string s : Util::Tokens(" \n hello  world\n\n", IsSpace))
    printf("[%s]\n", s.c_str());
  */
  CHECK_EQ((vector<string>{"hello", "world"}),
           Util::Tokens(" \n hello  world\n\n", IsSpace));
  CHECK_EQ((vector<string>{}),
           Util::Tokens(" \n \n\n", IsSpace));
  CHECK_EQ((vector<string>{}),
           Util::Tokens("", IsSpace));
}

static void TestNextToken() {
  {
    std::string s = " this  is a teststring   ";
    std::string_view view(s);
    CHECK_EQ(Util::NextToken(&view, ' '), "this");
    CHECK_EQ(Util::NextToken(&view, ' '), "is");
    CHECK_EQ(Util::NextToken(&view, ' '), "a");
    CHECK_EQ(Util::NextToken(&view, ' '), "teststring");
    CHECK(view.empty());
    CHECK(Util::NextToken(&view, ' ').empty());
    CHECK(view.empty());
    CHECK(!s.empty());
  }

  {
    std::string s = "";
    std::string_view view(s);
    CHECK(view.empty());
    CHECK(Util::NextToken(&view, ' ').empty());
    CHECK(view.empty());
  }

  {
    std::string s = "x";
    std::string_view view(s);
    CHECK_EQ(Util::NextToken(&view, ' '), "x");
    CHECK(view.empty());
    CHECK(Util::NextToken(&view, ' ').empty());
    CHECK(view.empty());
  }

  {
    std::string s = " ";
    std::string_view view(s);
    CHECK(!view.empty());
    CHECK(Util::NextToken(&view, ' ').empty());
    CHECK(view.empty());
    CHECK(!s.empty());
  }

}

static void TestTokenize() {
  CHECK_EQ((vector<string>{"hello", "world"}),
           Util::Tokenize("....hello.world...", '.'));
  CHECK_EQ((vector<string>{"hello", "world"}),
           Util::Tokenize("hello.world", '.'));
  CHECK_EQ((vector<string>{}),
           Util::Tokenize("...", '.'));
  CHECK_EQ((vector<string>{}),
           Util::Tokenize("", '.'));
}

static void TestCdup() {
  // Note: Failures here could be because util_test.o and util.o have
  // different values of DIRSEP. Probably should make this more sane!
  CHECK_EQ("abc" DIRSEP "de", Util::cdup("abc" DIRSEP "de" DIRSEP "f"));
  CHECK_EQ(".", Util::cdup("abc"));
}

static void TestPrefixSuffix() {
  CHECK(Util::StartsWith("anything", ""));
  CHECK(!Util::StartsWith("", "nonempty"));
  CHECK(Util::EndsWith("anything", ""));
  CHECK(!Util::EndsWith("", "nonempty"));

  CHECK(Util::StartsWith("food processor", "foo"));
  CHECK(!Util::StartsWith("food processor", "doo"));
  CHECK(Util::EndsWith("food processor", "sor"));
  CHECK(!Util::EndsWith("food processor", "sdr"));

  {
    // String versions.
    string s = "food processor";
    CHECK(!Util::TryStripPrefix("foods", &s));
    CHECK(Util::TryStripPrefix("foo", &s) &&
          s == "d processor");
    CHECK(!Util::TryStripSuffix("sord", &s));
    CHECK(Util::TryStripSuffix("sor", &s) &&
          s == "d proces");
  }

  {
    // string_piece versions
    string_view s = "food processor"sv;
    CHECK(!Util::TryStripPrefix("foods", &s));
    CHECK(Util::TryStripPrefix("foo", &s) &&
          s == "d processor");
    CHECK(!Util::TryStripSuffix("sord", &s));
    CHECK(Util::TryStripSuffix("sor", &s) &&
          s == "d proces");
  }
}

static void TestContains() {
  CHECK(Util::StrContains("haystack", "hay"));
  CHECK(Util::StrContains("haystack", ""));
  CHECK(Util::StrContains("haystack", "stack"));
  CHECK(Util::StrContains("haystack", "haystack"));
  CHECK(Util::StrContains("haystack", "a"));
  CHECK(!Util::StrContains("haystack", "stacks"));
  CHECK(!Util::StrContains("haystack", "chay"));
  CHECK(!Util::StrContains("haystack", "haystacks"));
  CHECK(!Util::StrContains("haystack", "k0"));
}

static void TestWriteFiles() {
  const string f = "util_test.deleteme";
  const string ss = "the\n\0contents";
  CHECK(Util::WriteFile(f, ss));
  string ssr = Util::ReadFile(f);
  CHECK(ss == ssr);

  CHECK(Util::WriteFile(f, ""));
  string sse = Util::ReadFile(f);
  CHECK(sse.empty());

  vector<string> lines = {
    "the first line",
    "",
    "",
    " white - space ",
  };

  CHECK(Util::WriteLinesToFile(f, lines));
  vector<string> rlines = Util::ReadFileToLines(f);
  CHECK(lines == rlines);
}

static void TestParseDouble() {
  CHECK(Util::ParseDouble(" +3 ") == 3.0);
  CHECK(Util::ParseDouble("  -3.5   ") == -3.5);
  CHECK(Util::ParseDouble("-3.5") == -3.5);
  CHECK(Util::ParseDouble("000") == 0.0);
  CHECK(Util::ParseDouble("000", 1.0) == 0.0);
  CHECK(Util::ParseDouble("0", 1.0) == 0.0);
  CHECK(Util::ParseDouble("-0", 1.0) == 0.0);
  double nnn = Util::ParseDouble("nan", 1.0);
  CHECK(std::isnan(nnn)) << "expected NaN";
  CHECK(Util::ParseDouble("-2e3") == -2e3);
  CHECK(Util::ParseDouble("asdf", 3.0) == 3.0);
  CHECK(Util::ParseDouble("33asdf", 2.0) == 2.0);
  CHECK(Util::ParseDouble("asdf33", 2.0) == 2.0);
}

static void TestFactorize() {
  CHECK((vector<int>{2, 2, 2, 3}) == Util::Factorize(2 * 2 * 2 * 3));
  CHECK((vector<int>{2, 2}) == Util::Factorize(4));
  CHECK((vector<int>{2}) == Util::Factorize(2));
  CHECK((vector<int>{5, 5}) == Util::Factorize(5 * 5));
  CHECK((vector<int>{31337}) == Util::Factorize(31337));
  CHECK((vector<int>{3, 5, 11, 13}) == Util::Factorize(3 * 5 * 11 * 13));
  CHECK((vector<int>{3, 5, 5, 11, 13}) ==
        Util::Factorize(3 * 5 * 5 * 11 * 13));
  CHECK((vector<int>{3, 5, 5, 11, 11, 11}) ==
        Util::Factorize(3 * 5 * 5 * 11 * 11 * 11));
  CHECK((vector<int>{3, 5, 5, 11, 13, 31337}) ==
        Util::Factorize(3 * 5 * 5 * 11 * 13 * 31337));
}

static void TestItos() {
  CHECK_EQ(Util::itos(1234), "1234");
  CHECK_EQ(Util::itos(-1234), "-1234");
}

static void TestStoi() {
  CHECK_EQ(Util::stoi("1234"), 1234);
  CHECK_EQ(Util::stoi("-1234"), -1234);
  CHECK_EQ(Util::stoi(""), 0);
  CHECK_EQ(Util::stoi("a"), 0);
}

static void TestMatchSpec() {
  CHECK(Util::MatchSpec("u", 'u'));
  CHECK(!Util::MatchSpec("u", 'v'));
  CHECK(Util::MatchSpec("uv", 'u'));
  CHECK(Util::MatchSpec("uv", 'v'));
  CHECK(Util::MatchSpec("0-9", '3'));
  CHECK(Util::MatchSpec("0-9", '0'));
  CHECK(Util::MatchSpec("0-9", '9'));
  CHECK(!Util::MatchSpec("0-9", 'a'));
  CHECK(Util::MatchSpec("0-9a-z", 'a'));
  CHECK(Util::MatchSpec("a-z0-9", 'a'));
  CHECK(Util::MatchSpec("a-z0-9", '3'));
  CHECK(Util::MatchSpec("0-9a-z", '3'));
  CHECK(Util::MatchSpec("-9a", '-'));
  CHECK(Util::MatchSpec("-9a", '9'));
  CHECK(Util::MatchSpec("-9a", 'a'));
  CHECK(Util::MatchSpec("9a-", '9'));
  CHECK(Util::MatchSpec("9a-", 'a'));
  CHECK(Util::MatchSpec("9a-", '-'));

  CHECK(!Util::MatchSpec("a-z", '-'));

  CHECK(!Util::MatchSpec("^u", 'u'));
  CHECK(Util::MatchSpec("^u", 'v'));
  CHECK(!Util::MatchSpec("^uv", 'u'));
  CHECK(!Util::MatchSpec("^uv", 'v'));
  CHECK(!Util::MatchSpec("^0-9", '3'));
  CHECK(!Util::MatchSpec("^0-9", '0'));
  CHECK(!Util::MatchSpec("^0-9", '9'));
  CHECK(Util::MatchSpec("^0-9", 'a'));
  CHECK(!Util::MatchSpec("^0-9a-z", 'a'));
  CHECK(!Util::MatchSpec("^a-z0-9", 'a'));
  CHECK(!Util::MatchSpec("^a-z0-9", '3'));
  CHECK(!Util::MatchSpec("^0-9a-z", '3'));

  CHECK(!Util::MatchSpec("-9a", 'b'));
  CHECK(!Util::MatchSpec("-9a", '8'));
  CHECK(!Util::MatchSpec("9a-", '8'));
  CHECK(!Util::MatchSpec("9a-", 'b'));


  // With string inputs.
  CHECK(Util::MatchSpec("", ""));
  CHECK(Util::MatchSpec("a-z", ""));
  CHECK(Util::MatchSpec("^a-z", ""));
  CHECK(Util::MatchSpec("^a-z", "1001"));
  CHECK(!Util::MatchSpec("^a-z", "10z1"));
  CHECK(Util::MatchSpec("a-z", "abraz"));
  CHECK(Util::MatchSpec("A-Za-z0-9", "Alphanumeric007"));
  CHECK(!Util::MatchSpec("A-Za-z0-9", "Alpha numeric 007!"));
}

static void TestMatchesWildcard() {
  CHECK(Util::MatchesWildcard("abc", "abc"));
  CHECK(!Util::MatchesWildcard("", "abc"));
  CHECK(Util::MatchesWildcard("", ""));
  CHECK(!Util::MatchesWildcard("", "a"));
  CHECK(Util::MatchesWildcard("a?c", "abc"));
  CHECK(!Util::MatchesWildcard("a?c", "ac"));

  CHECK(Util::MatchesWildcard("*.exe", "debug.exe"));
  CHECK(!Util::MatchesWildcard("*.exe", "debug.com"));
  CHECK(!Util::MatchesWildcard("*.exe", "debug.exeno"));
  CHECK(Util::MatchesWildcard("*.exe", ".exe"));

  CHECK(Util::MatchesWildcard("debug.*", "debug.exe"));
  CHECK(Util::MatchesWildcard("debug.*", "debug.com"));
  CHECK(Util::MatchesWildcard("debug.*", "debug."));
  CHECK(!Util::MatchesWildcard("debug.*", "format.com"));

  CHECK(Util::MatchesWildcard("f*.exe", "format.exe"));
  CHECK(!Util::MatchesWildcard("f*.exe", "debug.exe"));
  CHECK(!Util::MatchesWildcard("f*.exe", "f.com"));

  CHECK(Util::MatchesWildcard("f*.*.exe", "format.00.exe"));
  CHECK(!Util::MatchesWildcard("f*.*.exe", "format.exe"));

  CHECK(Util::MatchesWildcard("z*", "zzzz"));
  CHECK(Util::MatchesWildcard("z*", "z"));
  CHECK(Util::MatchesWildcard("*z*", "z"));

  CHECK(Util::MatchesWildcard("*?*", "z"));
  CHECK(Util::MatchesWildcard("*?*", "abcdefg"));
  CHECK(!Util::MatchesWildcard("*?*", ""));
  CHECK(!Util::MatchesWildcard("?", ""));
  CHECK(!Util::MatchesWildcard("?", "aa"));

  CHECK(Util::MatchesWildcard("*", ""));
  CHECK(Util::MatchesWildcard("**", ""));
  CHECK(Util::MatchesWildcard("*", "hi"));
  CHECK(Util::MatchesWildcard("***", "hi"));
  CHECK(Util::MatchesWildcard("h***", "hi"));
  CHECK(Util::MatchesWildcard("h***i", "hi"));
  CHECK(Util::MatchesWildcard("h***i***", "hi"));
  CHECK(Util::MatchesWildcard("h***?***", "hi"));
  CHECK(Util::MatchesWildcard("h***?***", "him"));
  CHECK(!Util::MatchesWildcard("**h***?***", "jh"));

  CHECK(Util::MatchesWildcard("dddddd_*_Layer-*.png",
                              "dddddd_0053_Layer-44.png"));
}

static void TestHex() {
  for (int i = 0; i < 0x10; i++) {
    char c = Util::HexDigit(i);
    CHECK(Util::IsHexDigit(c));
    CHECK(Util::HexDigitValue(c) == i);
    // Also test the other case for letters.
    if (c < '0' || c > '9') {
      CHECK(Util::IsHexDigit(c ^ 32));
      CHECK(Util::HexDigitValue(c ^ 32) == i);
    }
  }
}

// This can be done with HexString, but keeping it around
// so that HexString can also be tested against it.
static string ToHex(const std::string &s) {
  string out;
  out.resize(s.size() * 2);
  for (int i = 0; i < (int)s.size(); i++) {
    out[i * 2 + 0] = Util::HexDigit((s[i] >> 4) & 0xF);
    out[i * 2 + 1] = Util::HexDigit(s[i] & 0xF);
  }
  return out;
}

static void TestHexString() {
  CHECK_SEQ(Util::HexString(""), ToHex(""));
  CHECK_SEQ(Util::HexString("*"), ToHex("*"));
  CHECK_SEQ(Util::HexString("", "", ""), ToHex(""));
  CHECK_SEQ(Util::HexString("*", "", ""), ToHex("*"));
  CHECK_SEQ(Util::HexString("\x0f\xf0\x88\x77" "frog"),
            ToHex("\x0f\xf0\x88\x77" "frog"));
  {
    string s = "mystery";
    s[1] = '\0';
    s[4] = '\xFF';
    CHECK_SEQ(Util::HexString(s), ToHex(s));
  }

  CHECK_SEQ(Util::HexString("hi"), "6869");
  CHECK_SEQ(Util::HexString("hi", "_"), "68_69");
  CHECK_SEQ(Util::HexString("hi", ", ", "0x"), "0x68, 0x69");
}

static void TestReplace() {
  CHECK_SEQ(Util::Replace("abc", "b", "xxx"), "axxxc");
  CHECK_SEQ(Util::Replace("abc", "z", "xxx"), "abc");
  CHECK_SEQ(Util::Replace("aaa", "a", "aba"), "abaabaaba");
  CHECK_SEQ(Util::Replace("unaffected", "", "z"), "unaffected");
}

static void TestCommas() {
  CHECK_SEQ("1,000", Util::UnsignedWithCommas(1000));
  CHECK_SEQ("999", Util::UnsignedWithCommas(999));
  CHECK_SEQ("42", Util::UnsignedWithCommas(42));
  CHECK_SEQ("0", Util::UnsignedWithCommas(0));
  CHECK_SEQ("20,300,010", Util::UnsignedWithCommas(20300010));
  CHECK_SEQ("7,111,222,333,444",
            Util::UnsignedWithCommas(7'111'222'333'444ULL));
}

static void TestMemMem() {
  const char *s1 = "the quick brown quack foxed";
  const uint8_t *s2 = Util::MemMem((const uint8_t*)s1, strlen(s1),
                                   (const uint8_t*)"quack", 5);
  CHECK(s1 != nullptr);
  CHECK(0 == memcmp(s2, "quack", 5));

  CHECK(Util::MemMem((const uint8_t*)s1, strlen(s1),
                     (const uint8_t*)"quark", 5) == nullptr);

  CHECK((const uint8_t*)s1 == Util::MemMem((const uint8_t*)s1, strlen(s1),
                                           (const uint8_t*)"", 0));
  CHECK((const uint8_t*)s1 == Util::MemMem((const uint8_t*)s1, strlen(s1),
                                           (const uint8_t*)"t", 1));
}

static void TestCase() {
  CHECK_SEQ(Util::lcase("BoVeX 999\n"), "bovex 999\n");
  CHECK_SEQ(Util::ucase("BoVeX 999\n"), "BOVEX 999\n");
}

static void TestPaths() {
  CHECK_SEQ(Util::FileExtOf("test.pdf"), "pdf");
  CHECK_SEQ(Util::FileExtOf("no.test.pdf"), "pdf");
  CHECK_SEQ(Util::FileExtOf("test."), "");
  CHECK_SEQ(Util::FileExtOf("test"), "");
  CHECK_SEQ(Util::FileBaseOf("test.pdf"), "test");
  CHECK_SEQ(Util::FileBaseOf("no.test.pdf"), "no.test");
  CHECK_SEQ(Util::FileBaseOf("test."), "test");
  CHECK_SEQ(Util::FileBaseOf("test"), "test");
}

static void TestRemoveChars() {
  CHECK_SEQ(Util::RemoveCharsMatching(
                "get out of my string!",
                [](char c) {
                  return c == 'a' || c == 'e' || c == 'i' ||
                    c == 'o' || c == 'u';
                }), "gt t f my strng!");
  CHECK_SEQ(Util::RemoveChar("  east north  east ", ' '),
            "eastnortheast");
}

static void TestLoseWhiteL() {
  CHECK_EQ("", Util::LoseWhiteL(""));
  CHECK_EQ("", Util::LoseWhiteL("   \t\r\n"));
  CHECK_EQ("hello", Util::LoseWhiteL("   \t\r\nhello"));
  CHECK_EQ("hello world", Util::LoseWhiteL("  hello world"));
  CHECK_EQ("hello world  ", Util::LoseWhiteL("  hello world  "));
}

static void TestChop() {
  {
    string s = "  hello world";
    CHECK_EQ(Util::chop(s), "hello");
    CHECK_EQ(s, " world");
  }

  {
    string s = "  hello";
    CHECK_EQ(Util::chop(s), "hello");
    CHECK_EQ(s, "");
  }

  {
    string s = "hello";
    CHECK_EQ(Util::chop(s), "hello");
    CHECK_EQ(s, "");
  }

  {
    string s = "   ";
    CHECK_EQ(Util::chop(s), "");
    CHECK_EQ(s, "");
  }

  {
    string s = "";
    CHECK_EQ(Util::chop(s), "");
    CHECK_EQ(s, "");
  }
}

static void TestChopStringView() {
  {
    string str = "  hello world";
    std::string_view s = str;
    CHECK_EQ(Util::Chop(&s), "hello");
    CHECK_EQ(s, " world");
    CHECK_EQ(str, "  hello world");
  }

  {
    string str = "  hello";
    std::string_view s = str;
    CHECK_EQ(Util::Chop(&s), "hello");
    CHECK_EQ(s, "") << s;
  }

  {
    string str = "hello";
    std::string_view s = str;
    CHECK_EQ(Util::Chop(&s), "hello");
    CHECK_EQ(s, "");
  }

  {
    string str = "   ";
    std::string_view s = str;
    CHECK_EQ(Util::Chop(&s), "");
    CHECK_EQ(s, "");
  }

  {
    string str = "";
    std::string_view s = str;
    CHECK_EQ(Util::Chop(&s), "");
    CHECK_EQ(s, "");
  }
}


static void TestChopTo() {
  {
    string s = "hello,world";
    CHECK_EQ(Util::chopto(',', s), "hello");
    CHECK_EQ(s, "world");
  }

  {
    string s = "hello";
    CHECK_EQ(Util::chopto(',', s), "hello");
    CHECK_EQ(s, "");
  }

  {
    string s = ",hello";
    CHECK_EQ(Util::chopto(',', s), "");
    CHECK_EQ(s, "hello");
  }

  {
    string s = "";
    CHECK_EQ(Util::chopto(',', s), "");
    CHECK_EQ(s, "");
  }
}

static void TestFormatTime() {
  int64_t tt = time(nullptr);
  // TODO: Check something about it.
  // We can't just test for a particular string, though;
  // we don't want the test to depend on the current time zone!
  printf("The current time: %" PRIi64 " = [%s]\n",
         tt,
         Util::FormatTime("%Y-%m-%d, %H:%M:%S",
                          tt).c_str());
}

static void TestParseBinary() {
  CHECK(!Util::ParseBinary("").has_value());
  CHECK(!Util::ParseBinary("100a").has_value());
  CHECK(!Util::ParseBinary("a").has_value());
  CHECK(!Util::ParseBinary(
            // too many digits
            "000000000000000000000000000000000"
            "000000000000000000000000000000000").has_value());

  CHECK_OPT_VALUE(Util::ParseBinary("101010"), 42u);
  CHECK_OPT_VALUE(Util::ParseBinary("0"), 0u);
  CHECK_OPT_VALUE(Util::ParseBinary("1"), 1u);
  CHECK_OPT_VALUE(Util::ParseBinary(
                      // 64 one bits
                      "1111111111111111111111111111111111111111"
                      "111111111111111111111111"), ~uint64_t(0));
}

static void TestPathOf() {
  CHECK_SEQ(Util::PathOf("/asdf"), "/");
  CHECK_SEQ(Util::PathOf("/asdf/file.txt"), "/asdf/");
  CHECK_SEQ(Util::PathOf("file.txt"), ".");
  CHECK_SEQ(Util::PathOf("/asdf/"), "/asdf/");
}

static void TestFileOf() {
  CHECK_SEQ(Util::FileOf("/asdf"), "asdf");
  CHECK_SEQ(Util::FileOf("/asdf/file.txt"), "file.txt");
  CHECK_SEQ(Util::FileOf("file.txt"), "file.txt");
  CHECK_SEQ(Util::FileOf("/asdf/"), "");
}

int main(int argc, char **argv) {
  TestItos();
  TestStoi();
  TestReadFiles(argv[0]);
  TestWriteFiles();
  TestWhitespace();
  TestPad();
  TestPadEx();
  TestJoin();
  TestSplit();
  TestSplitWith();
  TestSplitToLines();
  TestTokens();
  TestTokenize();
  TestNextToken();
  TestCdup();
  TestPrefixSuffix();
  TestParseDouble();
  TestFactorize();
  TestMatchSpec();
  TestMatchesWildcard();
  TestHex();
  TestHexString();
  TestReplace();
  TestCommas();
  TestNormalizeLines();
  TestMemMem();
  TestCase();
  TestPaths();
  TestRemoveChars();
  TestLoseWhiteL();
  TestChop();
  TestChopStringView();
  TestChopTo();
  TestFormatTime();
  TestContains();
  TestParseBinary();
  TestPathOf();
  TestFileOf();

  printf("OK\n");
  return 0;
}


/* KEEP THIS LINE TOO */
