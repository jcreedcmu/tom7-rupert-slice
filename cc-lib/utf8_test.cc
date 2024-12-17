#include "utf8.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "arcfour.h"
#include "base/logging.h"
#include "util.h"

using namespace std;

static void TestUnicode() {
  CHECK("*" == UTF8::Encode('*'));
  // Katakana Letter Small Tu
  CHECK("\xE3\x83\x83" == UTF8::Encode(0x30C3));
  // Emoji Banana
  string banana = UTF8::Encode(0x1F34C);
  CHECK("\xF0\x9F\x8D\x8C" == banana) << Util::HexString(banana);

  CHECK(UTF8::Length("banana") == 6);
  for (int i = 1; i < 0x1000; i++) {
    // one possibly multibyte codepoint
    string ss = UTF8::Encode(i);
    // and one ascii character
    ss.push_back('c');
    size_t len = UTF8::Length(ss);
    CHECK(len == 2) << i;
  }


  ArcFour rc("utf8");
  for (int rep = 0; rep < 0x1000; rep++) {
    std::vector<uint32_t> expected;
    uint8_t numc = 1 + (rc.Byte() & 7);

    for (int j = 0; j < numc; j++) {
      uint32_t a = rc.Byte();
      a <<= 8;
      a |= rc.Byte();
      a <<= 8;
      a |= rc.Byte();
      switch (rc.Byte() & 3) {
      default:
      case 0:
        expected.push_back(a & 0x7f);
        break;
      case 1:
        expected.push_back(a & 0x7ff);
        break;
      case 2:
        expected.push_back(a & 0xffff);
        break;
      case 3:
        expected.push_back(a & 0x10ffff);
        break;
      }
    }
    expected.push_back(rep + 1);

    string str;
    for (uint32_t cp : expected)
      str += UTF8::Encode(cp);

    // Now expect the expected.
    std::vector<uint32_t> got = UTF8::Codepoints(str);

    if (expected != got) {
      printf("For codepoint sequence: ");
      for (uint32_t cp : expected) {
        printf("%04x ", cp);
      }
      printf("\n");
      printf("Got string: %s\n",
             Util::HexString(str, " ").c_str());
      printf("Decoded to: ");
      for (uint32_t cp : got) {
        printf("%04x ", cp);
      }
      printf("\n");
      LOG(FATAL) << "Wrong! Rep " << rep;
    }
  }

  // TODO: Test invalid encodings too.
}

int main(int argc, char **argv) {
  TestUnicode();

  printf("OK\n");
  return 0;
}


/* KEEP THIS LINE TOO */
