
#include "base/logging.h"
#include "base/stringprintf.h"

static void TestPrintf() {
  std::string s = StringPrintf("Hello %s", "world");
  StringAppendF(&s, "%d", 0x2A);
  CHECK(s == "Hello world42");
}

static void TestFormat() {
  #if __cplusplus >= 202002L
  std::string s;
  AppendFormat(&s, "He{} {}d", "llo", "worl");
  AppendFormat(&s, "{}", 0x2A);
  CHECK(s == "Hello world42");
  #endif
}

int main(int argc, char **argv) {

  TestPrintf();
  TestFormat();

  printf("OK\n");
  return 0;
}
