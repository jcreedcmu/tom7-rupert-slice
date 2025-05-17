
#include "status-bar.h"

#include <cstdio>

#include "ansi.h"

static void Test() {
  StatusBar bar(3);

  bar.EmitStatus(
      "You should not see " ARED("ANY") " of these lines persist!\n"
      "Not " ARED("this one") "...\n"
      "Nor " ARED("this one either") "!!\n");

  bar.Printf(AWHITE("This line should say forty-two: %d and be before "
                    "the status.") "\n",
             42);

  bar.Print(AYELLOW("This one should be yellow and say ") AGREEN("{}")
            AYELLOW("."), "green");

  bar.Statusf(
      "| This is the three-line " ABLUE("status bar") ".\n"
      "# this one should get overwritten!!!\n"
      "| This is the end of the status bar. " ACYAN("♥") "\n");


  bar.LineStatusf(
      1,
      // test that it strips the trailing newline
      "| It should appear below two %s lines about 42 and green.\n",
      "status");
}

static void TestIndexed() {
  StatusBar bar(10);

  bar.LineStatusf(7, "Testing that I can index into a line without "
                  "emitting first.");
}

int main(int argc, char **argv) {
  ANSI::Init();

  TestIndexed();
  printf("------\n");

  Test();

  printf("^ This test requires visual inspection\n");
  printf("OK\n");
  return 0;
}
