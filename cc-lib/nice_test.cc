
#include "nice.h"

#include <cstdio>

// We can't really test whether this did anything (and it is not
// guaranteed to), but we can at least make sure it links and
// doesn't crash.
int main(int argc, char **argv) {
  Nice::SetLowestPriority();
  Nice::SetLowPriority();
  Nice::SetHighPriority();

  printf("OK\n");
  return 0;
}
