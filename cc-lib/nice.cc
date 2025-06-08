#include "nice.h"

#if defined(__MINGW32__) || defined(_WIN32)

#include <windows.h>
#undef ARRAYSIZE

void Nice::SetLowestPriority() {
  (void)SetPriorityClass(GetCurrentProcess(), IDLE_PRIORITY_CLASS);
}

void Nice::SetLowPriority() {
  (void)SetPriorityClass(GetCurrentProcess(), BELOW_NORMAL_PRIORITY_CLASS);
}

void Nice::SetHighPriority() {
  (void)SetPriorityClass(GetCurrentProcess(), ABOVE_NORMAL_PRIORITY_CLASS);
}

#elif defined(__linux__) || defined(__APPLE__) || defined(__unix__)

#include <sys/resource.h>

void Nice::SetLowestPriority() {
  (void)setpriority(PRIO_PROCESS, 0, 19);
}

void Nice::SetLowPriority() {
  (void)setpriority(PRIO_PROCESS, 0, 10);
}

void Nice::SetHighPriority() {
  // Note that this is likely to fail if not root.
  (void)setpriority(PRIO_PROCESS, 0, -10);
}

#else

// No known implementation.
void Nice::SetLowestPriority() {}
void Nice::SetLowPriority() {}
void Nice::SetHighPriority() {}


#endif

