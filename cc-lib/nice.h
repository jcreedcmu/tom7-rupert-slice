
#ifndef _CC_LIB_NICE_H
#define _CC_LIB_NICE_H

struct Nice {
  // Unspecified semantics. Might do nothing if not implemented
  // on the current platform, or e.g. requires privileges that
  // you don't have.
  static void SetLowestPriority();
  static void SetLowPriority();
  static void SetHighPriority();

 private:
  Nice() = delete;
};

#endif
