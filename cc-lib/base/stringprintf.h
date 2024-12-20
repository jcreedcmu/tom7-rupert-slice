// Copyright 2002 and onwards Google Inc.
//
// Printf variants that place their output in a C++ string.
//
// Usage:
//      string result = StringPrintf("%d %s\n", 10, "hello");
//      SStringPrintf(&result, "%d %s\n", 10, "hello");
//      StringAppendF(&result, "%d %s\n", 20, "there");

#ifndef _CC_LIB_BASE_STRINGPRINTF_H
#define _CC_LIB_BASE_STRINGPRINTF_H

#include <stdarg.h>
#include <string>

#include "base/port.h"

// Return a C++ string.
extern std::string StringPrintf(const char* format, ...)
    // Tell the compiler to do printf format string checking.
    PRINTF_ATTRIBUTE(1,2);

// Store result into a supplied string and return it.
extern const std::string& SStringPrintf(std::string* dst,
                                        const char* format, ...)
    // Tell the compiler to do printf format string checking.
    PRINTF_ATTRIBUTE(2,3);

// Append result to a supplied string.
extern void StringAppendF(std::string* dst, const char* format, ...)
    // Tell the compiler to do printf format string checking.
    PRINTF_ATTRIBUTE(2,3);

// Lower-level routine that takes a va_list and appends to a specified
// string.  All other routines are just convenience wrappers around it.
extern void StringAppendV(std::string* dst, const char* format, va_list ap);

#endif
