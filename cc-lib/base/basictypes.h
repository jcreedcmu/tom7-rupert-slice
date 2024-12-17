//
// Copyright 2001 - 2003 Google, Inc.
//
// Port note: We may be able to remove this entirely from cc-lib.

#ifndef _CC_LIB_BASE_BASICTYPES_H_
#define _CC_LIB_BASE_BASICTYPES_H_

#include "base/integral_types.h"
#include "base/casts.h"
#include "base/port.h"

// The following enum should be used only as a constructor argument to indicate
// that the variable has static storage class, and that the constructor should
// do nothing to its state.  It indicates to the reader that it is legal to
// declare a static nistance of the class, provided the constructor is given
// the base::LINKER_INITIALIZED argument.  Normally, it is unsafe to declare a
// static variable that has a constructor or a destructor because invocation
// order is undefined.  However, IF the type can be initialized by filling with
// zeroes (which the loader does for static variables), AND the type's
// destructor does nothing to the storage, then a constructor for static initialization
// can be declared as
//       explicit MyClass(base::LinkerInitialized x) {}
// and invoked as
//       static MyClass my_variable_name(base::LINKER_INITIALIZED);
namespace base {
enum LinkerInitialized { LINKER_INITIALIZED };
}

#endif  // _CC_LIB_BASE_BASICTYPES_H_
