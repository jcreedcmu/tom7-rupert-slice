// Copyright 2010 Google
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "stdio.h"
#include "time.h"

#include "base/logging.h"

namespace google_base {

DateLogger::DateLogger() {
#if defined(_MSC_VER)
  _tzset();
#endif
}

const char* DateLogger::HumanDate() {
#ifdef __MINGW32__
  return "mingw";
#else
# if defined(_MSC_VER)
  _strtime_s(buffer_, sizeof(buffer_));
  time_t time_value = time(nullptr);
  struct tm now;
  localtime_r(&time_value, &now);
  snprintf(buffer_, sizeof(buffer_), "%02d:%02d:%02d\0",
           now.tm_hour, now.tm_min, now.tm_sec);
# endif
  return buffer_;
#endif
}

}  // namespace google_base
