
#ifndef _RUPERTS_RUN_Z3_H
#define _RUPERTS_RUN_Z3_H

#include <optional>
#include <string_view>

enum class Z3Result {
  SAT,
  UNSAT,
  UNKNOWN,
};

Z3Result RunZ3(std::string_view content,
               std::optional<double> timeout_seconds = {});

#endif
