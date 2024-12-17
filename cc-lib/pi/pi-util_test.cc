
#include "pi-util.h"

#include <optional>
#include <cstdint>
#include <cstdio>

#include "base/logging.h"

using namespace std;

int main(int argc, char **argv) {

  optional<uint64_t> so = PiUtil::GetSerial();
  CHECK(so.has_value()) << "No serial?";

  printf("%016llx\n", so.value());

  return 0;
}
