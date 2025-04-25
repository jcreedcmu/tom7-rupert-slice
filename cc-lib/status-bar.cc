
#include "status-bar.h"

#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

#include "base/logging.h"
#include "base/stringprintf.h"
#include "base/port.h"

#include "util.h"
#include "ansi.h"
#include "timer.h"

using namespace std;

StatusBar::StatusBar(int num_lines) : num_lines(num_lines) {
  CHECK(num_lines > 0);
  prev_status_lines.resize(num_lines);
}

void StatusBar::Printf(const char* format, ...) PRINTF_ATTRIBUTE(1, 2) {
  va_list ap;
  va_start(ap, format);
  string result;
  StringAppendV(&result, format, ap);
  va_end(ap);
  Emit(result);
}

void StatusBar::Emit(const std::string &s) {
  std::vector<std::string> lines = Util::SplitToLines(s);
  std::unique_lock<std::mutex> ml(m);
  MoveUp();
  for (const string &line : lines) {
    printf("%s\n", line.c_str());
  }
  // Maintain space for status.
  EmitStatusLinesWithLock(prev_status_lines);
}

void StatusBar::Statusf(const char* format, ...) PRINTF_ATTRIBUTE(1, 2) {
  va_list ap;
  va_start(ap, format);
  string result;
  StringAppendV(&result, format, ap);
  va_end(ap);
  EmitStatus(result);
}

void StatusBar::Progressf(int64_t numer, int64_t denom,
                          const char *format, ...) {
  va_list ap;
  va_start(ap, format);
  string result;
  StringAppendV(&result, format, ap);
  va_end(ap);
  EmitLine(num_lines - 1,
           ANSI::ProgressBar(numer, denom, result,
                             timer.Seconds()));
}

void StatusBar::EmitStatus(const std::string &s) {
  std::vector<std::string> lines = Util::SplitToLines(s);
  std::unique_lock<std::mutex> ml(m);
  prev_status_lines = lines;
  MoveUp();
  EmitStatusLinesWithLock(lines);
}

void StatusBar::EmitStatus(const std::vector<std::string> &lines) {
  std::unique_lock<std::mutex> ml(m);
  prev_status_lines = lines;
  MoveUp();
  EmitStatusLinesWithLock(lines);
}

// The idea is that we keep the screen in a state where there are
// num_lines of status at the bottom, right before the cursor. This
// is always throw-away space. When we print something other than
// status, we just pad with the number of blank lines so that the
// next call will not overwrite what we wrote.
void StatusBar::MoveUp() {
  if (!first) {
    for (int i = 0; i < num_lines; i++) {
      printf(
          // Cursor to beginning of previous line
          ANSI_PREVLINE
          // Clear line
          ANSI_CLEARLINE);
    }
  }
  first = false;
}

void StatusBar::EmitStatusLinesWithLock(const std::vector<std::string> &lines) {
  if ((int)lines.size() != num_lines) {
    printf(ARED("...wrong number of lines (have %d want %d)...") "\n",
           (int)lines.size(), num_lines);
  }
  for (const string &line : lines) {
    printf("%s\n", line.c_str());
  }
}

void StatusBar::LineStatusf(int idx, const char *format, ...) {
  va_list ap;
  va_start(ap, format);
  string result;
  StringAppendV(&result, format, ap);
  va_end(ap);
  EmitLine(idx, result);
}

void StatusBar::EmitLine(int idx, const std::string &s) {
  CHECK(idx >= 0 && idx < num_lines) << "StatusBar index out of bounds: "
                                     << idx << " vs " << num_lines;
  // Strip trailing newlines for convenience.
  std::string line = s;
  while (!line.empty() && line.back() == '\n') line.pop_back();

  std::unique_lock<std::mutex> ml(m);
  prev_status_lines[idx] = std::move(line);
  MoveUp();
  EmitStatusLinesWithLock(prev_status_lines);
}

void StatusBar::Clear() {
  EmitStatus(std::string(num_lines, '\n'));
}
