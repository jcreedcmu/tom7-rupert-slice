// It is very common for console apps to report some
// status using ANSI color etc., and to want to update
// that (like a progress bar). If the program also
// outputs other stuff, then we get ugly duplicated
// progress bars or messages get overwritten. This
// manages I/O so that it comes out cleanly.

#ifndef _CC_LIB_STATUS_BAR_H
#define _CC_LIB_STATUS_BAR_H

#include <string>
#include <vector>
#include <mutex>

#include "base/port.h"
#include "timer.h"

// Thread safe.
struct StatusBar {
  // Give the fixed number of lines that the status bar uses.
  // Must be greater than zero.
  explicit StatusBar(int num_lines);

  // Each of these immediately outputs to the screen.

  // Prints lines above the status bar. Adds trailing newline if not present.
  void Printf(const char *format, ...) PRINTF_ATTRIBUTE_MEMBER(1, 2);

  // Prints lines above the status bar. Adds trailing newline if not present.
  void Emit(const std::string &s);

  // Update the status bar. This should be done in one call that
  // contains num_lines lines. Trailing newline not necessary.
  void Statusf(const char *format, ...) PRINTF_ATTRIBUTE_MEMBER(1, 2);

  // Update the status bar with a string, which should contain num_lines
  // lines.
  void EmitStatus(const std::string &s);
  void EmitStatus(const std::vector<std::string> &lines);

  // Output an ANSI progress indicator in the last line of the status.
  // This is a convenience method since it is a very common use.
  // Uses the time since the status bar object was created. If you
  // want something else, just call Emit(ANSI::ProgressBar(...)).
  void Progressf(int64_t numer, int64_t denom, const char *format, ...)
  PRINTF_ATTRIBUTE_MEMBER(3, 4);

  // TODO: Finish(), which replaces the final line of status with
  // "complete" progress bar, giving the total time?

  // Update a particular line of the status bar. The index must be
  // in [0, num_lines). Immediately outputs the entire status bar, so
  // you should prefer one of the above routines if you are building the
  // entire bar.
  void LineStatusf(int idx, const char *format, ...) PRINTF_ATTRIBUTE_MEMBER(2, 3);
  void EmitLine(int idx, const std::string &s);

  // Set every status line empty; keeps any lines above.
  void Clear();

 private:
  void MoveUp();

  void EmitStatusLinesWithLock(
      const std::vector<std::string> &lines);

  std::mutex m;
  int num_lines = 0;
  bool first = true;
  // Always num_lines in length.
  std::vector<std::string> prev_status_lines;
  Timer timer;
};



#endif
