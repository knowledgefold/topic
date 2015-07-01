// A simple RAII Timer. 
//
// Usage:
//   {
//     Timer timer("LoadData");
//     ...
//   }
// 
// Note:
// - Use Get() to behave like a traditional non-RAII timer.
#pragma once

#include "logger.h"

#include <time.h>
#include <stdarg.h>

#define MAX_NAME_SIZE 128

// Get monotonic time in seconds from a starting point
static double get_time() {
  struct timespec start;
  clock_gettime(CLOCK_MONOTONIC, &start);
  return (start.tv_sec + start.tv_nsec/1e+9);
}

struct Timer {
  bool dying_msg_;
  char name_[MAX_NAME_SIZE];
  double start_;

  Timer(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(name_, MAX_NAME_SIZE, fmt, ap);
    va_end(ap);
    dying_msg_ = true;
    start_ = get_time();
  }

  ~Timer() {
    if (dying_msg_) {
      lg.Printf("%s took %6.4lf sec", name_, get_time() - start_);
    }
  }

  double Get() {
    dying_msg_ = false;
    return get_time() - start_;
  }
};
