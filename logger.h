// A simple Golang-like logger.
//
// Usage:
//   #include "logger.h"
//   ...
//   int s = 1 + 2;
//   lg.Printf("1 + 2 = %d", s);
//   if (s != 3) {
//     lg.Fatalf("Check failed: %d != 3", s);
//   }
//
// Note:
// - Header only contains datetime.
// - Appends '\n'.
// - Message should be less than MAX_LOG_SIZE bytes, otherwise undefined
#pragma once

#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <unistd.h>

#define MAX_LOG_SIZE 1024

struct Logger {
  char buf_[MAX_LOG_SIZE];

  static Logger& instance() { // singleton
    static Logger e;
    return e;
  }

  void Printf(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    Format(STDOUT_FILENO, fmt, ap);
    va_end(ap);
  }

  void Fatalf(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    Format(STDOUT_FILENO, fmt, ap);
    va_end(ap);
    exit(EXIT_FAILURE);
  }

  void Format(int fd, const char *fmt, va_list ap) {
    time_t time_since_epoch = time(NULL);
    struct tm* tm_info = localtime(&time_since_epoch);
    if (tm_info == NULL) {
      perror("localtime");
      exit(EXIT_FAILURE);
    }
    strftime(buf_, MAX_LOG_SIZE, "%Y/%m/%d %H:%M:%S ", tm_info);
    vsprintf(buf_ + strlen(buf_), fmt, ap);
    strcat(buf_, "\n");
    if (write(fd, buf_, strlen(buf_)) < 0) {
      perror("write() syscall");
      exit(EXIT_FAILURE);
    }
  }
};

static auto& lg = Logger::instance();
