#pragma once

#include <glog/logging.h>
#include <gflags/gflags.h>

#include <time.h>
#include <math.h>
#include <ctype.h>
#include <stdlib.h>

// Container hack
#define RANGE(x) ((x).begin()), ((x).end())
#define SUM(x)   (std::accumulate(RANGE(x), .0))

// Random hack
#include <chrono>
#include <random>
#define CLOCK (std::chrono::system_clock::now().time_since_epoch().count())
static std::mt19937 _rng(CLOCK);
static std::uniform_real_distribution<double> _unif01;
static std::normal_distribution<double> _stdnormal;

// Google Log hack
#define LI LOG(INFO)
#define LD DLOG(INFO)

// Get monotonic time in seconds from a starting point
static double get_time() {
  struct timespec start;
  clock_gettime(CLOCK_MONOTONIC, &start);
  return (start.tv_sec + start.tv_nsec/1000000000.0);
}

class Timer {
public:
  void   tic() { start_ = get_time(); }
  double toc() { double ret = get_time() - start_; time_ += ret; return ret; }
  double get() { return time_; }
private:
  double time_  = .0;
  double start_ = get_time();
};

// Google flags hack
static void print_help() {
  fprintf(stderr, "Program Flags:\n");
  std::vector<google::CommandLineFlagInfo> all_flags;
  google::GetAllFlags(&all_flags);
  for (const auto& flag : all_flags) {
    if (flag.filename.find("src/") != 0) // HACK: filter out built-in flags
      fprintf(stderr,
              "-%s: %s (%s, default:%s)\n",
              flag.name.c_str(),
              flag.description.c_str(),
              flag.type.c_str(),
              flag.default_value.c_str());
  }
  exit(1);
}

// Google flags hack
static void print_flags() {
  LI << "---------------------------------------------------------------------";
  std::vector<google::CommandLineFlagInfo> all_flags;
  google::GetAllFlags(&all_flags);
  for (const auto& flag : all_flags) {
    if (flag.filename.find("src/") != 0) // HACK: filter out built-in flags
      LI << flag.name << ": " << flag.current_value;
  }
  LI << "---------------------------------------------------------------------";
}

// Faster strtol without error checking.
static long int strtol(const char *nptr, char **endptr) {
  // Skip spaces
  while (isspace(*nptr)) ++nptr;
  // Sign
  bool is_negative = false;
  if (*nptr == '-') {
    is_negative = true;
    ++nptr;
  } else if (*nptr == '+') {
    ++nptr;
  }
  // Go!
  long int res = 0;
  while (isdigit(*nptr)) {
    res = (res * 10) + (*nptr - '0');
    ++nptr;
  }
  if (endptr != NULL) *endptr = (char *)nptr;
  if (is_negative) return -res;
  return res;
}

