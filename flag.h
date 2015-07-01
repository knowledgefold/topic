// A simple Golang-like flag parser. Supports bool, int, float, and std::string.
//
// Usage (source file):
//   #include "flag.h"
//   ...
//   auto *b = flag.Bool("b", false, "bool flag");
//   auto *num = flag.Int("num", 10, "int flag");
//   auto *eps = flag.Float("eps", 1e-2, "float flag");
//   auto *st = flag.String("st", "hello", "string flag");
//   ...
//   int main(int argc, char **argv) {
//    flag.Parse(argc, argv);
//    ...
//    printf("%d, %d, %f, %s\n", *b, *num, *eps, st->c_str());
//   }
//
// Usage (command line):
//   ./a.out -h
//   ./a.out -help
//   ./a.out -b true -num 0x1234 -eps -0.0001 -st gogogo
//
// Note:
// - Does not support flags across multiple files, like DECLARE_* in gflags.
// - Bool flags only take value in {true,false}, not 0/1/yes/no/etc.
// - Flags can only be passed through "-flag", not "--flag".
// - Only allow "-flag arg", not "-flag=arg".
// - Flags "-h" and "-help" are reserved, so cannot be defined
#pragma once

#include <map>
#include <string>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define F_VALUE_AS(T,val) (*reinterpret_cast<T*>(val))
#define F_COPY_AS(T,dest,src) do { dest = new T(F_VALUE_AS(T,src)); } while (0)
#define F_SET_AS(T,dest,src) do { F_VALUE_AS(T,dest) = (src); } while (0)
#define F_FATAL(fmt,args...) do { \
  fprintf(stdout, fmt "\n", ##args); \
  exit(EXIT_FAILURE);  \
} while (0)

enum FlagType { F_BOOL, F_INT, F_FLOAT, F_STRING };

struct FlagValue {
  FlagType type_;
  void *buf_;

  FlagValue(FlagType type, void *value) : type_(type) { // copy value to buf
    switch (type_) {
      case F_BOOL:   F_COPY_AS(bool, buf_, value); break;
      case F_INT:    F_COPY_AS(int, buf_, value); break;
      case F_FLOAT:  F_COPY_AS(float, buf_, value); break;
      case F_STRING: F_COPY_AS(std::string, buf_, value); break;
      default: F_FATAL("unkown type");
    }
  }

  ~FlagValue() { // release buf appropriately
    switch (type_) {
      case F_BOOL:   delete reinterpret_cast<bool*>(buf_); break;
      case F_INT:    delete reinterpret_cast<int*>(buf_); break;
      case F_FLOAT:  delete reinterpret_cast<float*>(buf_); break;
      case F_STRING: delete reinterpret_cast<std::string*>(buf_); break;
      default: F_FATAL("unkown type");
    }
  }

  void Set(const char* value) { // set buf from cmd line inputs
    char *end;
    int base = (value[0] == '0' and value[1] == 'x') ? 16 : 10;
    errno = 0;
    switch (type_) {
      case F_BOOL: { // only allow "true" and "false"
        if (strcmp(value, "true") == 0) {
          F_SET_AS(bool, buf_, true);
        } else if (strcmp(value, "false") == 0) {
          F_SET_AS(bool, buf_, false);
        } else {
          F_FATAL("invalid value: %s", value);
        }
        break;
      }
      case F_INT: { // allow both dec and hex
        int r = strtol(value, &end, base);
        if (errno or end != value + strlen(value)) {
          F_FATAL("invalid value: %s", value);
        }
        F_SET_AS(int, buf_, r);
        break;
      }
      case F_FLOAT: {
        float r = strtof(value, &end);
        if (errno or end != value + strlen(value)) {
          F_FATAL("invalid value: %s", value);
        }
        F_SET_AS(float, buf_, r);
        break;
      }
      case F_STRING: {
        F_SET_AS(std::string, buf_, value);
        break;
      }
      default: F_FATAL("unkown type");
    }
  }

  std::string String() { // put a pretty quote around strings
    char b[64];
    switch (type_) {
      case F_BOOL:   return F_VALUE_AS(bool, buf_) ? "true" : "false";
      case F_INT:    snprintf(b, 64, "%d", F_VALUE_AS(int, buf_)); return b;
      case F_FLOAT:  snprintf(b, 64, "%g", F_VALUE_AS(float, buf_)); return b;
      case F_STRING: return "\"" + F_VALUE_AS(std::string, buf_) + "\"";
      default: F_FATAL("unkown type");
    }
  }
};

struct Flag {
  const char *name_;
  const char *usage_;
  FlagValue value_;

  Flag(const char *name, const char *usage, FlagType type, void *value)
    : name_(name), usage_(usage), value_(type, value) {}
};

struct StrComp {
  bool operator() (const char *s1, const char *s2) {
    return (strcmp(s1, s2) < 0);
  }
};

struct FlagSet {
  const char *prog_; // argv[0]
  std::map<const char*, Flag*, StrComp> body_; // map[name]Flag*

  static FlagSet& instance() { // singleton
    static FlagSet e;
    return e;
  }

  ~FlagSet() {
    for (auto& it : body_) {
      delete it.second;
    }
  }

  void Parse(int argc, char **argv) {
    prog_ = argv[0];
    int i = 1;
    while (i < argc) {
      const char *s = argv[i];
      if (strlen(s) <= 1 or s[0] != '-') { // only allow "-flag"
        F_FATAL("bad flag syntax: %s", s);
      }
      const char *name = s + 1;
      if (strcmp(name, "h") == 0 or strcmp(name, "help") == 0) { // help msg
        Usage(); // exits here
      }
      auto it = body_.find(name);
      if (it == body_.end()) { // new name
        F_FATAL("flag undefined: %s", name);
      }
      ++i;
      if (i == argc) {
        F_FATAL("flag needs an argument: -%s", name);
      }
      Flag *f = it->second;
      f->value_.Set(argv[i]);
      ++i;
    }
  }

  bool* Bool(const char *name, bool value, const char *usage) {
    Var(name, usage, F_BOOL, (void*)&value);
    return reinterpret_cast<bool*>(body_[name]->value_.buf_);
  }

  int* Int(const char *name, int value, const char *usage) {
    Var(name, usage, F_INT, (void*)&value);
    return reinterpret_cast<int*>(body_[name]->value_.buf_);
  }

  float* Float(const char *name, float value, const char *usage) {
    Var(name, usage, F_FLOAT, (void*)&value);
    return reinterpret_cast<float*>(body_[name]->value_.buf_);
  }

  std::string* String(const char *name, std::string value, const char *usage) {
    Var(name, usage, F_STRING, (void*)&value);
    return reinterpret_cast<std::string*>(body_[name]->value_.buf_);
  }

  // Define a flag
  void Var(const char *name, const char *usage, FlagType type, void *value) {
    if (strcmp(name, "h") == 0 or strcmp(name, "help") == 0) {
      F_FATAL("flag reserved: %s", name);
    }
    auto it = body_.find(name);
    if (it != body_.end()) { // existing name
      F_FATAL("flag redefined: %s", name);
    }
    Flag *f = new Flag(name, usage, type, value);
    body_.emplace(name, f);
  }

  void Print() { // print current values, should be used after Parse()
    fprintf(stdout, "Flags of %s:\n", prog_);
    for (auto& it : body_) {
      Flag *f = it.second;
      fprintf(stdout, "  -%s=%s\n", f->name_, f->value_.String().c_str());
    }
  }

  void Usage() { // called when -h or -help is given
    fprintf(stdout, "Usage of %s:\n", prog_);
    for (auto& it : body_) {
      Flag *f = it.second;
      fprintf(stdout, "  -%s=%s: %s\n", f->name_, f->value_.String().c_str(), f->usage_);
    }
    exit(EXIT_SUCCESS);
  }
};

static auto& flag = FlagSet::instance();
