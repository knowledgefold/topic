#pragma once

#include "dict.h"
#include "util.h"
#include "timer.h"
#include "logger.h"
#include "reader.h"

#include <string>
#include <vector>

struct Document {
  struct Pair {
    int tok_, asg_; // token and assignment
    Pair(int tok, int asg) : tok_(tok), asg_(asg) {}
  };
  std::vector<Pair> body_;
  IArray test_nkd_; // only used in test corpus, which is small, so we can afford memory
};

struct Corpus {
  std::vector<Document> corpus_;
  int num_doc_, num_token_;

  Corpus() : num_doc_(0), num_token_(0) {}

  void ReadData(const char *data_file) {
    Timer read_timer("ReadData");
    Reader reader(data_file);
    num_token_ = reader.Read([this](char* line) {
      Document doc;
      char *ptr = strtok(line, " "); // skip first field
      int line_token = 0;
      ptr = strtok(NULL, " ");
      while (ptr != NULL) {
        char *colon = strchr(ptr, ':');
        int word_id = dict.InsertWord(std::string(ptr, colon));
        int count = strtol(colon + 1, NULL, 10);
        for (int i = 0; i < count; ++i) {
          doc.body_.emplace_back(word_id, -1);
        }
        line_token += count;
        ptr = strtok(NULL, " ");
      }
      corpus_.emplace_back(std::move(doc));
      return line_token;
    });
    num_doc_ = corpus_.size();
    lg.Printf("doc = %d, token = %d, word = %d", num_doc_, num_token_, dict.size_);
  }
};
