// Bidirectional mapping between word string and 0-based consecutive word id.
#pragma once

#include <stdio.h>
#include <string>
#include <unordered_map>

#define D_FATAL(fmt,args...) do { \
  fprintf(stdout, fmt "\n", ##args); \
  exit(EXIT_FAILURE);  \
} while (0)

struct Dict {
  int size_ = 0;
  std::unordered_map<std::string, int> word2id_;
  std::unordered_map<int, std::string> id2word_;

  static Dict& instance() { // singleton
    static Dict e;
    return e;
  }
  
  int InsertWord(const std::string& word) { // return id if exists
    auto it = word2id_.find(word);
    if (it != word2id_.end()) { // found
      return it->second;
    } else { // new
      word2id_[word] = size_;
      id2word_[size_] = word;
      return size_++;
    }
  }

  int GetId(const std::string& word) { // error if not exists
    auto it = word2id_.find(word);
    if (it == word2id_.end()) {
      D_FATAL("word not found: %s", word.c_str());
    }
    return it->second;
  }

  std::string GetWord(int id) { // error if not exists
    auto it = id2word_.find(id);
    if (it == id2word_.end()) {
      D_FATAL("word id not found: %d", id);
    }
    return it->second;
  }
};

static auto& dict = Dict::instance();
