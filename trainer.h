#pragma once

#include "topic_count.h"
#include "util.h"
#include "dict.h"

#include <string>
#include <vector>

struct Sample {
  std::vector<int> token_;
  std::vector<int> assignment_;
};

class Trainer {
public:
  void Train();
  virtual void ReadData(std::string data_file) = 0;
  virtual void TrainOneSample(Sample& data) = 0;
  virtual void PrintPerplexity() = 0;
  virtual void PrintLogLikelihood() = 0;

protected:
  // if sample-by-doc:
  //   train = document, stat = word-topic count
  // if sample-by-word:
  //   train = word (i.e., inverted index), stat = doc-topic count
  // in either case, test represents test documents
  std::vector<Sample> train_, test_;
  std::vector<TopicCount> stat_;
  std::vector<int> summary_;
  Dict  dict_;
  Timer timer_;
};

class Trainer1 : public Trainer { // sample-by-doc
public:
  void ReadData(std::string data_file);

private:
  void TrainOneSample(Sample& data);
  void PrintPerplexity();
  void PrintLogLikelihood();
};

class Trainer2 : public Trainer { // sample-by-word
public:
  void ReadData(std::string data_file);

private:
  void TrainOneSample(Sample& data);
  void PrintPerplexity();
  void PrintLogLikelihood();
};

