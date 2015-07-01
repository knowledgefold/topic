#pragma once

#include "corpus.h"
#include "sparse_count.h"

class Trainer {
public:
  void Train(); // parameter estimation on training dataset

private:
  void initialize(); // TODO: fix header, compile
  void train_one_document(Document& doc);
  real evaluate_joint();
  real evaluate_llh();
  real evaluate_test_llh();
  void test_one_document(Document& doc);
  void save_result();

private:
  Corpus train_, test_; // train/test documents
  std::vector<SparseCount> nkw_; // K x V, topic word counts
  IMAtrix test_nkw_; // K x V, topic word counts
  IArray nk_, test_nk_; // K x 1, topic counts
  EArray alpha_; // K x 1
  real alpha_sum_, beta_, beta_sum_;
  std::vector<real> iter_time_, joint_, llh_, test_llh_;
};
