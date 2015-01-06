#include "trainer.h"

#include <math.h>
#include <stdio.h>
#include <list>
#include <algorithm>

const int EVAL_ITER = 100;

DEFINE_double(alpha, 1.0, "Parameter of prior on doc-topic distribution");
DEFINE_double(beta, 0.01, "Parameter of prior on topic-word distribution");
DEFINE_double(test_ratio, 0.1, "Heldout ratio, i.e. test/all");
DEFINE_int32(num_iter, 10, "Number of training iteration");
DEFINE_int32(num_topic, 100, "Model size, usually called K");
DEFINE_int32(eval_every, 1, "Evaluate the model every N iterations");

void Trainer::Train() {
  // Init
  timer_.tic();
  summary_.resize(FLAGS_num_topic);
  for (auto& sample : train_) {
    for (const auto idx : sample.token_) {
      int rand_topic = _unif01(_rng) * FLAGS_num_topic;
      sample.assignment_.push_back(rand_topic);
      stat_[idx].AddCount(rand_topic);
      ++summary_[rand_topic];
    }
  } // end of for each sample
  auto init_time = timer_.toc();
  double num_million_token = SUM(summary_) / 1e+6;
  LW << "Initialization took " << init_time << " sec, throughput: "
     << num_million_token / init_time << " (M token/sec)";
  PrintPerplexity();
  PrintLogLikelihood();

  // Go!
  for (int iter = 1; iter <= FLAGS_num_iter; ++iter) {
    timer_.tic();
    for (auto& sample : train_) TrainOneSample(sample);
    auto iter_time = timer_.toc();
    LW << "Iteration " << iter << ", Elapsed time: " << timer_.get() << " sec, "
       << "throughput: " << num_million_token / iter_time << " (M token/sec)";
    if (iter==1 or iter==FLAGS_num_iter or iter%FLAGS_eval_every==0) {
      PrintPerplexity();
      PrintLogLikelihood();
    }
  } // end of for every iter

  LI << "--------------------------------------------------------------";
  LW << "Training Time: " << timer_.get() << " sec";
}

// -------------------------------------------------------------------------- //

void Trainer1::ReadData(std::string data_file) {
  size_t num_token = 0;
  FILE *data_fp = fopen(data_file.c_str(), "r"); CHECK_NOTNULL(data_fp);
  char *line = NULL; size_t num_byte;
  while (getline(&line, &num_byte, data_fp) != -1) {
    Sample doc;
    char *ptr = line;
    while (*ptr != '\n') {
      char *colon = strchr(ptr, ':'); // at colon
      ptr = colon; while (*ptr != ' ') --ptr; // ptr at space before colon
      int word_id = dict_.insert_word(std::string(ptr+1, colon));
      ptr = colon;
      int count = strtol(++ptr, &ptr); // ptr at space or \n
      for (int i = 0; i < count; ++i)
        doc.token_.push_back(word_id);
      num_token += count;
    }
    train_.emplace_back(std::move(doc));
  }
  fclose(data_fp);
  free(line);

  // Separate test set from training set
  std::shuffle(RANGE(train_), _rng);
  CHECK_LT(FLAGS_test_ratio, 1.0);
  int test_size = (int)(train_.size() * FLAGS_test_ratio);
  test_.assign(train_.end() - test_size, train_.end());
  train_.resize(train_.size() - test_size);
  stat_.resize(dict_.size());

  LW << "num doc (train): " << train_.size();
  LW << "num doc (test): " << test_.size();
  LW << "num word (total): " << dict_.size(); 
  LW << "num token (total): " << num_token;
  LI << "--------------------------------------------------------------";
}

void Trainer1::TrainOneSample(Sample& doc) {
  double alpha_beta = FLAGS_alpha * FLAGS_beta;
  double beta_sum = FLAGS_beta * stat_.size();

  // Too much allocation, but I believe in tcmalloc...
  // Construct doc topic count on the fly to save memory
  std::vector<int> doc_topic_count(FLAGS_num_topic, 0);
  for (const auto topic : doc.assignment_) ++doc_topic_count[topic];

  // Compute per doc cached values
  // Abar and Ccoeff can be computed outside, but it's not elegant :P
  double Abar = .0;
  double Bbar = .0;
  std::vector<double> Ccoeff(FLAGS_num_topic, .0);
  std::list<int> doc_topic_index; // exploit sparsity and get O(1) insert/erase
  for (int k = 0; k < FLAGS_num_topic; ++k) {
    double denom = summary_[k] + beta_sum;
    Abar += alpha_beta / denom;
    int count = doc_topic_count[k];
    if (count != 0) {
      doc_topic_index.push_back(k);
      Bbar += FLAGS_beta * count / denom;
      Ccoeff[k] = (FLAGS_alpha + count) / denom;
    } else {
      Ccoeff[k] = FLAGS_alpha / denom;
    }
  } // end of preparation

  // Go!
  std::vector<double> Cval(FLAGS_num_topic, .0); // only access first x entries
  for (size_t n = 0; n < doc.token_.size(); ++n) {
    // Localize
    int old_topic = doc.assignment_[n];
    auto& word = stat_[doc.token_[n]];

    // Decrement
    double denom = summary_[old_topic] + beta_sum;
    int count = doc_topic_count[old_topic];
    Abar -= alpha_beta / denom;
    Bbar -= FLAGS_beta * count / denom;
    --doc_topic_count[old_topic];
    --count;
    if (count == 0) { // shrink index
      auto pos = std::lower_bound(RANGE(doc_topic_index), old_topic);
      doc_topic_index.erase(pos);
    }
    --summary_[old_topic];
    --denom;
    Abar += alpha_beta / denom;
    Bbar += FLAGS_beta * count / denom;
    Ccoeff[old_topic] = (FLAGS_alpha + count) / denom;

    // Taking advantage of sparsity
    double Cbar = .0;
    for (size_t i = 0; i < word.item_.size(); ++i) {
      auto pair = word.item_[i];
      auto cnt = (pair.top_ == old_topic) ? pair.cnt_ - 1 : pair.cnt_;
      double val = Ccoeff[pair.top_] * cnt;
      Cval[i] = val;
      Cbar += val;
    }

    // Sample
    double sample = _unif01(_rng) * (Abar + Bbar + Cbar);
    int new_topic = -1;
    if (sample < Cbar) {
      new_topic = word.item_.back().top_; // item_ shouldn't be empty
      for (size_t i = 0; i < word.item_.size() - 1; ++i) {
        sample -= Cval[i];
        if (sample <= .0) { new_topic = word.item_[i].top_; break; }
      }
    } // end of C bucket
    else {
      sample -= Cbar;
      if (sample < Bbar) {
        sample /= FLAGS_beta;
        for (const auto top : doc_topic_index) {
          sample -= doc_topic_count[top] / (summary_[top] + beta_sum);
          if (sample <= .0) { new_topic = top; break; }
        }
      } // end of B bucket
      else {
        sample -= Bbar;
        sample /= alpha_beta;
        new_topic = FLAGS_num_topic - 1;
        for (int k = 0; k < FLAGS_num_topic - 1; ++k) {
          sample -= 1.0 / (summary_[k] + beta_sum);
          if (sample <= .0) { new_topic = k; break; }
        }
      } // end of A bucket
    } // end of choosing bucket

    CHECK_GE(new_topic, 0);
    CHECK_LT(new_topic, FLAGS_num_topic);

    // Increment
    count = doc_topic_count[new_topic];
    denom = summary_[new_topic] + beta_sum;
    Abar -= alpha_beta / denom;
    Bbar -= FLAGS_beta * count / denom;
    ++doc_topic_count[new_topic];
    ++count;
    if (count == 1) { // augment index
      auto pos = std::lower_bound(RANGE(doc_topic_index), new_topic);
      doc_topic_index.insert(pos, new_topic);
    }
    ++summary_[new_topic];
    ++denom;
    Abar += alpha_beta / denom;
    Bbar += FLAGS_beta * count / denom;
    Ccoeff[new_topic] = (FLAGS_alpha + count) / denom;

    // Set
    doc.assignment_[n] = new_topic;
    word.UpdateCount(old_topic, new_topic);
  } // end of iter over tokens
}

void Trainer1::PrintLogLikelihood() {
  double alpha_sum = FLAGS_alpha * FLAGS_num_topic;
  double beta_sum = FLAGS_beta * stat_.size();

  double doc_loglikelihood = .0;
  double nonzero_doc_topic = 0;
  for (const auto& doc : train_) {
    std::vector<int> doc_topic_count(FLAGS_num_topic, 0);
    for (const auto topic : doc.assignment_)
      ++doc_topic_count[topic]; // Reconstruct doc stats
    for (int k = 0; k < FLAGS_num_topic; ++k) {
      int count = doc_topic_count[k];
      if (count != 0) {
        ++nonzero_doc_topic;
        doc_loglikelihood += lgamma(count + FLAGS_alpha);
      }
    }
    doc_loglikelihood -= lgamma(doc.token_.size() + alpha_sum);
  }
  doc_loglikelihood -= nonzero_doc_topic * lgamma(FLAGS_alpha);
  doc_loglikelihood += train_.size() * lgamma(alpha_sum);

  double model_loglikelihood = .0;
  for (const auto count : summary_) {
    model_loglikelihood -= lgamma(count + beta_sum);
    model_loglikelihood += lgamma(beta_sum);
  }
  double nonzero_word_topic = 0;
  for (const auto& word : stat_) {
    nonzero_word_topic += word.item_.size();
    for (const auto& pair : word.item_)
      model_loglikelihood += lgamma(pair.cnt_ + FLAGS_beta);
  }
  model_loglikelihood -= nonzero_word_topic * lgamma(FLAGS_beta);

  LI << "nonzero: doc,word = "
     << nonzero_doc_topic / train_.size() << "/" << FLAGS_num_topic  << ", "
     << nonzero_word_topic / stat_.size() << "/" << FLAGS_num_topic;
  LI << "loglikelihood: doc,model,total = "
     << doc_loglikelihood << ", "
     << model_loglikelihood << ", "
     << doc_loglikelihood + model_loglikelihood;
}

void Trainer1::PrintPerplexity() {
  double beta_sum = FLAGS_beta * stat_.size();
  std::vector<int> doc_topic_count(FLAGS_num_topic);
  std::vector<double> prob(FLAGS_num_topic);
  std::vector<double> theta(FLAGS_num_topic);

  // Cache phi
  std::vector< std::vector<double>> phi;
  for (const auto& word : stat_) {
    std::vector<double> phi_w(FLAGS_num_topic, .0);
    for (const auto& pair : word.item_)
      phi_w[pair.top_] = pair.cnt_;
    for (int k = 0; k < FLAGS_num_topic; ++k)
      phi_w[k] = (phi_w[k] + FLAGS_beta) / (summary_[k] + beta_sum);
    phi.emplace_back(phi_w);
  } // end of each word

  double numer = .0, denom = .0;
  for (auto& doc : test_) {
    // Initialize to most probable topic assignments
    doc.assignment_.clear();
    for (const auto word_id : doc.token_) {
      int most_probable_topic = (stat_[word_id].item_.empty())
                                ? (_unif01(_rng) * FLAGS_num_topic) // OOV
                                : (stat_[word_id].item_[0].top_);
      doc.assignment_.push_back(most_probable_topic);
    }

    // Construct doc topic count on the fly
    std::fill(RANGE(doc_topic_count), 0);
    for (const auto topic : doc.assignment_) ++doc_topic_count[topic];

    // Perform Gibbs sampling to obtain an estimate of theta
    for (int iter = 1; iter <= EVAL_ITER; ++iter) {
      for (size_t n = 0; n < doc.token_.size(); ++n) {
        // Localize
        int old_topic = doc.assignment_[n];
        const auto& phi_w = phi[doc.token_[n]];
        // Decrement
        --doc_topic_count[old_topic];
        // Compute prob
        std::fill(RANGE(prob), .0);
        for (int k = 0; k < FLAGS_num_topic; ++k) {
          prob[k] = ((k == 0) ? 0 : prob[k-1])
                    + phi_w[k] * (doc_topic_count[k] + FLAGS_alpha);
        }
        // Sample
        int new_topic = -1;
        double sample = _unif01(_rng) * prob[FLAGS_num_topic - 1];
        for (new_topic = 0; prob[new_topic] < sample; ++new_topic);
        CHECK_GE(new_topic, 0);
        CHECK_LT(new_topic, FLAGS_num_topic);
        // Increment
        ++doc_topic_count[new_topic];
        // Set
        doc.assignment_[n] = new_topic;
      } // end of for each n
    } // end of iter

    // Compute theta
    std::fill(RANGE(theta), .0);
    double theta_denom = doc.token_.size() + FLAGS_alpha * FLAGS_num_topic;
    for (int k = 0; k < FLAGS_num_topic; ++k) {
      theta[k] = (doc_topic_count[k] + FLAGS_alpha) / theta_denom;
    }

    // Compute numer for one doc
    for (size_t n = 0; n < doc.token_.size(); ++n) {
      double lhood = .0;
      const auto& phi_w = phi[doc.token_[n]];
      for (int k = 0; k < FLAGS_num_topic; ++k) lhood += phi_w[k] * theta[k];
      numer += log(lhood);
    } // end of each n

    denom += doc.token_.size();
  } // end of for each doc

  LI << "Perplexity: " << exp(- numer / denom);
}

// -------------------------------------------------------------------------- //

void Trainer2::ReadData(std::string data_file) {
  size_t num_token = 0;
  FILE *data_fp = fopen(data_file.c_str(), "r"); CHECK_NOTNULL(data_fp);
  char *line = NULL; size_t num_byte;
  while (getline(&line, &num_byte, data_fp) != -1) {
    Sample doc;
    char *ptr = line;
    while (*ptr != '\n') {
      char *colon = strchr(ptr, ':'); // at colon
      ptr = colon; while (*ptr != ' ') --ptr; // ptr at space before colon
      int word_id = dict_.insert_word(std::string(ptr+1, colon));
      ptr = colon;
      int count = strtol(++ptr, &ptr); // ptr at space or \n
      for (int i = 0; i < count; ++i)
        doc.token_.push_back(word_id);
      num_token += count;
    }
    test_.emplace_back(std::move(doc)); // reason in next paragraph
  }
  fclose(data_fp);
  free(line);

  // Construct inverted index from docs, while separating out test set
  std::shuffle(RANGE(test_), _rng);
  CHECK_LT(FLAGS_test_ratio, 1.0);
  int test_size = (int)(test_.size() * FLAGS_test_ratio);
  train_.resize(dict_.size());
  for (size_t doc_id = 0; doc_id < test_.size() - test_size; ++doc_id) {
    for (const auto word_id : test_[doc_id].token_) {
      train_[word_id].token_.push_back(doc_id);
    }
  }
  stat_.resize(test_.size() - test_size);
  test_.erase(test_.begin(), test_.end() - test_size);

  LW << "num doc (train): " << stat_.size();
  LW << "num doc (test): " << test_.size();
  LW << "num word (total): " << dict_.size(); 
  LW << "num token (total): " << num_token;
  LI << "--------------------------------------------------------------";
}

void Trainer2::TrainOneSample(Sample& word) {
  double beta_sum = FLAGS_beta * train_.size();

  // Construct word topic count on the fly to save memory
  std::vector<int> word_topic_count(FLAGS_num_topic, 0);
  for (const auto topic : word.assignment_) ++word_topic_count[topic];

  // Compute per word cached values
  double Xbar = .0;
  std::vector<double> phi_w(FLAGS_num_topic, .0);
  for (int k = 0; k < FLAGS_num_topic; ++k) {
    phi_w[k] = (word_topic_count[k] + FLAGS_beta) / (summary_[k] + beta_sum);
    Xbar += phi_w[k];
  }
  Xbar *= FLAGS_alpha;

  // Go!
  std::vector<double> Yval(FLAGS_num_topic, .0); // only access first x entries
  for (size_t n = 0; n < word.token_.size(); ++n) {
    // Localize
    int old_topic = word.assignment_[n];
    auto& doc = stat_[word.token_[n]];

    // Decrement
    Xbar -= FLAGS_alpha * phi_w[old_topic];
    --word_topic_count[old_topic];
    --summary_[old_topic];
    phi_w[old_topic] = (word_topic_count[old_topic] + FLAGS_beta)
                       / (summary_[old_topic] + beta_sum);
    Xbar += FLAGS_alpha * phi_w[old_topic];

    // Taking advantage of sparsity
    double Ybar = .0;
    for (size_t i = 0; i < doc.item_.size(); ++i) {
      auto pair = doc.item_[i];
      auto cnt = (pair.top_ == old_topic) ? (pair.cnt_ - 1) : pair.cnt_;
      double val = phi_w[pair.top_] * cnt;
      Yval[i] = val;
      Ybar += val;
    }

    // Sample
    double sample = _unif01(_rng) * (Xbar + Ybar);
    int new_topic = -1;
    if (sample < Ybar) {
      new_topic = doc.item_.back().top_; // item shouldn't be empty
      for (size_t i = 0; i < doc.item_.size() - 1; ++i) {
        sample -= Yval[i];
        if (sample <= .0) { new_topic = doc.item_[i].top_; break; }
      }
    } // end of Y bucket
    else {
      sample -= Ybar;
      sample /= FLAGS_alpha;
      new_topic = FLAGS_num_topic - 1;
      for (int k = 0; k < FLAGS_num_topic - 1; ++k) {
        sample -= phi_w[k];
        if (sample <= .0) { new_topic = k; break; }
      }
    } // end of choosing bucket

    CHECK_GE(new_topic, 0);
    CHECK_LT(new_topic, FLAGS_num_topic);

    // Increment
    Xbar -= FLAGS_alpha * phi_w[new_topic];
    ++word_topic_count[new_topic];
    ++summary_[new_topic];
    phi_w[new_topic] = (word_topic_count[new_topic] + FLAGS_beta)
                       / (summary_[new_topic] + beta_sum);
    Xbar += FLAGS_alpha * phi_w[new_topic];

    // Set
    word.assignment_[n] = new_topic;
    doc.UpdateCount(old_topic, new_topic);
  } // end of iter over tokens
}

void Trainer2::PrintLogLikelihood() {
  double alpha_sum = FLAGS_alpha * FLAGS_num_topic;
  double beta_sum = FLAGS_beta * train_.size();

  double doc_loglikelihood = .0;
  double nonzero_doc_topic = 0;
  for (const auto& doc : stat_) {
    nonzero_doc_topic += doc.item_.size();
    int len = 0;
    for (const auto& pair : doc.item_) {
      doc_loglikelihood += lgamma(pair.cnt_ + FLAGS_alpha);
      len += pair.cnt_;
    }
    doc_loglikelihood -= lgamma(len + alpha_sum);
  }
  doc_loglikelihood -= nonzero_doc_topic * lgamma(FLAGS_alpha);
  doc_loglikelihood += stat_.size() * lgamma(alpha_sum);

  double model_loglikelihood = .0;
  for (const auto count : summary_) {
    model_loglikelihood -= lgamma(count + beta_sum);
    model_loglikelihood += lgamma(beta_sum);
  }
  double nonzero_word_topic = 0;
  std::vector<int> word_topic_count(FLAGS_num_topic);
  for (const auto& word : train_) {
    std::fill(RANGE(word_topic_count), 0);
    for (const auto topic : word.assignment_) ++word_topic_count[topic];
    for (int k = 0; k < FLAGS_num_topic; ++k) {
      int count = word_topic_count[k];
      if (count != 0) {
        ++nonzero_word_topic;
        model_loglikelihood += lgamma(count + FLAGS_beta);
      }
    }
  }
  model_loglikelihood -= nonzero_word_topic * lgamma(FLAGS_beta);

  LI << "nonzero: doc,word = "
     << nonzero_doc_topic / stat_.size() << "/" << FLAGS_num_topic << ", "
     << nonzero_word_topic / train_.size() << "/" << FLAGS_num_topic;
  LI << "loglikelihood: doc,model,total = "
     << doc_loglikelihood << ", "
     << model_loglikelihood << ", "
     << doc_loglikelihood + model_loglikelihood;
}

void Trainer2::PrintPerplexity() {
  double beta_sum = FLAGS_beta * train_.size();
  std::vector<int> doc_topic_count(FLAGS_num_topic);
  std::vector<double> prob(FLAGS_num_topic);
  std::vector<double> theta(FLAGS_num_topic);

  // Cache phi
  std::vector<std::vector<double>> phi;
  for (const auto& word : train_) {
    std::vector<double> phi_w(FLAGS_num_topic, .0);
    for (const auto topic : word.assignment_)
      ++phi_w[topic];
    for (int k = 0; k < FLAGS_num_topic; ++k)
      phi_w[k] = (phi_w[k] + FLAGS_beta) / (summary_[k] + beta_sum);
    phi.emplace_back(phi_w);
  } // end of each word

  // Cache top topic for each phi_w
  std::vector<double> top_topic;
  for (const auto& phi_w : phi)
    top_topic.push_back(std::max_element(RANGE(phi_w)) - phi_w.begin());

  double numer = .0, denom = .0;
  for (auto& doc : test_) {
    // Initialize uniformly
    doc.assignment_.clear();
    for (const auto word_id : doc.token_) {
      int most_probable_topic = (train_[word_id].token_.empty())
                                ? (_unif01(_rng) * FLAGS_num_topic) // OOV
                                : top_topic[word_id];
      doc.assignment_.push_back(most_probable_topic);
    }

    // Construct doc topic count on the fly
    std::fill(RANGE(doc_topic_count), 0);
    for (const auto topic : doc.assignment_) ++doc_topic_count[topic];

    // Perform Gibbs sampling to obtain an estimate of theta
    for (int iter = 1; iter <= EVAL_ITER; ++iter) {
      for (size_t n = 0; n < doc.token_.size(); ++n) {
        // Localize
        int old_topic = doc.assignment_[n];
        const auto& phi_w = phi[doc.token_[n]];
        // Decrement
        --doc_topic_count[old_topic];
        // Compute prob
        std::fill(RANGE(prob), .0);
        for (int k = 0; k < FLAGS_num_topic; ++k) {
          prob[k] = ((k == 0) ? 0 : prob[k-1])
                    + phi_w[k] * (doc_topic_count[k] + FLAGS_alpha);
        }
        // Sample
        int new_topic = -1;
        double sample = _unif01(_rng) * prob[FLAGS_num_topic - 1];
        for (new_topic = 0; prob[new_topic] < sample; ++new_topic);
        CHECK_GE(new_topic, 0);
        CHECK_LT(new_topic, FLAGS_num_topic);
        // Increment
        ++doc_topic_count[new_topic];
        // Set
        doc.assignment_[n] = new_topic;
      } // end of for each n
    } // end of iter

    // Compute theta
    std::fill(RANGE(theta), .0);
    double theta_denom = doc.token_.size() + FLAGS_alpha * FLAGS_num_topic;
    for (int k = 0; k < FLAGS_num_topic; ++k) {
      theta[k] = (doc_topic_count[k] + FLAGS_alpha) / theta_denom;
    }

    // Compute numer for one doc
    for (size_t n = 0; n < doc.token_.size(); ++n) {
      double lhood = .0;
      const auto& phi_w = phi[doc.token_[n]];
      for (int k = 0; k < FLAGS_num_topic; ++k) lhood += phi_w[k] * theta[k];
      numer += log(lhood);
    } // end of each n

    denom += doc.token_.size();
  } // end of for each doc

  LI << "Perplexity: " << exp(- numer / denom);
}

