#include "trainer.h"
#include "logger.h"
#include "timer.h"
#include "flag.h"
#include "util.h"

#include <list>
#include <algorithm>

auto *train_file = flag.String("train_file", "", "Text file in LIBSVM format");
auto *test_file = flag.String("test_file", "", "Text file in LIBSVM format");
auto *dump_prefix = flag.String("dump_prefix", "", "Prefix for training results");
auto *num_iter = flag.Int("num_iter", 10, "Number of training iteration");
auto *num_topic = flag.Int("num_topic", 100, "Model size, usually called K");

const int MAX_TEST_ITER = 20;

void Trainer::Train() {
  initialize();
  lg.Printf("");
  lg.Printf("iter   iter_time       joint         llh    test_llh");

  // Initial statistics
  iter_time_.push_back(0);
  joint_.push_back(evaluate_joint());
  llh_.push_back(evaluate_llh());
  test_llh_.push_back(evaluate_test_llh());
  lg.Printf("%4d%12.4lf%12.4lf%12.4lf%12.4lf",
            0, iter_time_.back(), joint_.back(), llh_.back(), test_llh_.back());

  for (int iter = 1; iter <= *num_iter; ++iter) {
    Timer iter_timer("");
    for (auto& doc : train_.corpus_) {
      train_one_document(doc);
    }
    // Collect statistics
    iter_time_.push_back(iter_timer.Get());
    joint_.push_back(evaluate_joint());
    llh_.push_back(evaluate_llh());
    test_llh_.push_back(evaluate_test_llh());
    lg.Printf("%4d%12.4lf%12.4lf%12.4lf%12.4lf",
              iter, iter_time_.back(), joint_.back(), llh_.back(), test_llh_.back());
  }

  // Output
  if (*dump_prefix != "") {
    save_result();
  }
}

void Trainer::initialize() {
  // Init train
  train_.ReadData(train_file->c_str());
  nkw_.resize(dict.size_);
  nk_.setZero(*num_topic);
  for (auto& doc : train_.corpus_) {
    for (auto& pair : doc.body_) {
      pair.asg_ = Dice(*num_topic);
      nkw_[pair.tok_].AddCount(pair.asg_);
      ++nk_(pair.asg_);
    }
  }

  // Init test
  if (*test_file != "") {
    test_.ReadData(test_file->c_str());
    // Resize the matrix
    nkw_.resize(dict.size_);
    test_nkw_.setZero(*num_topic, dict.size_);
    test_nk_.setZero(*num_topic);
    for (auto& doc : test_.corpus_) {
      doc.test_nkd_.setZero(*num_topic);
      for (auto& pair : doc.body_) {
        pair.asg_ = Dice(*num_topic);
        ++doc.test_nkd_(pair.asg_);
        ++test_nkw_(pair.asg_,pair.tok_);
        ++test_nk_(pair.asg_);
      }
    }
  }

  // Init hyperparam
  alpha_sum_ = (real)(train_.num_token_) / train_.num_doc_ / 10; // avg doc length / 10
  alpha_.setConstant(*num_topic, alpha_sum_ / *num_topic);
  beta_sum_ = (real)(train_.num_token_) / *num_topic / 10; // avg topic count / 10
  beta_ = beta_sum_ / dict.size_;
  lg.Printf("alpha sum = %6.4lf, beta = %6.4lf", alpha_sum_, beta_);
}

void Trainer::train_one_document(Document& doc) {
  // Construct doc topic count on the fly to save memory
  IArray nkd(*num_topic);
  for (auto& pair : doc.body_) {
    ++nkd(pair.asg_);
  }

  // Compute cached values
  EArray denom = EREAL(nk_) + beta_sum_;
  real r_sum = (alpha_ / denom).sum() * beta_;
  real s_sum = 0.0;
  EArray t_coeff = (EREAL(nkd) + alpha_) / denom;
  std::list<int> nkd_index; // sorted, exploit sparsity, O(1) insert/erase
  for (int k = 0; k < *num_topic; ++k) {
    int cnt = nkd(k);
    if (cnt != 0) {
      nkd_index.push_back(k);
      s_sum += cnt / denom(k);
    }
  }
  s_sum *= beta_;

  // Construct dist
  EArray t_cumsum(*num_topic); // only access first nkw_size entries
  for (int n = 0; n < (int)doc.body_.size(); ++n) {
    // Localize
    int word_id   = doc.body_[n].tok_;
    int old_topic = doc.body_[n].asg_;
    auto& word = nkw_[word_id]; // sparse word
    int nkw_size = word.item_.size();

    // Decrement
    real nk_betasum = nk_(old_topic) + beta_sum_;
    int cnt = nkd(old_topic);
    r_sum -= alpha_(old_topic) * beta_ / nk_betasum;
    s_sum -= cnt * beta_ / nk_betasum;
    --nkd(old_topic);
    --cnt;
    if (cnt == 0) { // shrink index
      auto pos = std::lower_bound(RANGE(nkd_index), old_topic);
      nkd_index.erase(pos);
    }
    --nk_(old_topic);
    --nk_betasum;
    r_sum += alpha_(old_topic) * beta_ / nk_betasum;
    s_sum += cnt * beta_ / nk_betasum;
    t_coeff(old_topic) = (cnt + alpha_(old_topic)) / nk_betasum;

    // Taking advantage of sparsity
    real t_sum = 0.0;
    t_cumsum.setZero();
    for (int i = 0; i < nkw_size; ++i) {
      auto pair = word.item_[i];
      int nkw_val = (pair.top_ == old_topic) ? pair.cnt_ - 1 : pair.cnt_;
      t_sum += t_coeff(pair.top_) * nkw_val;
      t_cumsum(i) = t_sum;
    }

    // Draw
    real u = Unif01() * (r_sum + s_sum + t_sum);
    int new_topic = -1;
    if (u < t_sum) { // binary search on t_cumsum
      real *t_head = t_cumsum.data();
      int index = std::lower_bound(t_head, t_head + nkw_size, u) - t_head;
      new_topic = word.item_[index].top_;
    } // end of t bucket
    else {
      u -= t_sum;
      if (u < s_sum) { // make use of sparsity in nkd
        u /= beta_;
        new_topic = nkd_index.back(); // numerical reasons
        for (int k : nkd_index) {
          u -= nkd(k) / (nk_(k) + beta_sum_);
          if (u <= 0.0) {
            new_topic = k;
            break;
          }
        }
      } // end of s bucket
      else { // linear serach
        u -= s_sum;
        u /= beta_;
        new_topic = nkd_index.back(); // numerical reasons
        for (int k = 0; k < *num_topic; ++k) {
          u -= alpha_(k) / (nk_(k) + beta_sum_);
          if (u <= 0.0) {
            new_topic = k;
            break;
          }
        }
      } // end of r bucket
    }
    
    // Increment
    nk_betasum = nk_(new_topic) + beta_sum_;
    cnt = nkd(new_topic);
    r_sum -= alpha_(new_topic) * beta_ / nk_betasum;
    s_sum -= cnt * beta_ / nk_betasum;
    ++nkd(new_topic);
    ++cnt;
    if (cnt == 1) { // augment index
      auto pos = std::lower_bound(RANGE(nkd_index), new_topic);
      nkd_index.insert(pos, new_topic);
    }
    ++nk_(new_topic);
    ++nk_betasum;
    r_sum += alpha_(new_topic) * beta_ / nk_betasum;
    s_sum += cnt * beta_ / nk_betasum;
    t_coeff(new_topic) = (cnt + alpha_(new_topic)) / nk_betasum;

    // Set
    if (new_topic != old_topic) {
      doc.body_[n].asg_ = new_topic;
      word.UpdateCount(old_topic, new_topic);
    }
  } // end of iter over tokens
}

/*
void Trainer1::PrintPerplexity() {
  real beta_sum = beta_ * stat_.size();
  std::vector<int> nkd(FLAGS_num_topic);
  std::vector<real> prob(FLAGS_num_topic);
  std::vector<real> theta(FLAGS_num_topic);

  // Cache phi
  std::vector< std::vector<real>> phi;
  for (const auto& word : stat_) {
    std::vector<real> phi_w(FLAGS_num_topic, .0);
    for (const auto& pair : word.item_)
      phi_w[pair.top_] = pair.cnt_;
    for (int k = 0; k < FLAGS_num_topic; ++k)
      phi_w[k] = (phi_w[k] + beta_) / (summary_[k] + beta_sum);
    phi.emplace_back(phi_w);
  } // end of each word

  real numer = .0, denom = .0;
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
    std::fill(RANGE(nkd), 0);
    for (const auto topic : doc.assignment_) ++nkd[topic];

    // Perform Gibbs sampling to obtain an estimate of theta
    for (int iter = 1; iter <= EVAL_ITER; ++iter) {
      for (size_t n = 0; n < doc.token_.size(); ++n) {
        // Localize
        int old_topic = doc.assignment_[n];
        const auto& phi_w = phi[doc.token_[n]];
        // Decrement
        --nkd[old_topic];
        // Compute prob
        std::fill(RANGE(prob), .0);
        for (int k = 0; k < FLAGS_num_topic; ++k) {
          prob[k] = ((k == 0) ? 0 : prob[k-1])
                    + phi_w[k] * (nkd[k] + FLAGS_alpha);
        }
        // Sample
        int new_topic = -1;
        real sample = _unif01(_rng) * prob[FLAGS_num_topic - 1];
        for (new_topic = 0; prob[new_topic] < sample; ++new_topic);
        CHECK_GE(new_topic, 0);
        CHECK_LT(new_topic, FLAGS_num_topic);
        // Increment
        ++nkd[new_topic];
        // Set
        doc.assignment_[n] = new_topic;
      } // end of for each n
    } // end of iter

    // Compute theta
    std::fill(RANGE(theta), .0);
    real theta_denom = doc.token_.size() + FLAGS_alpha * FLAGS_num_topic;
    for (int k = 0; k < FLAGS_num_topic; ++k) {
      theta[k] = (nkd[k] + FLAGS_alpha) / theta_denom;
    }

    // Compute numer for one doc
    for (size_t n = 0; n < doc.token_.size(); ++n) {
      real lhood = .0;
      const auto& phi_w = phi[doc.token_[n]];
      for (int k = 0; k < FLAGS_num_topic; ++k) lhood += phi_w[k] * theta[k];
      numer += log(lhood);
    } // end of each n

    denom += doc.token_.size();
  } // end of for each doc

  LI << "Perplexity: " << exp(- numer / denom);
}
*/

real Trainer::evaluate_joint() {
  real doc_llh = 0.0;
  IArray nkd(*num_topic);
  for (const auto& doc : train_.corpus_) {
    nkd.setZero();
    for (const auto& pair : doc.body_) {
      ++nkd(pair.asg_);
    }
    for (int k = 0; k < *num_topic; ++k) {
      int cnt = nkd(k);
      if (cnt != 0) {
        doc_llh += lgamma(cnt + alpha_(k)) - lgamma(alpha_(k));
      }
    }
    doc_llh -= lgamma(doc.body_.size() + alpha_sum_);
  }
  doc_llh += train_.num_doc_ * lgamma(alpha_sum_);

  real model_llh = 0.0;
  for (int k = 0; k < *num_topic; ++k) {
    model_llh -= lgamma(nk_(k) + beta_sum_);
    model_llh += lgamma(beta_sum_);
  }
  real nonzero_nkw = 0;
  for (const auto& word : nkw_) {
    nonzero_nkw += word.item_.size();
    for (const auto& pair : word.item_)
      model_llh += lgamma(pair.cnt_ + beta_);
  }
  model_llh -= nonzero_nkw * lgamma(beta_);

  return (doc_llh + model_llh) / (real)(train_.num_token_);
}

real Trainer::evaluate_llh() {
  real llh = 0.0;
  EArray denom = EREAL(nk_) + beta_sum_;
  EArray cached_term(*num_topic);
  for (const auto& doc : train_.corpus_) {
    cached_term.setZero();
    for (auto& pair : doc.body_) {
      cached_term(pair.asg_) += 1;
    }
    cached_term = (cached_term + alpha_) / denom;
    int nd = doc.body_.size();
    for (int n = 0; n < nd; ++n) {
      int word_id = doc.body_[n].tok_;
      real s = (cached_term * (nkw_[word_id].Array(*num_topic) + beta_)).sum();
      llh += log(s);
    }
    llh -= nd * log(nd + alpha_sum_);
  }
  return llh / (real)(train_.num_token_);
}

real Trainer::evaluate_test_llh() {
  if (test_.num_doc_ == 0) {
    return 0.0;
  }
  for (auto& doc : test_.corpus_) {
    test_one_document(doc);
  }
  real test_llh = 0.0;
  EArray denom = EREAL(nk_ + test_nk_) + beta_sum_;
  EArray cached_term(*num_topic);
  for (const auto& doc : test_.corpus_) {
    cached_term = (EREAL(doc.test_nkd_) + alpha_) / denom;
    int nd = doc.body_.size();
    for (int n = 0; n < nd; ++n) {
      int word_id = doc.body_[n].tok_;
      real s = (cached_term
                * (nkw_[word_id].Array(*num_topic)
                   + EREAL(test_nkw_.col(word_id)) + beta_)).sum();
      test_llh += log(s);
    }
    test_llh -= nd * log(nd + alpha_sum_);
  }
  return test_llh / (real)(test_.num_token_);
}

void Trainer::test_one_document(Document& doc) {
  std::vector<real> cumsum(*num_topic);
  for (int iter = 1; iter <= MAX_TEST_ITER; ++iter) {
    for (size_t n = 0; n < doc.body_.size(); ++n) {
      int word_id   = doc.body_[n].tok_;
      int old_topic = doc.body_[n].asg_;
      --doc.test_nkd_(old_topic);
      --test_nkw_(old_topic,word_id);
      --test_nk_(old_topic);
      real sum = 0.0;
      EArray dense_nkw = nkw_[word_id].Array(*num_topic);
      for (int k = 0; k < *num_topic; ++k) {
        sum += (doc.test_nkd_(k) + alpha_(k))
               * (dense_nkw(k) + test_nkw_(k,word_id) + beta_)
               / (nk_(k) + test_nk_(k) + beta_sum_);
        cumsum[k] = sum;
      }
      real r = Unif01() * sum;
      int new_topic = std::lower_bound(RANGE(cumsum), r) - cumsum.begin();
      ++doc.test_nkd_(new_topic);
      ++test_nkw_(new_topic,word_id);
      ++test_nk_(new_topic);
      doc.body_[n].asg_ = new_topic;
    }
  } // end of iter
}

void Trainer::save_result() {
}
