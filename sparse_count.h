#pragma once

#include "util.h"

#include <vector>

struct SparseCount {
  struct CountPair {
    int top_, cnt_;
    CountPair(int t, int c) : top_(t), cnt_(c) {}
  };
  std::vector<CountPair> item_;

  void AddCount(int topic) {
    int index = -1;
    for (int i = 0; i < (int)item_.size(); ++i) {
      if (item_[i].top_ == topic) {
        index = i;
      }
    }
    if (index != -1) {
      increment_existing(index);
    } else {
      item_.emplace_back(topic, 1);
    }
  }

  void UpdateCount(int old_topic, int new_topic) {
    if (old_topic == new_topic) {
      return;
    }

    // Find old and new index
    int old_index = -1;
    int new_index = -1;
    for (size_t i = 0; i < item_.size(); ++i) {
      if (item_[i].top_ == old_topic) {
        old_index = i;
      }
      if (item_[i].top_ == new_topic) {
        new_index = i;
      }
      if (old_index != -1 and new_index != -1) { // early stop
        break;
      }
    }

    // Decrement old index while maintaining new index
    CountPair temp(item_[old_index].top_, item_[old_index].cnt_ - 1);
    int last_index = item_.size() - 1;
    if (item_[old_index].cnt_ == 1) {
      if (new_index == last_index) { // maintain
        new_index = old_index;
      }
      std::swap(item_[old_index], item_.back()); // swap with last/self
      item_.pop_back();
    } // end of "need to shrink"
    else if (old_index < last_index and temp.cnt_ < item_[old_index+1].cnt_) {
      auto it = std::lower_bound(item_.begin() + old_index + 1,
                                 item_.end(),
                                 temp,
                                 [](CountPair a, CountPair b){ return a.cnt_ > b.cnt_; })-1;
      if (item_.begin() + new_index == it) { // maintain
        new_index = old_index;
      }
      item_[old_index] = *it;
      *it = temp;
    } // end of "no need to shrink but need to rearrange"
    else {
      --item_[old_index].cnt_;
    } // end of "no need to rearrange"
      
    // Increment new
    if (new_index != -1) {
      increment_existing(new_index);
    } else {
      item_.emplace_back(new_topic, 1);
    }
  }

  void increment_existing(int index) {
    CountPair temp(item_[index].top_, item_[index].cnt_ + 1);
    if (index > 0 and temp.cnt_ > item_[index-1].cnt_) {
      auto it = std::upper_bound(item_.begin(),
                                 item_.begin() + index,
                                 temp,
                                 [](CountPair a, CountPair b){ return a.cnt_ > b.cnt_; });
      item_[index] = *it;
      *it = temp;
    } // end of "need to rearrange"
    else {
      ++item_[index].cnt_;
    } // end of "no need to rearrange"
  }

  EArray Array(int len) {
    EArray arr(len);
    for (const auto& pr : item_) {
      arr(pr.top_) = pr.cnt_;
    }
    return arr;
  }
};
