// Copyright (c) 2015 Tencent Inc.
// Author: Yafei Zhang (zhangyafeikimi@gmail.com)
//

#include "lda/alias.h"
#include "lda/rand.h"

void Alias::Construct(const std::vector<double>& prob) {
  double prob_sum = 0.0;
  for (int i = 0, size = (int)prob.size(); i < size; i++) {
    prob_sum += prob[i];
  }
  Construct(prob, prob_sum);
}

void Alias::Construct(const std::vector<double>& prob, double prob_sum) {
  table_.resize(prob.size());
  n_ = (int)prob.size();

  // use cached buffers
  normalized_prob_.resize(n_);
  small_.resize(n_);
  large_.resize(n_);

  for (int i = 0; i < n_; ++i) {
    normalized_prob_[i] = (prob[i] * n_) / prob_sum;
  }

  int small_begin = 0, small_end = 0;
  int large_begin = 0, large_end = 0;

  for (int i = 0; i < n_; ++i) {
    if (normalized_prob_[i] < 1.0) {
      small_[small_end++] = i;
    } else {
      large_[large_end++] = i;
    }
  }

  while (small_begin != small_end && large_begin != large_end) {
    const int l = small_[small_begin++];
    const int g = large_[large_begin++];
    table_[l].prob = normalized_prob_[l];
    table_[l].index = g;
    normalized_prob_[g] = (normalized_prob_[g] + normalized_prob_[l]) - 1;
    if (normalized_prob_[g] < 1.0) {
      small_[small_end++] = g;
    } else {
      large_[large_end++] = g;
    }
  }

  while (large_begin != large_end) {
    const int g = large_[large_begin++];
    table_[g].prob = 1.0;
  }

  while (small_begin != small_end) {
    const int l = small_[small_begin++];
    table_[l].prob = 1.0;
  }
}

int Alias::Sample() const {
  return Sample(Rand::Double01(), Rand::Double01());
}
