// Copyright (c) 2015 Tencent Inc.
// Author: Yafei Zhang (zhangyafeikimi@gmail.com)
//

#include <queue>

#include "common/mt64.h"
#include "common/mt19937ar.h"
#include "lda/alias.h"

class RandInializer {
 public:
  RandInializer() {
    init_genrand64(0312);
    init_genrand(0705);
  }
};
static RandInializer rand_inializer;

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
  usage_ = 0;

  std::vector<double> normalized_prob;
  normalized_prob.reserve(n_);
  for (int i = 0; i < n_; ++i) {
    normalized_prob.push_back((prob[i] * n_) / prob_sum);
  }

  std::queue<int> small, large;
  for (int i = 0; i < n_; ++i) {
    if (normalized_prob[i] < 1.0) {
      small.push(i);
    } else {
      large.push(i);
    }
  }

  while (!(small.empty() || large.empty())) {
    int l = small.front();
    small.pop();
    int g = large.front();
    large.pop();
    table_[l].prob = normalized_prob[l];
    table_[l].index = g;
    normalized_prob[g] = (normalized_prob[g] + normalized_prob[l]) - 1;
    if (normalized_prob[g] < 1.0) {
      small.push(g);
    } else {
      large.push(g);
    }
  }

  while (!large.empty()) {
    int g = large.front();
    large.pop();
    table_[g].prob = 1.0;
  }

  while (!small.empty()) {
    int l = small.front();
    small.pop();
    table_[l].prob = 1.0;
  }
}

int Alias::Sample() const {
  return Sample(genrand64_real2(), genrand64_real2());
}
