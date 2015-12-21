// Copyright (c) 2015 Tencent Inc.
// Author: Yafei Zhang (zhangyafeikimi@gmail.com)
//
// Vose's alias table algorithm
//

#ifndef SRC_LDA_ALIAS_H_
#define SRC_LDA_ALIAS_H_

#include <vector>

class Alias {
 private:
  struct AliasItem {
    double prob;
    int index;
  };

  std::vector<AliasItem> table_;
  int n_;
  volatile int usage_;

 public:
  int n() const {
    return n_;
  }

  int usage() const {
    return usage_;
  }

 public:
  Alias() : n_(0), usage_(0) {}
  void Construct(const std::vector<double>& prob);
  void Construct(const std::vector<double>& prob, double prob_sum);

  int Sample() const;
  // u1, u2 are in [0, 1)
  int Sample(double u1, double u2) const {
    int i = (int)(u1 * n_);
    return (u2 < table_[i].prob) ? i : table_[i].index;
  }
};

#endif  // SRC_LDA_ALIAS_H_