// Copyright (c) 2015 Tencent Inc.
// Author: Yafei Zhang (zhangyafeikimi@gmail.com)
//
// document, vocabulary and corpus
//

#ifndef SRC_LDA_MODEL_H_
#define SRC_LDA_MODEL_H_

#include <stdio.h>
#include <string>
#include <vector>

struct Doc {
  int index;  // index in "Model::words_"
  int N;  // # of words
};

struct Word {
  int v;  // word id in vocabulary, starts from 0
  int k;  // topic id assign to this word, starts from 0
};

template <class T>
class Array2D {
 private:
  int d1_;
  int d2_;
  std::vector<T> a_;

 public:
  void Init(int d1, int d2) {
    d1_ = d1;
    d2_ = d2;
    a_.resize(d1 * d2);
  }

  T* operator[](int i) {
    return &a_[0] + d2_ * i;
  }

  const T* operator[](int i) const {
    return &a_[0] + d2_ * i;
  }
};

class LDAModel {
 protected:
  // corpus
  std::vector<std::string> doc_ids_;
  std::vector<Doc> docs_;
  std::vector<Word> words_;
  int M_;  // # of docs
  int V_;  // # of vocabulary

  // model parameters
  int K_;  // # of topics
  // N_mk[m][k]: # of words in doc m assigned to topic k
  Array2D<int> N_mk_;
  // N_k: # of words assigned to topic k
  std::vector<int> N_k_;
  // N_kv[k][v]: # of word v assigned to topic k
  Array2D<int> N_kv_;

  // theta_mk[m][k]: doc m's topic k' proportion
  Array2D<double> theta_mk_;
  // phi_k[k][v]: the probability that word v is assigned to topic k
  Array2D<double> phi_kv_;

  // hyper parameters alpha_k[k]: asymmetric doc-topic prior for topic k
  std::vector<double> alpha_k_;
  // sums of alpha_k
  double total_alpha_;
  // hyper parameters beta: symmetric topic-word prior for topic k
  double beta_;

  int total_iteration_;
  int iteration_;

 public:
  LDAModel() : K_(0), total_alpha_(0.0), beta_(0.0),
    total_iteration_(0) {}

  int& K() {
    return K_;
  }

  int K() const {
    return K_;
  }

  double& alpha() {
    return total_alpha_;
  }

  double alpha() const {
    return total_alpha_;
  }

  double& beta() {
    return beta_;
  }

  double beta() const {
    return beta_;
  }

  int& total_iteration() {
    return total_iteration_;
  }

  int total_iteration() const {
    return total_iteration_;
  }

  void LoadCorpus(FILE* fp, int with_id);
  void LoadModel(const std::string& prefix);
  void SaveModel(const std::string& prefix) const;
  virtual void InitializeModel();
  virtual void OptimizeHyper();
  virtual void CollectThetaPhi();
  virtual double LogLikelyhood() const;
  virtual void SampleCorpus() = 0;
  virtual void Train() = 0;
};

class PlainGibbsSampler : public LDAModel {
 protected:
  // topics CDF for SampleCorpus
  std::vector<double> topic_cdf_;

 public:
  virtual void InitializeModel();
  virtual void SampleCorpus();
  virtual void Train();
};

#endif  // SRC_LDA_MODEL_H_
