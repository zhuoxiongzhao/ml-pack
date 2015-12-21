// Copyright (c) 2015 Tencent Inc.
// Author: Yafei Zhang (zhangyafeikimi@gmail.com)
//
// document, corpus and trainer
//

#ifndef SRC_LDA_TRAIN_H_
#define SRC_LDA_TRAIN_H_

#include <stdio.h>
#include <string>
#include <vector>
#include "lda/array2d.h"
#include "lda/hist.h"

struct Doc {
  int index;  // index in "Model::words_"
  int N;  // # of words
};

struct Word {
  int v;  // word id in vocabulary, starts from 0
  int k;  // topic id assign to this word, starts from 0
};

class PlainGibbsSampler {
 protected:
  // corpus
  std::vector<std::string> doc_ids_;
  std::vector<Doc> docs_;
  std::vector<Word> words_;
  int M_;  // # of docs
  int V_;  // # of vocabulary

  // model parameters
  int K_;  // # of topics
  // N_k: # of words assigned to topic k
  std::vector<int> N_k_;
  // N_mk[m][k]: # of words in doc m assigned to topic k
  Array2D<int> N_mk_;
  // N_kv[k][v]: # of word v assigned to topic k
  Array2D<int> N_kv_;
  // topic CDF for sampling
  std::vector<double> topic_cdf_;

  // hyper parameters
  // alpha_k[k]: asymmetric doc-topic prior for topic k
  std::vector<double> hp_alpha_k_;
  // sums of alpha_k
  double hp_total_alpha_;
  // beta: symmetric topic-word prior for topic k
  double hp_beta_;
  double hp_total_beta_;

  // hyper parameters optimizations
  int hp_opt_;
  int hp_opt_interval_;
  double hp_opt_alpha_shape_;
  double hp_opt_alpha_scale_;
  int hp_opt_alpha_iteration_;
  int hp_opt_beta_iteration_;
  // hp_opt_topic_doc_count_[k][n]:
  // # of documents in which topic "k" occurs "n" times.
  std::vector<std::vector<int> > hp_opt_topic_doc_count_;
  // hp_opt_doc_len_count_[n]:
  // # of documents whose length are "n".
  std::vector<int> hp_opt_doc_len_count_;
  // hp_opt_word_topic_count_[n]:
  // # of words which are assigned to a topic "n" times.
  std::vector<int> hp_opt_word_topic_count_;
  // hp_opt_topic_len_count_[n]:
  // # of topics which occurs "n" times.
  std::vector<int> hp_opt_topic_len_count_;

  // iteration variables
  int total_iteration_;
  int burnin_iteration_;
  int log_likelyhood_interval_;
  int iteration_;

 public:
  PlainGibbsSampler() : K_(0),
    hp_total_alpha_(0.0),
    hp_beta_(0.0),
    hp_opt_(0),
    hp_opt_interval_(0),
    hp_opt_alpha_shape_(0.0),
    hp_opt_alpha_scale_(0.0),
    hp_opt_alpha_iteration_(0),
    hp_opt_beta_iteration_(0),
    total_iteration_(0),
    burnin_iteration_(0),
    log_likelyhood_interval_(0) {}
  virtual ~PlainGibbsSampler();

  // setters
  int& K() {
    return K_;
  }

  double& alpha() {
    return hp_total_alpha_;
  }

  double& beta() {
    return hp_beta_;
  }

  int& hp_optimze() {
    return hp_opt_;
  }

  int& hp_opt_interval() {
    return hp_opt_interval_;
  }

  double& hp_opt_alpha_shape() {
    return hp_opt_alpha_shape_;
  }

  double& hp_opt_alpha_scale() {
    return hp_opt_alpha_scale_;
  }

  int& hp_opt_alpha_iteration() {
    return hp_opt_alpha_iteration_;
  }

  int& hp_opt_beta_iteration() {
    return hp_opt_beta_iteration_;
  }

  int& total_iteration() {
    return total_iteration_;
  }

  int& burnin_iteration() {
    return burnin_iteration_;
  }

  int& log_likelyhood_interval() {
    return log_likelyhood_interval_;
  }
  // end of setters

  void LoadCorpus(FILE* fp, int with_id);
  void SaveModel(const std::string& prefix) const;
  int InitializeParam();
  virtual int InitializeSampler();
  virtual void UninitializeSampler();
  virtual void CollectTheta(Array2D<double>* theta) const;
  virtual void CollectPhi(Array2D<double>* phi) const;
  virtual double LogLikelyhood() const;
  virtual int Train();
  virtual void PreSampleCorpus();
  virtual void PostSampleCorpus();
  virtual void SampleCorpus();
  virtual void PreSampleDocument(int m);
  virtual void PostSampleDocument(int m);
  virtual void SampleDocument(int m);
  virtual void HPOpt_Initialize();
  virtual void HPOpt_Optimize();
  virtual void HPOpt_OptimizeAlpha();
  virtual void HPOpt_PrepareOptimizeBeta();
  virtual void HPOpt_OptimizeBeta();
  virtual void HPOpt_PostSampleDocument(int m);

  int HPOpt_Enabled() const {
    if (hp_opt_ && iteration_ > burnin_iteration_
        && (iteration_ % hp_opt_interval_) == 0) {
      return 1;
    }
    return 0;
  }
};

class SparseGibbsSampler : public PlainGibbsSampler {
 protected:
  std::vector<SparseHist> sparse_N_mk_;
  std::vector<SparseHist> sparse_N_kv_;

 public:
  virtual int InitializeSampler();
  virtual void CollectTheta(Array2D<double>* theta) const;
  virtual void CollectPhi(Array2D<double>* phi) const;
  virtual double LogLikelyhood() const;
  virtual void SampleDocument(int m);
};

#endif  // SRC_LDA_TRAIN_H_
