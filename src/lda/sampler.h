// Copyright (c) 2015 Tencent Inc.
// Author: Yafei Zhang (zhangyafeikimi@gmail.com)
//
// document, corpus and lda sampler
//

#ifndef SRC_LDA_SAMPLER_H_
#define SRC_LDA_SAMPLER_H_

#include <stdio.h>
#include <string>
#include <vector>
#include "lda/alias.h"
#include "lda/array.h"

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
  // topics_count_[k]: # of words assigned to topic k
  IntDenseTable topics_count_;
  // docs_topics_count_[m][k]: # of words in doc m assigned to topic k
  IntTables docs_topics_count_;
  // words_topics_count_[v][k]: # of word v assigned to topic k
  IntTables words_topics_count_;

  // model hyper parameters
  // hp_alpha_[k]: asymmetric doc-topic prior for topic k
  std::vector<double> hp_alpha_;
  double hp_sum_alpha_;
  // beta: symmetric topic-word prior for topic k
  double hp_beta_;
  double hp_sum_beta_;

  // hyper parameters optimizations
  int hp_opt_;
  int hp_opt_interval_;
  double hp_opt_alpha_shape_;
  double hp_opt_alpha_scale_;
  int hp_opt_alpha_iteration_;
  int hp_opt_beta_iteration_;
  // docs_topic_count_hist_[k][n]:
  // # of documents in which topic "k" occurs "n" times.
  std::vector<std::vector<int> > docs_topic_count_hist_;
  // doc_len_hist_[n]:
  // # of documents whose length are "n".
  std::vector<int> doc_len_hist_;
  // word_topic_count_hist_[n]:
  // # of words which are assigned to a topic "n" times.
  std::vector<int> word_topic_count_hist_;
  // topic_len_hist_[n]:
  // # of topics which occurs "n" times.
  std::vector<int> topic_len_hist_;

  // iteration variables
  int total_iteration_;
  int burnin_iteration_;
  int log_likelyhood_interval_;
  int iteration_;
  // a value of enum TableType
  int storage_type_;

  // topic CDF only for PlainGibbsSampler
  std::vector<double> topic_cdf_;

 public:
  PlainGibbsSampler() : K_(0),
    hp_sum_alpha_(0.0),
    hp_beta_(0.0),
    hp_opt_(0),
    hp_opt_interval_(0),
    hp_opt_alpha_shape_(0.0),
    hp_opt_alpha_scale_(0.0),
    hp_opt_alpha_iteration_(0),
    hp_opt_beta_iteration_(0),
    total_iteration_(0),
    burnin_iteration_(-1),
    log_likelyhood_interval_(0),
    storage_type_(kSparseHist) {}
  virtual ~PlainGibbsSampler();

  // setters
  int& K() {
    return K_;
  }

  double& alpha() {
    return hp_sum_alpha_;
  }

  double& beta() {
    return hp_beta_;
  }

  int& hp_opt() {
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

  int& storage_type() {
    return storage_type_;
  }
  // end of setters

  void LoadCorpus(FILE* fp, int with_id);
  void SaveModel(const std::string& prefix) const;
  int Initialize();
  virtual int InitializeSampler();
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

class SparseLDASampler : public PlainGibbsSampler {
 private:
  double smooth_sum_;
  double doc_bucket_sum_;
  double word_bucket_sum_;
  std::vector<double> smooth_bucket_;
  std::vector<double> doc_bucket_;
  std::vector<double> word_bucket_;
  std::vector<double> cache_;

 public:
  SparseLDASampler() {}

  virtual int InitializeSampler();
  virtual void PostSampleCorpus();
  virtual void PostSampleDocument(int m);
  virtual void SampleDocument(int m);
  void RemoveOrAddWordTopic(int m, int v, int k, int remove);
  int SampleDocumentWord(int m, int v);
  void PrepareSmoothBucket();
  void PrepareDocBucket(int m);
  void PrepareWordBucket(int v);
};

class LightLDASampler : public PlainGibbsSampler {
 private:
  Alias hp_alpha_alias_table_;
  Alias word_alias_table_;
  std::vector<double> word_topics_prob_;
  std::vector<std::vector<int> > cached_words_topic_samples_;
  int mh_step_;

 public:
  LightLDASampler() : mh_step_(0) {}

  int& mh_step() {
    return mh_step_;
  }

  virtual int InitializeSampler();
  virtual void PostSampleCorpus();
  virtual void SampleDocument(int m);
  int SampleWithWord(int v);
  double WordAcceptRate(int m,
                        int v,
                        int old_k,
                        int s,
                        int t) const;
  int SampleWithDoc(const Doc& doc, int v);
  double DocAcceptRate(int m,
                       int v,
                       int old_k,
                       int s,
                       int t) const;
};

#endif  // SRC_LDA_SAMPLER_H_
