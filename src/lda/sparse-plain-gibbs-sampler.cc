// Copyright (c) 2015 Tencent Inc.
// Author: Yafei Zhang (zhangyafeikimi@gmail.com)
//
// N_mk and N_kv are sparse represented
//

#include "common/mt19937ar.h"
#include "common/mt64.h"
#include "common/x.h"
#include "lda/train.h"

int SparseGibbsSampler::InitializeSampler() {
  sparse_N_mk_.resize(M_);
  sparse_N_kv_.resize(K_);
  topic_cdf_.resize(K_);

  for (int m = 0; m < M_; m++) {
    const Doc& doc = docs_[m];
    Word* word = &words_[doc.index];
    SparseHist& topic_hist = sparse_N_mk_[m];

    for (int n = 0; n < doc.N; n++, word++) {
      int v = word->v;
      int k = (int)(genrand_int32() % K_);
      word->k = k;
      topic_hist.inc(k);
      sparse_N_kv_[k].inc(v);
      N_k_[k]++;
    }
  }
  return 0;
}

void SparseGibbsSampler::CollectTheta(Array2D<double>* theta) const {
  for (int m = 0; m < M_; m++) {
    const Doc& doc = docs_[m];
    Array1D<double> theta_m = (*theta)[m];
    const SparseHist& topic_hist = sparse_N_mk_[m];

    for (int k = 0; k < K_; k++) {
      theta_m[k] = 0.0;
    }

    for (int i = 0, size = topic_hist.size(); i < size; i++) {
      const IdCount& topic_count = topic_hist[i];
      int k = topic_count.id;
      int count = topic_count.count;
      theta_m[k] += (count + hp_alpha_k_[k]) / (doc.N + hp_total_alpha_);
    }
  }
}

void SparseGibbsSampler::CollectPhi(Array2D<double>* phi) const {
  for (int k = 0; k < K_; k++) {
    Array1D<double> phi_k = (*phi)[k];
    const SparseHist& word_hist = sparse_N_kv_[k];

    for (int v = 0; v < V_; v++) {
      phi_k[v] = 0.0;
    }

    for (int i = 0, size = word_hist.size(); i < size; i++) {
      const IdCount& word_count = word_hist[i];
      int v = word_count.id;
      int count = word_count.count;
      phi_k[v] += (count + hp_beta_) / (N_k_[k] + hp_total_beta_);
    }
  }
}

double SparseGibbsSampler::LogLikelyhood() const {
  double sum = 0.0;
  for (int m = 0; m < M_; m++) {
    const Doc& doc = docs_[m];
    const Word* word = &words_[doc.index];
    const SparseHist& topic_hist = sparse_N_mk_[m];

    for (int n = 0; n < doc.N; n++, word++) {
      int v = word->v;
      double word_sum = 0.0;

      for (int i = 0, size = topic_hist.size(); i < size; i++) {
        const IdCount& topic_count = topic_hist[i];
        int k = topic_count.id;
        int count = topic_count.count;
        word_sum += (count + hp_alpha_k_[k])
                    * (sparse_N_kv_[k].count(v) + hp_beta_)
                    / (N_k_[k] + hp_total_beta_);
      }
      word_sum /= (doc.N + hp_total_alpha_);
      sum += log(word_sum);
    }
  }
  return sum;
}

void SparseGibbsSampler::SampleDocument(int m) {
  const Doc& doc = docs_[m];
  Word* word = &words_[doc.index];
  SparseHist& topic_hist = sparse_N_mk_[m];
  double talpha = doc.N - 1 + hp_total_alpha_;

  for (int n = 0; n < doc.N; n++, word++) {
    int v = word->v;
    int k = word->k;

    topic_hist.dec(k);
    sparse_N_kv_[k].dec(v);
    N_k_[k]--;

    topic_hist.reset_cache();
    topic_cdf_[0] = 0.0;
    for (k = 0; k < K_ - 1; k++) {
      topic_cdf_[k] += (sparse_N_kv_[k].count(v) + hp_beta_)
                       / (N_k_[k] + hp_total_beta_)
                       * (topic_hist.cached_count(k) + hp_alpha_k_[k])
                       / talpha;
      topic_cdf_[k + 1] = topic_cdf_[k];
    }
    topic_cdf_[K_ - 1] += (sparse_N_kv_[K_ - 1].count(v) + hp_beta_)
                          / (N_k_[K_ - 1] + hp_total_beta_)
                          * (topic_hist.cached_count(K_ - 1)
                             + hp_alpha_k_[K_ - 1])
                          / talpha;

    double r = genrand64_real3() * topic_cdf_[K_ - 1];
    for (k = 0; k < K_; k++) {
      if (topic_cdf_[k] >= r) {
        break;
      }
    }

    topic_hist.inc(k);
    sparse_N_kv_[k].inc(v);
    N_k_[k]++;
    word->k = k;
  }
}
