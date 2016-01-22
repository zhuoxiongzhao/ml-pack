// Copyright (c) 2015 Tencent Inc.
// Author: Yafei Zhang (zhangyafeikimi@gmail.com)
//

#include "lda/rand.h"
#include "lda/sampler.h"

int AliasLDASampler::InitializeSampler() {
  p_pdf_.resize(K_);
  q_sums_.resize(V_);
  q_samples_.resize(V_);
  q_pdf_.resize(K_);
  if (mh_step_ == 0) {
    mh_step_ = 8;
  }
  return 0;
}

void AliasLDASampler::SampleDocument(int m) {
  const Doc& doc = docs_[m];
  Word* word = &words_[doc.index];
  IntTable& doc_m_topics_count = docs_topics_count_[m];

  for (int n = 0; n < doc.N; n++, word++) {
    const int v = word->v;
    IntTable& word_v_topics_count = words_topics_count_[v];
    const int old_k = word->k;
    int s = old_k;
    int t;

    // construct p: first part of the proposal
    double p_sum = 0.0;
    IntTable::const_iterator first = doc_m_topics_count.begin();
    IntTable::const_iterator last = doc_m_topics_count.end();
    for (; first != last; ++first) {
      const int k = first.id();
      double& pdf = p_pdf_[k];
      pdf = first.count() * (word_v_topics_count[k] + hp_beta_)
            / (topics_count_[k] + hp_sum_beta_);
      p_sum  += pdf;
    }

    // prepare samples from q: second part of the proposal
    double q_sum = 0.0;
    std::vector<int>& word_v_q_samples = q_samples_[v];
    int word_v_q_samples_size = (int)word_v_q_samples.size();
    if (word_v_q_samples_size < mh_step_) {
      if (storage_type_ == kSparseHist) {
        q_pdf_.assign(K_, 0.0);
        IntTable::const_iterator first = word_v_topics_count.begin();
        IntTable::const_iterator last = word_v_topics_count.end();
        for (; first != last; ++first) {
          const int k = first.id();
          double& pdf = q_pdf_[k];
          pdf = (word_v_topics_count[k] + hp_beta_)
                / (topics_count_[k] + hp_sum_beta_);
          q_sum += pdf;
        }
        for (int k = 0; k < K_; k++) {
          double& pdf = q_pdf_[k];
          if (pdf == 0.0) {
            pdf = hp_beta_ / (topics_count_[k] + hp_sum_beta_);
            q_sum += pdf;
          }
        }
      } else {
        // kDenseHist or kArrayBufHist
        for (int k = 0; k < K_; k++) {
          double& pdf = q_pdf_[k];
          pdf = (word_v_topics_count[k] + hp_beta_)
                / (topics_count_[k] + hp_sum_beta_);
          q_sum += pdf;
        }
      }
      q_sums_[v] = q_sum;
      q_alias_table_.Build(q_pdf_, q_sum);

      const int cached_samples = K_ * mh_step_ - word_v_q_samples_size;
      word_v_q_samples.reserve(cached_samples);
      for (int i = 0; i < cached_samples; i++) {
        word_v_q_samples.push_back(q_alias_table_.Sample());
      }
    } else {
      q_sum = q_sums_[v];
    }

    for (int step = 0; step < mh_step_; step++) {
      double sample = Rand::Double01() * (p_sum + q_sum);
      if (sample < p_sum) {
        // sample from p
        IntTable::const_iterator first = doc_m_topics_count.begin();
        IntTable::const_iterator last = doc_m_topics_count.end();
        for (; first != last; ++first) {
          const int k = first.id();
          sample -= p_pdf_[k];
          if (sample <= 0.0) {
            break;
          }
        }
        t = first.id();
      } else {
        // sample from q
        t = word_v_q_samples.back();
        word_v_q_samples.pop_back();
      }

      if (s != t) {
        double accept_rate = 0.0;
        //double temp_old = (n_wk[w][topic] + beta) / (n_k[topic] + Vbeta);
        //double temp_new = (n_wk[w][new_topic] + beta) / (n_k[new_topic] + Vbeta);
        //double acceptance = (nd_m[new_topic] + alpha) / (nd_m[topic] + alpha)
        //  *temp_new / temp_old
        //  *(nd_m[topic] * temp_old + alpha*q[w].w[topic])
        //  / (nd_m[new_topic] * temp_new + alpha*q[w].w[new_topic]);

        if (/*accept_rate >= 1.0 || */Rand::Double01() < accept_rate) {
          word->k = t;
          s = t;
        }
      }
    }

    if (old_k != s) {
      --topics_count_[old_k];
      --doc_m_topics_count[old_k];
      --word_v_topics_count[old_k];
      ++topics_count_[s];
      ++doc_m_topics_count[s];
      ++word_v_topics_count[s];
    }
  }
}
