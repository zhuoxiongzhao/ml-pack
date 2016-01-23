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
  int s, t;
  // Macro SMOLA_ALIAS_LDA implements the pure algorithm from
  // Alex Smola's paper. Otherwise,
  // some of my changes are made to make it easier and faster.
  // #define SMOLA_ALIAS_LDA
#if defined SMOLA_ALIAS_LDA
  int N_ms, N_vs, N_s, N_mt, N_vt, N_t;
#endif
  int N_ms_prime, N_vs_prime, N_s_prime;
  int N_mt_prime, N_vt_prime, N_t_prime;
  double hp_alpha_s, hp_alpha_t;
  double accept_rate;
  double temp_s, temp_t;
  double p_sum, q_sum;
  double sample;

  for (int n = 0; n < doc.N; n++, word++) {
    const int v = word->v;
    IntTable& word_v_topics_count = words_topics_count_[v];
    const int old_k = word->k;
    s = old_k;

    N_ms_prime = --doc_m_topics_count[s];
    N_vs_prime = --word_v_topics_count[s];
    N_s_prime = --topics_count_[s];
#if defined SMOLA_ALIAS_LDA
    N_ms = N_ms_prime + 1;
    N_vs = N_vs_prime + 1;
    N_s = N_s_prime + 1;
#endif
    temp_s = (N_vs_prime + hp_beta_) / (N_s_prime + hp_sum_beta_);
    hp_alpha_s = hp_alpha_[s];

    // construct p: first part of the proposal
    p_sum = 0.0;
    IntTable::const_iterator first = doc_m_topics_count.begin();
    IntTable::const_iterator last = doc_m_topics_count.end();
    for (; first != last; ++first) {
      const int k = first.id();
      double& pdf = p_pdf_[k];
      pdf = first.count()
            * (word_v_topics_count[k] + hp_beta_)
            / (topics_count_[k] + hp_sum_beta_);
      p_sum  += pdf;
    }

    // prepare samples from q: second part of the proposal
    q_sum = 0.0;
    std::vector<int>& word_v_q_samples = q_samples_[v];
    const int word_v_q_samples_size = (int)word_v_q_samples.size();
    if (word_v_q_samples_size < mh_step_) {
      // construct q
      if (storage_type_ == kSparseHist) {
        q_pdf_.assign(K_, 0.0);
        IntTable::const_iterator first = word_v_topics_count.begin();
        IntTable::const_iterator last = word_v_topics_count.end();
        for (; first != last; ++first) {
          const int k = first.id();
          double& pdf = q_pdf_[k];
          pdf = hp_alpha_[k]
                * (word_v_topics_count[k] + hp_beta_)
                / (topics_count_[k] + hp_sum_beta_);
          q_sum += pdf;
        }
        for (int k = 0; k < K_; k++) {
          double& pdf = q_pdf_[k];
          if (pdf == 0.0) {
            pdf = hp_alpha_[k]
                  * hp_beta_
                  / (topics_count_[k] + hp_sum_beta_);
            q_sum += pdf;
          }
        }
      } else {
        // kDenseHist or kArrayBufHist
        for (int k = 0; k < K_; k++) {
          double& pdf = q_pdf_[k];
          pdf = hp_alpha_[k]
                * (word_v_topics_count[k] + hp_beta_)
                / (topics_count_[k] + hp_sum_beta_);
          q_sum += pdf;
        }
      }
#if defined SMOLA_ALIAS_LDA
      double& pdf_s = q_pdf_[s];
      q_sum -= pdf_s;
      pdf_s = hp_alpha_s
              * (N_vs + hp_beta_)
              / (N_s + hp_sum_beta_);
      q_sum += pdf_s;
#endif
      q_sums_[v] = q_sum;
      q_alias_table_.Build(q_pdf_, q_sum);

      const int cached_samples = K_ * mh_step_;
      word_v_q_samples.reserve(cached_samples);
      for (int i = word_v_q_samples_size; i < cached_samples; i++) {
        word_v_q_samples.push_back(q_alias_table_.Sample());
      }
    } else {
      q_sum = q_sums_[v];
    }

    for (int step = 0; step < mh_step_; step++) {
      sample = Rand::Double01() * (p_sum + q_sum);
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
        N_mt_prime = doc_m_topics_count[t];
        N_vt_prime = word_v_topics_count[t];
        N_t_prime = topics_count_[t];
#if defined SMOLA_ALIAS_LDA
        N_mt = N_mt_prime;
        N_vt = N_vt_prime;
        N_t = N_t_prime;
        if (old_k == t) {
          N_mt++;
          N_vt++;
          N_t++;
        }
#endif
        temp_t = (N_vt_prime + hp_beta_) / (N_t_prime + hp_sum_beta_);
        hp_alpha_t = hp_alpha_[t];
#if defined SMOLA_ALIAS_LDA
        accept_rate =
          (N_mt_prime + hp_alpha_t) / (N_ms_prime + hp_alpha_s)
          * temp_t / temp_s
          * (N_ms_prime * temp_s
             + hp_alpha_s * (N_vs + hp_beta_) / (N_s + hp_sum_beta_))
          / (N_mt_prime * temp_t
             + hp_alpha_t* (N_vt + hp_beta_) / (N_t + hp_sum_beta_));
#else
        accept_rate =
          (N_mt_prime + hp_alpha_t) / (N_ms_prime + hp_alpha_s)
          * temp_t / temp_s
          * (N_ms_prime + hp_alpha_s) * temp_s
          / (N_mt_prime + hp_alpha_t) / temp_t;
#endif
        if (/*accept_rate >= 1.0 || */Rand::Double01() < accept_rate) {
          word->k = t;
          s = t;
#if defined SMOLA_ALIAS_LDA
          N_ms = N_mt;
          N_vs = N_vt;
          N_s = N_t;
#endif
          N_ms_prime = N_mt_prime;
          N_vs_prime = N_vt_prime;
          N_s_prime = N_t_prime;
          temp_s = temp_t;
          hp_alpha_s = hp_alpha_t;
        }
      }
    }

    ++topics_count_[s];
    ++doc_m_topics_count[s];
    ++word_v_topics_count[s];
  }
}
