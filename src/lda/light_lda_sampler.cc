// Copyright (c) 2015 Tencent Inc.
// Author: Yafei Zhang (zhangyafeikimi@gmail.com)
//

#include "lda/rand.h"
#include "lda/sampler.h"

int LightLDASampler::InitializeSampler() {
  hp_alpha_alias_table_.Build(hp_alpha_, hp_sum_alpha_);
  word_topics_pdf_.resize(K_);
  words_topic_samples_.resize(V_);
  if (mh_step_ == 0) {
    mh_step_ = 8;
  }
  return 0;
}

void LightLDASampler::PostSampleCorpus() {
  SamplerBase::PostSampleCorpus();

  if (HPOpt_Enabled()) {
    if (hp_opt_alpha_iteration_ > 0) {
      hp_alpha_alias_table_.Build(hp_alpha_, hp_sum_alpha_);
    }
  }
}

void LightLDASampler::SampleDocument(int m) {
  const Doc& doc = docs_[m];
  Word* word = &words_[doc.index];
  IntTable& doc_m_topics_count = docs_topics_count_[m];

  int s, t;
  int N_ms, N_ms_prime, N_mt, N_mt_prime;
  int N_vs, N_vs_prime, N_vt, N_vt_prime;
  int N_s, N_s_prime, N_t, N_t_prime;
  double hp_alpha_s, hp_alpha_t;
  double accept_rate;

  for (int n = 0; n < doc.N; n++, word++) {
    const int v = word->v;
    IntTable& word_v_topics_count = words_topics_count_[v];
    const int old_k = word->k;
    s = old_k;

    N_ms = doc_m_topics_count[s];
    N_vs = word_v_topics_count[s];
    N_s = topics_count_[s];
    N_ms_prime = N_ms;
    N_vs_prime = N_vs;
    N_s_prime = N_s;
    if (old_k == s) {
      N_ms_prime--;
      N_vs_prime--;
      N_s_prime--;
    }
    hp_alpha_s = hp_alpha_[s];

    for (int step = 0; step < mh_step_; step++) {
      if (enable_word_proposal_) {
        // sample new topic from word proposal
        t = SampleWithWord(v);

        if (s != t) {
          // calculate accept rate from topic s to topic t:
          // (N^{'}_{mt} + \alpha_t)(N^{'}_{vt} + \beta)
          // -----------------------------------------------
          // (N^{'}_{ms} + \alpha_s)(N^{'}_{vs} + \beta)
          // *
          // (N^{'}_s + \sum\beta)
          // -----------------------
          // (N^{'}_t + \sum\beta)
          // *
          // (N_{vs} + \beta)(N_t + \sum\beta)
          // ---------------------------------
          // (N_{vt} + \beta)(N_s + \sum\beta)
          N_mt = doc_m_topics_count[t];
          N_vt = word_v_topics_count[t];
          N_t = topics_count_[t];
          N_mt_prime = N_mt;
          N_vt_prime = N_vt;
          N_t_prime = N_t;
          if (old_k == t) {
            N_mt_prime--;
            N_vt_prime--;
            N_t_prime--;
          }
          hp_alpha_t = hp_alpha_[t];

          accept_rate =
            (N_mt_prime + hp_alpha_t) / (N_ms_prime + hp_alpha_s)
            * (N_vt_prime + hp_beta_) / (N_vs_prime + hp_beta_)
            * (N_s_prime + hp_sum_beta_) / (N_t_prime + hp_sum_beta_)
            * (N_vs + hp_beta_) / (N_vt + hp_beta_)
            * (N_t + hp_sum_beta_) / (N_s + hp_sum_beta_);

          if (/*accept_rate >= 1.0 || */Rand::Double01() < accept_rate) {
            word->k = t;
            s = t;
            N_ms = N_mt;
            N_vs = N_vt;
            N_s = N_t;
            N_ms_prime = N_mt_prime;
            N_vs_prime = N_vt_prime;
            N_s_prime = N_t_prime;
            hp_alpha_s = hp_alpha_t;
          }
        }
      }

      if (enable_doc_proposal_) {
        t = SampleWithDoc(doc, v);
        if (s != t) {
          // calculate accept rate from topic s to topic t:
          // (N^{'}_{mt} + \alpha_t)(N^{'}_{vt} + \beta)
          // -----------------------------------------------
          // (N^{'}_{ms} + \alpha_s)(N^{'}_{vs} + \beta)
          // *
          // (N^{'}_s + \sum\beta)
          // -----------------------
          // (N^{'}_t + \sum\beta)
          // *
          // (N_{ms} + \alpha_s)
          // -------------------
          // (N_{mt} + \alpha_t)
          N_mt = doc_m_topics_count[t];
          N_vt = word_v_topics_count[t];
          N_t = topics_count_[t];
          N_mt_prime = N_mt;
          N_vt_prime = N_vt;
          N_t_prime = N_t;
          if (old_k == t) {
            N_mt_prime--;
            N_vt_prime--;
            N_t_prime--;
          }
          hp_alpha_t = hp_alpha_[t];

          accept_rate =
            (N_mt_prime + hp_alpha_t) / (N_ms_prime + hp_alpha_s)
            * (N_vt_prime + hp_beta_) / (N_vs_prime + hp_beta_)
            * (N_s_prime + hp_sum_beta_) / (N_t_prime + hp_sum_beta_)
            * (N_ms + hp_alpha_s) / (N_mt + hp_alpha_t);

          if (/*accept_rate >= 1.0 || */Rand::Double01() < accept_rate) {
            word->k = t;
            s = t;
            N_ms = N_mt;
            N_vs = N_vt;
            N_s = N_t;
            N_ms_prime = N_mt_prime;
            N_vs_prime = N_vt_prime;
            N_s_prime = N_t_prime;
            hp_alpha_s = hp_alpha_t;
          }
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

int LightLDASampler::SampleWithWord(int v) {
  // word-proposal: (N_vk + beta)/(N_k + sum_beta)
  std::vector<int>& word_v_topic_samples = words_topic_samples_[v];
  if (word_v_topic_samples.empty()) {
    double sum = 0.0;
    const IntTable& word_v_topics_count = words_topics_count_[v];
    if (storage_type_ == kSparseHist) {
      word_topics_pdf_.assign(K_, 0.0);
      IntTable::const_iterator first = word_v_topics_count.begin();
      IntTable::const_iterator last = word_v_topics_count.end();
      for (; first != last; ++first) {
        const int k = first.id();
        double& pdf = word_topics_pdf_[k];
        pdf = (first.count() + hp_beta_) / (topics_count_[k] + hp_sum_beta_);
        sum += pdf;
      }

      for (int k = 0; k < K_; k++) {
        double& pdf = word_topics_pdf_[k];
        if (pdf == 0.0) {
          pdf = hp_beta_ / (topics_count_[k] + hp_sum_beta_);
          sum += pdf;
        }
      }
    } else {
      for (int k = 0; k < K_; k++) {
        double& pdf = word_topics_pdf_[k];
        pdf = (word_v_topics_count[k] + hp_beta_)
              / (topics_count_[k] + hp_sum_beta_);
        sum += pdf;
      }
    }

    word_alias_table_.Build(word_topics_pdf_, sum);
    int cached_samples = K_ * mh_step_;
    word_v_topic_samples.reserve(cached_samples);
    for (int i = 0; i < cached_samples; i++) {
      word_v_topic_samples.push_back(word_alias_table_.Sample());
    }
  }

  const int new_k = word_v_topic_samples.back();
  word_v_topic_samples.pop_back();
  return new_k;
}

int LightLDASampler::SampleWithDoc(const Doc& doc, int v) {
  // doc-proposal: N_mk + alpha_k
  const double sum = hp_sum_alpha_ + doc.N;
  double sample = Rand::Double01() * sum;
  if (sample < hp_sum_alpha_) {
    return hp_alpha_alias_table_.Sample(sample / hp_sum_alpha_);
  } else {
    int offset = (int)(sample - hp_sum_alpha_);
    int index;
    if (offset != doc.N) {
      index = doc.index + offset;
    } else {
      // rare numerical errors may lie in this branch
      index = doc.index + offset - 1;
    }
    return words_[index].k;
  }
}
