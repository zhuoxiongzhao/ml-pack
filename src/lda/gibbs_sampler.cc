// Copyright (c) 2015 Tencent Inc.
// Author: Yafei Zhang (zhangyafeikimi@gmail.com)
//

#include "lda/rand.h"
#include "lda/sampler.h"

int GibbsSampler::InitializeSampler() {
  word_topic_cdf_.resize(K_);
  return 0;
}

void GibbsSampler::SampleDocument(int m) {
  const Doc& doc = docs_[m];
  Word* word = &words_[doc.index];
  IntTable& doc_m_topics_count = docs_topics_count_[m];

  for (int n = 0; n < doc.N; n++, word++) {
    const int v = word->v;
    const int old_k = word->k;
    IntTable& word_v_topics_count = words_topics_count_[v];
    int k, new_k;

    --topics_count_[old_k];
    --doc_m_topics_count[old_k];
    --word_v_topics_count[old_k];

    word_topic_cdf_[0] = 0.0;
    for (k = 0; k < K_ - 1; k++) {
      word_topic_cdf_[k] += (word_v_topics_count[k] + hp_beta_)
                            / (topics_count_[k] + hp_sum_beta_)
                            * (doc_m_topics_count[k] + hp_alpha_[k]);
      word_topic_cdf_[k + 1] = word_topic_cdf_[k];
    }
    word_topic_cdf_[k] += (word_v_topics_count[k] + hp_beta_)
                          / (topics_count_[k] + hp_sum_beta_)
                          * (doc_m_topics_count[k] + hp_alpha_[k]);

    const double sample = Rand::Double01() * word_topic_cdf_[k];
    if (K_ < 128) {
      // brute force search
      for (new_k = 0; new_k < K_; new_k++) {
        if (word_topic_cdf_[new_k] >= sample) {
          break;
        }
      }
    } else  {
      // binary search
      int count = K_, half_count;
      int first = 0, middle;
      while (count > 0) {
        half_count = count / 2;
        middle = first + half_count;
        if (sample <= word_topic_cdf_[middle]) {
          count = half_count;
        } else {
          first = middle + 1;
          count -= (half_count + 1);
        }
      }
      new_k = first;
    }

    ++topics_count_[new_k];
    ++doc_m_topics_count[new_k];
    ++word_v_topics_count[new_k];
    word->k = new_k;
  }
}
