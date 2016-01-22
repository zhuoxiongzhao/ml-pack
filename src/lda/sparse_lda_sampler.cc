// Copyright (c) 2015 Tencent Inc.
// Author: Yafei Zhang (zhangyafeikimi@gmail.com)
//

#include "lda/rand.h"
#include "lda/sampler.h"

int SparseLDASampler::InitializeSampler() {
  smooth_pdf_.resize(K_);
  doc_pdf_.resize(K_);
  word_pdf_.resize(K_);
  cache_.resize(K_);
  PrepareSmoothBucket();
  return 0;
}

void SparseLDASampler::PostSampleCorpus() {
  SamplerBase::PostSampleCorpus();

  if (HPOpt_Enabled()) {
    PrepareSmoothBucket();
  }
}

void SparseLDASampler::PostSampleDocument(int m) {
  const IntTable& doc_m_topics_count = docs_topics_count_[m];
  IntTable::const_iterator first = doc_m_topics_count.begin();
  IntTable::const_iterator last = doc_m_topics_count.end();
  for (; first != last; ++first) {
    const int k = first.id();
    cache_[k] = hp_alpha_[k] / (topics_count_[k] + hp_sum_beta_);
  }

  SamplerBase::HPOpt_PostSampleDocument(m);
}

void SparseLDASampler::SampleDocument(int m) {
  PrepareDocBucket(m);

  const Doc& doc = docs_[m];
  Word* word = &words_[doc.index];

  for (int n = 0; n < doc.N; n++, word++) {
    const int v = word->v;
    const int old_k = word->k;
    RemoveOrAddWordTopic(m, v, old_k, 1);
    PrepareWordBucket(v);
    const int new_k = SampleDocumentWord(m, v);
    RemoveOrAddWordTopic(m, v, new_k, 0);
    word->k = new_k;
  }
}

void SparseLDASampler::RemoveOrAddWordTopic(int m, int v, int k, int remove) {
  IntTable& doc_m_topics_count = docs_topics_count_[m];
  IntTable& word_v_topics_count = words_topics_count_[v];
  double& smooth_bucket_k = smooth_pdf_[k];
  double& doc_bucket_k = doc_pdf_[k];
  const double hp_alpha_k = hp_alpha_[k];
  int doc_topic_count;
  int topic_count;

  smooth_sum_ -= smooth_bucket_k;
  doc_sum_ -= doc_bucket_k;

  if (remove) {
    doc_topic_count = --doc_m_topics_count[k];
    --word_v_topics_count[k];
    topic_count = --topics_count_[k];
  } else {
    doc_topic_count = ++doc_m_topics_count[k];
    ++word_v_topics_count[k];
    topic_count = ++topics_count_[k];
  }

  smooth_bucket_k = hp_alpha_k * hp_beta_ / (topic_count + hp_sum_beta_);
  doc_bucket_k = doc_topic_count * hp_beta_ / (topic_count + hp_sum_beta_);
  smooth_sum_ += smooth_bucket_k;
  doc_sum_ += doc_bucket_k;
  cache_[k] = (doc_topic_count + hp_alpha_k) / (topic_count + hp_sum_beta_);
}

int SparseLDASampler::SampleDocumentWord(int m, int v) {
  const double sum = smooth_sum_ + doc_sum_ + word_sum_;
  double sample = Rand::Double01() * sum;
  int new_k = -1;

  if (sample < word_sum_) {
    const IntTable& word_v_topics_count = words_topics_count_[v];
    IntTable::const_iterator first = word_v_topics_count.begin();
    IntTable::const_iterator last = word_v_topics_count.end();
    for (; first != last; ++first) {
      const int k = first.id();
      sample -= word_pdf_[k];
      if (sample <= 0.0) {
        break;
      }
    }
    new_k = first.id();
  } else {
    sample -= word_sum_;
    if (sample < doc_sum_) {
      const IntTable& doc_m_topics_count = docs_topics_count_[m];
      IntTable::const_iterator first = doc_m_topics_count.begin();
      IntTable::const_iterator last = doc_m_topics_count.end();
      for (; first != last; ++first) {
        const int k = first.id();
        sample -= doc_pdf_[k];
        if (sample <= 0.0) {
          break;
        }
      }
      new_k = first.id();
    } else {
      sample -= doc_sum_;
      int k;
      for (k = 0; k < K_; k++) {
        sample -= smooth_pdf_[k];
        if (sample <= 0.0) {
          break;
        }
      }
      new_k = k;
    }
  }

  return new_k;
}

void SparseLDASampler::PrepareSmoothBucket() {
  smooth_sum_ = 0.0;
  for (int k = 0; k < K_; k++) {
    const double tmp = hp_alpha_[k] / (topics_count_[k] + hp_sum_beta_);
    const double pdf = tmp * hp_beta_;
    smooth_pdf_[k] = pdf;
    smooth_sum_ += pdf;
    cache_[k] = tmp;
  }
}

void SparseLDASampler::PrepareDocBucket(int m) {
  doc_sum_ = 0.0;
  doc_pdf_.assign(K_, 0);
  const IntTable& doc_m_topics_count = docs_topics_count_[m];
  IntTable::const_iterator first = doc_m_topics_count.begin();
  IntTable::const_iterator last = doc_m_topics_count.end();
  for (; first != last; ++first) {
    const int k = first.id();
    const double tmp = topics_count_[k] + hp_sum_beta_;
    const double pdf = first.count() * hp_beta_ / tmp;
    doc_pdf_[k] = pdf;
    doc_sum_ += pdf;
    cache_[k] = (first.count() + hp_alpha_[k]) / tmp;
  }
}

void SparseLDASampler::PrepareWordBucket(int v) {
  word_sum_ = 0.0;
  word_pdf_.assign(K_, 0);
  const IntTable& word_v_topics_count = words_topics_count_[v];
  IntTable::const_iterator first = word_v_topics_count.begin();
  IntTable::const_iterator last = word_v_topics_count.end();
  for (; first != last; ++first) {
    const int k = first.id();
    const double pdf = first.count() * cache_[k];
    word_pdf_[k] = pdf;
    word_sum_ += pdf;
  }
}
