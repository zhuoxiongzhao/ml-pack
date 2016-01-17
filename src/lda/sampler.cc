// Copyright (c) 2015 Tencent Inc.
// Author: Yafei Zhang (zhangyafeikimi@gmail.com)
//

#include <assert.h>
#include "common/line-reader.h"
#include "common/x.h"
#include "lda/rand.h"
#include "lda/sampler.h"

/************************************************************************/
/* PlainGibbsSampler */
/************************************************************************/
PlainGibbsSampler::~PlainGibbsSampler() {}

void PlainGibbsSampler::LoadCorpus(FILE* fp, int with_id) {
  LineReader line_reader;
  int line_no = 0;
  char* endptr;
  char* doc_id = NULL;
  char* word_id;
  char* word_count;
  char* word_begin;
  Doc doc;
  Word word;
  int id, i, count;

  Log("Loading corpus.\n");
  V_ = 0;
  while (line_reader.ReadLine(fp) != NULL) {
    line_no++;

    doc.index = (int)words_.size();
    doc.N = 0;

    if (with_id) {
      doc_id = strtok(line_reader.buf, DELIMITER);
      if (doc_id == NULL) {
        Error("line %d, empty line.\n", line_no);
        continue;
      }
      word_begin = NULL;
    } else {
      word_begin = line_reader.buf;
    }

    for (;;) {
      word_id = strtok(word_begin, DELIMITER);
      word_begin = NULL;
      if (word_id == NULL) {
        break;
      }

      word_count = strrchr(word_id, ':');
      if (word_count) {
        if (word_count == word_id) {
          Error("line %d, word id is empty.\n", line_no);
          continue;
        }
        *word_count = '\0';
        word_count++;
        count = (int)strtoll(word_count, &endptr, 10);
        if (*endptr != '\0') {
          Error("line %d, word count error \"%s\".\n", line_no, word_count);
          continue;
        }
      } else {
        count = 1;
      }

      id = (int)strtoll(word_id, &endptr, 10);
      if (*endptr != '\0') {
        Error("line %d, word id error \"%s\".\n", line_no, word_id);
        continue;
      }
      if (id == 0) {
        Error("line %d, word id must start from 1.\n", line_no);
        continue;
      }

      if (id > V_) {
        V_ = id;
      }
      word.v = id - 1;
      for (i = 0; i < count; i++) {
        words_.push_back(word);
        doc.N++;
      }
    }

    if (doc.N) {
      if (with_id) {
        doc_ids_.push_back(doc_id);
      }
      docs_.push_back(doc);
    }
  }

  M_ = (int)docs_.size();
  Log("Loaded %d documents with a %d-size vocabulary.\n", M_, V_);
}

void PlainGibbsSampler::SaveModel(const std::string& prefix) const {
  std::string filename;
  Log("Saving model.\n");
  {
    filename = prefix + "-stat";
    ScopedFile fp(filename.c_str(), ScopedFile::Write);
    fprintf(fp, "M=%d\n", M_);
    fprintf(fp, "V=%d\n", V_);
    fprintf(fp, "K=%d\n", K_);
  }
  {
    // theta_mk[m][k]: doc m's topic k' proportion
    Array2D<double> theta_mk;
    theta_mk.Init(M_, K_);
    CollectTheta(&theta_mk);

    filename = prefix + "-doc-topic";
    ScopedFile fp(filename.c_str(), ScopedFile::Write);
    for (int m = 0; m < M_; m++) {
      if (!doc_ids_.empty()) {
        fprintf(fp, "%s ", doc_ids_[m].c_str());
      }
      for (int k = 0; k < K_ - 1; k++) {
        fprintf(fp, "%lg ", theta_mk[m][k]);
      }
      fprintf(fp, "%lg\n", theta_mk[m][K_ - 1]);
    }
  }
  {
    // phi_k[k][v]: the probability that word v is assigned to topic k
    Array2D<double> phi_kv;
    phi_kv.Init(K_, V_);
    CollectPhi(&phi_kv);

    filename = prefix + "-topic-word";
    ScopedFile fp(filename.c_str(), ScopedFile::Write);
    for (int k = 0; k < K_; k++) {
      for (int v = 0; v < V_ - 1; v++) {
        fprintf(fp, "%lg ", phi_kv[k][v]);
      }
      fprintf(fp, "%lg\n", phi_kv[k][V_ - 1]);
    }
  }
  {
    filename = prefix + "-alpha";
    ScopedFile fp(filename.c_str(), ScopedFile::Write);
    for (int k = 0; k < K_; k++) {
      fprintf(fp, "%lg\n", hp_alpha_[k]);
    }
  }
  {
    filename = prefix + "-beta";
    ScopedFile fp(filename.c_str(), ScopedFile::Write);
    fprintf(fp, "%lg\n", hp_beta_);
  }
  Log("Done.\n");
}

int PlainGibbsSampler::Initialize() {
  if (hp_sum_alpha_ <= 0.0) {
    double avg_doc_len = (double)words_.size() / docs_.size();
    hp_alpha_.resize(K_, avg_doc_len / K_);
    hp_sum_alpha_ = avg_doc_len;
  } else {
    hp_alpha_.resize(K_, hp_sum_alpha_);
    hp_sum_alpha_ = hp_sum_alpha_ * K_;
  }

  if (hp_beta_ <= 0.0) {
    hp_beta_ = 0.1;
  }
  hp_sum_beta_ = V_ * hp_beta_;

  if (hp_opt_) {
    if (hp_opt_interval_ == 0) {
      hp_opt_interval_ = 5;
    }
    if (hp_opt_alpha_scale_ == 0.0) {
      hp_opt_alpha_scale_ = 100000.0;
    }
    if (hp_opt_alpha_iteration_ == 0) {
      hp_opt_alpha_iteration_ = 2;
    }
    if (hp_opt_beta_iteration_ == 0) {
      hp_opt_beta_iteration_ = 200;
    }
  }

  if (total_iteration_ == 0) {
    total_iteration_ = 200;
  }
  iteration_ = 1;

  topics_count_.Init(K_);
  docs_topics_count_.Init(M_, K_, storage_type_);
  words_topics_count_.Init(V_, K_, storage_type_);

  // random initialize topics
  for (int m = 0; m < M_; m++) {
    const Doc& doc = docs_[m];
    Word* word = &words_[doc.index];
    IntTable& doc_m_topics_count = docs_topics_count_[m];
    for (int n = 0; n < doc.N; n++, word++) {
      const int v = word->v;
      const int new_topic = (int)Rand::UInt(K_);
      word->k = new_topic;
      ++topics_count_[new_topic];
      ++doc_m_topics_count[new_topic];
      ++words_topics_count_[v][new_topic];
    }
  }

  const double llh = LogLikelihood();
  Log("LogLikelihood(total/word)=%lg/%lg\n", llh, llh / words_.size());
  return 0;
}

int PlainGibbsSampler::InitializeSampler() {
  topic_cdf_.resize(K_);
  return 0;
}

void PlainGibbsSampler::CollectTheta(Array2D<double>* theta) const {
  for (int m = 0; m < M_; m++) {
    const Doc& doc = docs_[m];
    const IntTable& doc_m_topics_count = docs_topics_count_[m];
    double* theta_m = (*theta)[m];
    for (int k = 0; k < K_; k++) {
      theta_m[k] = (doc_m_topics_count[k] + hp_alpha_[k])
                   / (doc.N + hp_sum_alpha_);
    }
  }
}

void PlainGibbsSampler::CollectPhi(Array2D<double>* phi) const {
  for (int k = 0; k < K_; k++) {
    const int topics_count_k = topics_count_[k];
    double* phi_k = (*phi)[k];
    for (int v = 0; v < V_; v++) {
      phi_k[v] = (words_topics_count_[v][k] + hp_beta_)
                 / (topics_count_k + hp_sum_beta_);
    }
  }
}

double PlainGibbsSampler::LogLikelihood() const {
  double sum = 0.0;
  for (int m = 0; m < M_; m++) {
    const Doc& doc = docs_[m];
    const Word* word = &words_[doc.index];
    const IntTable& doc_m_topics_count = docs_topics_count_[m];
    for (int n = 0; n < doc.N; n++, word++) {
      const int v = word->v;
      const IntTable& word_v_topics_count = words_topics_count_[v];
      double word_sum = 0.0;
      for (int k = 0; k < K_; k++) {
        word_sum += (doc_m_topics_count[k] + hp_alpha_[k])
                    * (word_v_topics_count[k] + hp_beta_)
                    / (topics_count_[k] + hp_sum_beta_);
      }
      word_sum /= (doc.N + hp_sum_alpha_);
      sum += log(word_sum);
    }
  }
  return sum;
}

int PlainGibbsSampler::Train() {
  if (Initialize() != 0) {
    return -1;
  }

  if (InitializeSampler() != 0) {
    return -2;
  }

  for (iteration_ = 1; iteration_ <= total_iteration_; iteration_++) {
    PreSampleCorpus();
    SampleCorpus();
    PostSampleCorpus();
  }
  return 0;
}

void PlainGibbsSampler::PreSampleCorpus() {
  Log("Iteration %d started.\n", iteration_);
  HPOpt_Initialize();
}

void PlainGibbsSampler::PostSampleCorpus() {
  HPOpt_Optimize();
  if (iteration_ > burnin_iteration_
      && iteration_ % log_likelihood_interval_ == 0) {
    const double llh = LogLikelihood();
    Log("LogLikelihood(total/word)=%lg/%lg\n", llh, llh / words_.size());
  }
}

void PlainGibbsSampler::SampleCorpus() {
  for (int m = 0; m < M_; m++) {
    PreSampleDocument(m);
    SampleDocument(m);
    PostSampleDocument(m);
  }
}

void PlainGibbsSampler::PreSampleDocument(int m) {}

void PlainGibbsSampler::PostSampleDocument(int m) {
  HPOpt_PostSampleDocument(m);
}

void PlainGibbsSampler::SampleDocument(int m) {
  const Doc& doc = docs_[m];
  Word* word = &words_[doc.index];
  IntTable& doc_m_topics_count = docs_topics_count_[m];

  for (int n = 0; n < doc.N; n++, word++) {
    const int v = word->v;
    const int old_k = word->k;
    IntTable& word_v_topics_count = words_topics_count_[v];

    --topics_count_[old_k];
    --doc_m_topics_count[old_k];
    --word_v_topics_count[old_k];

    topic_cdf_[0] = 0.0;
    for (int k = 0; k < K_ - 1; k++) {
      topic_cdf_[k] += (word_v_topics_count[k] + hp_beta_)
                       / (topics_count_[k] + hp_sum_beta_)
                       * (doc_m_topics_count[k] + hp_alpha_[k]);
      topic_cdf_[k + 1] = topic_cdf_[k];
    }
    topic_cdf_[K_ - 1] += (word_v_topics_count[K_ - 1] + hp_beta_)
                          / (topics_count_[K_ - 1] + hp_sum_beta_)
                          * (doc_m_topics_count[K_ - 1] + hp_alpha_[K_ - 1]);

    double r = Rand::Double01() * topic_cdf_[K_ - 1];
    int new_k = -1;
    for (new_k = 0; new_k < K_; new_k++) {
      if (topic_cdf_[new_k] >= r) {
        break;
      }
    }
    assert(new_k != -1);

    ++topics_count_[new_k];
    ++doc_m_topics_count[new_k];
    ++word_v_topics_count[new_k];
    word->k = new_k;
  }
}

void PlainGibbsSampler::HPOpt_Initialize() {
  if (!HPOpt_Enabled()) {
    return;
  }

  Log("Hyper optimization will be carried out in this iteration.\n");
  docs_topic_count_hist_.clear();
  docs_topic_count_hist_.resize(K_);
  doc_len_hist_.clear();
  word_topic_count_hist_.clear();
  topic_len_hist_.clear();
}

void PlainGibbsSampler::HPOpt_Optimize() {
  if (!HPOpt_Enabled()) {
    return;
  }

  if (hp_opt_alpha_iteration_ > 0) {
    Log("Hyper optimizing alpha.\n");
    HPOpt_OptimizeAlpha();
  }
  if (hp_opt_beta_iteration_ > 0) {
    Log("Hyper optimizing beta.\n");
    HPOpt_PrepareOptimizeBeta();
    HPOpt_OptimizeBeta();
  }
}

void PlainGibbsSampler::HPOpt_OptimizeAlpha() {
  for (int i = 0; i < hp_opt_alpha_iteration_; i++) {
    double denom = 0.0;
    double diff_digamma = 0.0;
    for (int j = 1, size = (int)doc_len_hist_.size(); j < size; j++) {
      diff_digamma += 1.0 / (j - 1 + hp_sum_alpha_);
      denom += doc_len_hist_[j] * diff_digamma;
    }
    denom -= 1.0 / hp_opt_alpha_scale_;

    hp_sum_alpha_ = 0.0;
    for (int k = 0, size = (int)docs_topic_count_hist_.size();
         k < size; k++) {
      double num = 0.0;
      double alpha_k = hp_alpha_[k];
      const std::vector<int>& docs_topic_k_count_hist =
        docs_topic_count_hist_[k];
      diff_digamma = 0.0;
      for (int j = 1, size = (int)docs_topic_count_hist_[k].size();
           j < size; j++) {
        diff_digamma += 1.0 / (j - 1 + alpha_k);
        num += docs_topic_k_count_hist[j] * diff_digamma;
      }
      alpha_k = (alpha_k * num + hp_opt_alpha_shape_) / denom;
      hp_alpha_[k] = alpha_k;
      hp_sum_alpha_ += alpha_k;
    }
  }
}

void PlainGibbsSampler::HPOpt_PrepareOptimizeBeta() {
  for (int m = 0; m < M_; m++) {
    const IntTable& doc_m_topics_count = docs_topics_count_[m];
    for (int k = 0; k < K_; k++) {
      const int count = doc_m_topics_count[k];
      if (count == 0) {
        continue;
      }
      if ((int)word_topic_count_hist_.size() <= count) {
        word_topic_count_hist_.resize(count + 1);
      }
      ++word_topic_count_hist_[count];
    }
  }

  for (int k = 0; k < K_; k++) {
    const int count = topics_count_[k];
    if (count == 0) {
      continue;
    }
    if ((int)topic_len_hist_.size() <= count) {
      topic_len_hist_.resize(count + 1);
    }
    ++topic_len_hist_[count];
  }
}

void PlainGibbsSampler::HPOpt_OptimizeBeta() {
  for (int i = 0; i < hp_opt_beta_iteration_; i++) {
    double num = 0.0;
    double diff_digamma = 0.0;
    for (int j = 1, size = (int)word_topic_count_hist_.size();
         j < size; j++) {
      diff_digamma += 1.0 / (j - 1 + hp_beta_);
      num += diff_digamma * word_topic_count_hist_[j];
    }

    double denom = 0.0;
    diff_digamma = 0.0;
    for (int j = 1, size = (int)topic_len_hist_.size(); j < size; j++) {
      diff_digamma += 1.0 / (j - 1 + hp_sum_beta_);
      denom += diff_digamma * topic_len_hist_[j];
    }
    hp_sum_beta_ = hp_beta_ * num / denom;
    hp_beta_ = hp_sum_beta_ / V_;
  }
}

void PlainGibbsSampler::HPOpt_PostSampleDocument(int m) {
  if (!HPOpt_Enabled()) {
    return;
  }

  if (hp_opt_alpha_iteration_ > 0) {
    const Doc& doc = docs_[m];
    const IntTable& doc_m_topics_count = docs_topics_count_[m];
    for (int k = 0; k < K_; k++) {
      const int count = doc_m_topics_count[k];
      if (count == 0) {
        continue;
      }
      std::vector<int>& docs_topic_k_count_hist = docs_topic_count_hist_[k];
      if ((int)docs_topic_k_count_hist.size() <= count) {
        docs_topic_k_count_hist.resize(count + 1);
      }
      ++docs_topic_k_count_hist[count];
    }

    if (doc.N) {
      if ((int)doc_len_hist_.size() <= doc.N) {
        doc_len_hist_.resize(doc.N + 1);
      }
      ++doc_len_hist_[doc.N];
    }
  }
}

/************************************************************************/
/* SparseLDASampler */
/************************************************************************/
int SparseLDASampler::InitializeSampler() {
  smooth_bucket_.resize(K_);
  doc_bucket_.resize(K_);
  word_bucket_.resize(K_);
  cache_.resize(K_);
  PrepareSmoothBucket();
  return 0;
}

void SparseLDASampler::PostSampleCorpus() {
  PlainGibbsSampler::PostSampleCorpus();

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

  PlainGibbsSampler::HPOpt_PostSampleDocument(m);
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
  double& smooth_bucket_k = smooth_bucket_[k];
  double& doc_bucket_k = doc_bucket_[k];
  double hp_alpha_k = hp_alpha_[k];
  int doc_topic_count;
  int topic_count;

  smooth_sum_ -= smooth_bucket_k;
  doc_bucket_sum_ -= doc_bucket_k;

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
  doc_bucket_sum_ += doc_bucket_k;
  cache_[k] = (doc_topic_count + hp_alpha_k) / (topic_count + hp_sum_beta_);
}

int SparseLDASampler::SampleDocumentWord(int m, int v) {
  double sum = smooth_sum_ + doc_bucket_sum_ + word_bucket_sum_;
  double sample = Rand::Double01() * sum;
  int new_k = -1;

  if (sample < word_bucket_sum_) {
    const IntTable& word_v_topics_count = words_topics_count_[v];
    IntTable::const_iterator first = word_v_topics_count.begin();
    IntTable::const_iterator last = word_v_topics_count.end();
    for (; first != last; ++first) {
      const int k = first.id();
      sample -= word_bucket_[k];
      if (sample <= 0.0) {
        break;
      }
    }
    new_k = first.id();
  } else {
    sample -= word_bucket_sum_;
    if (sample < doc_bucket_sum_) {
      const IntTable& doc_m_topics_count = docs_topics_count_[m];
      IntTable::const_iterator first = doc_m_topics_count.begin();
      IntTable::const_iterator last = doc_m_topics_count.end();
      for (; first != last; ++first) {
        const int k = first.id();
        sample -= doc_bucket_[k];
        if (sample <= 0.0) {
          break;
        }
      }
      new_k = first.id();
    } else {
      sample -= doc_bucket_sum_;
      int k;
      for (k = 0; k < K_; k++) {
        sample -= smooth_bucket_[k];
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
    smooth_bucket_[k] = pdf;
    smooth_sum_ += pdf;
    cache_[k] = tmp;
  }
}

void SparseLDASampler::PrepareDocBucket(int m) {
  doc_bucket_sum_ = 0.0;
  doc_bucket_.assign(K_, 0);
  const IntTable& doc_m_topics_count = docs_topics_count_[m];
  IntTable::const_iterator first = doc_m_topics_count.begin();
  IntTable::const_iterator last = doc_m_topics_count.end();
  for (; first != last; ++first) {
    const int k = first.id();
    const double tmp = topics_count_[k] + hp_sum_beta_;
    const double pdf = first.count() * hp_beta_ / tmp;
    doc_bucket_[k] = pdf;
    doc_bucket_sum_ += pdf;
    cache_[k] = (first.count() + hp_alpha_[k]) / tmp;
  }
}

void SparseLDASampler::PrepareWordBucket(int v) {
  word_bucket_sum_ = 0.0;
  word_bucket_.assign(K_, 0);
  const IntTable& word_v_topics_count = words_topics_count_[v];
  IntTable::const_iterator first = word_v_topics_count.begin();
  IntTable::const_iterator last = word_v_topics_count.end();
  for (; first != last; ++first) {
    const int k = first.id();
    const double pdf = first.count() * cache_[k];
    word_bucket_[k] = pdf;
    word_bucket_sum_ += pdf;
  }
}

/************************************************************************/
/* LightLDASampler */
/************************************************************************/
int LightLDASampler::InitializeSampler() {
  hp_alpha_alias_table_.Construct(hp_alpha_, hp_sum_alpha_);
  cached_words_topic_samples_.resize(V_);
  if (mh_step_ == 0) {
    mh_step_ = 8;
  }
  return 0;
}

void LightLDASampler::PostSampleCorpus() {
  PlainGibbsSampler::PostSampleCorpus();

  if (HPOpt_Enabled()) {
    if (hp_opt_alpha_iteration_ > 0) {
      hp_alpha_alias_table_.Construct(hp_alpha_, hp_sum_alpha_);
    }
  }
}

void LightLDASampler::SampleDocument(int m) {
  const Doc& doc = docs_[m];
  Word* word = &words_[doc.index];
  IntTable& doc_m_topics_count = docs_topics_count_[m];

  for (int n = 0; n < doc.N; n++, word++) {
    const int v = word->v;
    IntTable& word_v_topics_count = words_topics_count_[v];
    const int old_k = word->k;
    int s = word->k;
    int t;
    int N_ms;
    int N_mt;
    int N_vs;
    int N_vt;
    int N_s;
    int N_t;
    double hp_alpha_s;
    double hp_alpha_t;
    double accept_rate;

    for (int step = 0; step < mh_step_; step++) {
      // sample new topic from word proposal
      t = SampleWithWord(v);

      if (s != t) {
        // calculate accept rate from topic s to topic t:
        // (N^{-mn}_{mt} + \alpha_t)(N^{-mn}_{vt} + \beta)
        // -----------------------------------------------
        // (N^{-mn}_{ms} + \alpha_s)(N^{-mn}_{vs} + \beta)
        // *
        // (N^{-mn}_s + \sum\beta)
        // -----------------------
        // (N^{-mn}_t + \sum\beta)
        // *
        // (N_{vs} + \beta)(N_t + \sum\beta)
        // ---------------------------------
        // (N_{vt} + \beta)(N_s + \sum\beta)
        N_ms = doc_m_topics_count[s];
        N_mt = doc_m_topics_count[t];
        N_vs = word_v_topics_count[s];
        N_vt = word_v_topics_count[t];
        N_s = topics_count_[s];
        N_t = topics_count_[t];
        hp_alpha_s = hp_alpha_[s];
        hp_alpha_t = hp_alpha_[t];
        // part 4, 5
        accept_rate = (N_vs + hp_beta_) / (N_vt + hp_beta_)
                      * (N_t + hp_sum_beta_) / (N_s + hp_sum_beta_);
        // NOTE that: s != t
        if (old_k == s) {
          N_ms--;
          N_vs--;
          N_s--;
        } else if (old_k == t) {
          N_mt--;
          N_vt--;
          N_t--;
        }
        // part 1, 2, 3
        accept_rate *=
          (N_mt + hp_alpha_t) / (N_ms + hp_alpha_s)
          * (N_vt + hp_beta_) / (N_vs + hp_beta_)
          * (N_s + hp_sum_beta_) / (N_t + hp_sum_beta_);

        if (/*accept_rate >= 1.0 || */Rand::Double01() < accept_rate) {
          word->k = t;
          s = t;
        }
      }

      t = SampleWithDoc(doc, v);
      if (s != t) {
        // calculate accept rate from topic s to topic t:
        // (N^{-mn}_{mt} + \alpha_t)(N^{-mn}_{vt} + \beta)
        // -----------------------------------------------
        // (N^{-mn}_{ms} + \alpha_s)(N^{-mn}_{vs} + \beta)
        // *
        // (N^{-mn}_s + \sum\beta)
        // -----------------------
        // (N^{-mn}_t + \sum\beta)
        // *
        // (N_{ms} + \alpha_s)
        // -------------------
        // (N_{mt} + \alpha_t)
        N_ms = doc_m_topics_count[s];
        N_mt = doc_m_topics_count[t];
        N_vs = word_v_topics_count[s];
        N_vt = word_v_topics_count[t];
        N_s = topics_count_[s];
        N_t = topics_count_[t];
        hp_alpha_s = hp_alpha_[s];
        hp_alpha_t = hp_alpha_[t];
        // part 4
        accept_rate = (N_ms + hp_alpha_s) / (N_mt + hp_alpha_t);
        // NOTE that: s != t
        if (old_k == s) {
          N_ms--;
          N_vs--;
          N_s--;
        } else if (old_k == t) {
          N_mt--;
          N_vt--;
          N_t--;
        }
        // part 1, 2, 3
        accept_rate *=
          (N_mt + hp_alpha_t) / (N_ms + hp_alpha_s)
          * (N_vt + hp_beta_) / (N_vs + hp_beta_)
          * (N_s + hp_sum_beta_) / (N_t + hp_sum_beta_);

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

int LightLDASampler::SampleWithWord(int v) {
  // word-proposal: (N_vk + beta)/(N_k + sum_beta)
  std::vector<int>& word_v_topic_samples_ = cached_words_topic_samples_[v];
  if (word_v_topic_samples_.empty()) {
    word_topics_prob_.assign(K_, 0.0);
    const IntTable& word_v_topics_count = words_topics_count_[v];
    IntTable::const_iterator first = word_v_topics_count.begin();
    IntTable::const_iterator last = word_v_topics_count.end();
    for (; first != last; ++first) {
      const int k = first.id();
      word_topics_prob_[k] =
        (first.count() + hp_beta_) / (topics_count_[k] + hp_sum_beta_);
    }

    for (int k = 0; k < K_; k++) {
      double& prob_k = word_topics_prob_[k];
      if (prob_k == 0.0) {
        prob_k = hp_beta_ / (topics_count_[k] + hp_sum_beta_);
      }
    }

    word_alias_table_.Construct(word_topics_prob_);
    int cached_samples = K_ * mh_step_;
    word_v_topic_samples_.reserve(cached_samples);
    for (int i = 0; i < cached_samples; i++) {
      word_v_topic_samples_.push_back(word_alias_table_.Sample());
    }
  }

  const int new_k = word_v_topic_samples_.back();
  word_v_topic_samples_.pop_back();
  return new_k;
}

int LightLDASampler::SampleWithDoc(const Doc& doc, int v) {
  // doc-proposal: N_mk + alpha_k
  double sum = hp_sum_alpha_ + doc.N;
  double sample = Rand::Double01() * sum;
  if (sample < hp_sum_alpha_) {
    return hp_alpha_alias_table_.Sample(
             sample / hp_sum_alpha_,
             Rand::Double01());
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
