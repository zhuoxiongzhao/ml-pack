// Copyright (c) 2015 Tencent Inc.
// Author: Yafei Zhang (zhangyafeikimi@gmail.com)
//

#include "common/line-reader.h"
#include "common/x.h"
#include "lda/rand.h"
#include "lda/sampler.h"

SamplerBase::~SamplerBase() {}

void SamplerBase::LoadCorpus(FILE* fp, int with_id) {
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

void SamplerBase::SaveModel(const std::string& prefix) const {
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

int SamplerBase::Initialize() {
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

int SamplerBase::InitializeSampler() {
  return 0;
}

void SamplerBase::CollectTheta(Array2D<double>* theta) const {
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

void SamplerBase::CollectPhi(Array2D<double>* phi) const {
  for (int k = 0; k < K_; k++) {
    const int topics_count_k = topics_count_[k];
    double* phi_k = (*phi)[k];
    for (int v = 0; v < V_; v++) {
      phi_k[v] = (words_topics_count_[v][k] + hp_beta_)
                 / (topics_count_k + hp_sum_beta_);
    }
  }
}

double SamplerBase::LogLikelihood() const {
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

int SamplerBase::Train() {
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

void SamplerBase::PreSampleCorpus() {
  Log("Iteration %d started.\n", iteration_);
  HPOpt_Initialize();
}

void SamplerBase::PostSampleCorpus() {
  HPOpt_Optimize();
  if (iteration_ > burnin_iteration_
      && iteration_ % log_likelihood_interval_ == 0) {
    const double llh = LogLikelihood();
    Log("LogLikelihood(total/word)=%lg/%lg\n", llh, llh / words_.size());
  }
}

void SamplerBase::SampleCorpus() {
  for (int m = 0; m < M_; m++) {
    PreSampleDocument(m);
    SampleDocument(m);
    PostSampleDocument(m);
  }
}

void SamplerBase::PreSampleDocument(int m) {}

void SamplerBase::PostSampleDocument(int m) {
  HPOpt_PostSampleDocument(m);
}

void SamplerBase::HPOpt_Initialize() {
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

void SamplerBase::HPOpt_Optimize() {
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

void SamplerBase::HPOpt_OptimizeAlpha() {
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

void SamplerBase::HPOpt_PrepareOptimizeBeta() {
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

void SamplerBase::HPOpt_OptimizeBeta() {
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

void SamplerBase::HPOpt_PostSampleDocument(int m) {
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
