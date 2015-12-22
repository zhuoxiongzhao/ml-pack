// Copyright (c) 2015 Tencent Inc.
// Author: Yafei Zhang (zhangyafeikimi@gmail.com)
//

#include "common/line-reader.h"
#include "common/mt19937ar.h"
#include "common/mt64.h"
#include "common/x.h"
#include "lda/sampler.h"

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
    theta_mk.Resize(M_, K_);
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
    phi_kv.Resize(K_, V_);
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
  Log("Done.\n");
}

int PlainGibbsSampler::InitializeParam() {
  if (K_ == 0) {
    return -1;
  }

  N_k_.resize(K_);

  if (hp_total_alpha_ <= 0.0) {
    double avg_doc_len = (double)words_.size() / docs_.size();
    hp_alpha_k_.resize(K_, avg_doc_len / K_);
    hp_total_alpha_ = avg_doc_len;
  } else {
    hp_alpha_k_.resize(K_, hp_total_alpha_);
    hp_total_alpha_ = hp_total_alpha_ * K_;
  }

  if (hp_beta_ <= 0.0) {
    hp_beta_ = 0.01;
  }
  hp_total_beta_ = V_ * hp_beta_;

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
    total_iteration_ = 50;
  }
  if (burnin_iteration_ == 0) {
    burnin_iteration_ = 10;
  }
  if (log_likelyhood_interval_ == 0) {
    log_likelyhood_interval_ = total_iteration_ / 10;
  }
  iteration_ = 1;
  return 0;
}

int PlainGibbsSampler::InitializeSampler() {
  N_mk_.Init(M_, K_, hist_type_);
  N_vk_.Init(V_, K_, hist_type_);
  topic_cdf_.resize(K_);

  for (int m = 0; m < M_; m++) {
    const Doc& doc = docs_[m];
    Word* word = &words_[doc.index];
    Hist& N_m = N_mk_[m];

    for (int n = 0; n < doc.N; n++, word++) {
      int v = word->v;
      int k = (int)(genrand_int32() % K_);
      word->k = k;
      N_m[k]++;
      N_vk_[v][k]++;
      N_k_[k]++;
    }
  }
  return 0;
}

void PlainGibbsSampler::UninitializeSampler() {
}

void PlainGibbsSampler::CollectTheta(Array2D<double>* theta) const {
  for (int m = 0; m < M_; m++) {
    const Doc& doc = docs_[m];
    double* theta_m = (*theta)[m];
    const Hist& N_m = N_mk_[m];

    for (int k = 0; k < K_; k++) {
      theta_m[k] += (N_m[k] + hp_alpha_k_[k]) / (doc.N + hp_total_alpha_);
    }
  }
}

void PlainGibbsSampler::CollectPhi(Array2D<double>* phi) const {
  for (int k = 0; k < K_; k++) {
    double* phi_k = (*phi)[k];

    for (int v = 0; v < V_; v++) {
      phi_k[v] += (N_vk_[v][k] + hp_beta_) / (N_k_[k] + hp_total_beta_);
    }
  }
}

double PlainGibbsSampler::LogLikelyhood() const {
  double sum = 0.0;
  for (int m = 0; m < M_; m++) {
    const Doc& doc = docs_[m];
    const Word* word = &words_[doc.index];
    const Hist& N_m = N_mk_[m];

    for (int n = 0; n < doc.N; n++, word++) {
      int v = word->v;
      double word_sum = 0.0;
      const Hist& N_v = N_vk_[v];
      for (int k = 0; k < K_; k++) {
        word_sum += (N_m[k] + hp_alpha_k_[k])
                    * (N_v[k] + hp_beta_)
                    / (N_k_[k] + hp_total_beta_);
      }
      word_sum /= (doc.N + hp_total_alpha_);
      sum += log(word_sum);
    }
  }
  return sum;
}

int PlainGibbsSampler::Train() {
  if (InitializeParam() != 0) {
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

  UninitializeSampler();
  return 0;
}

void PlainGibbsSampler::PreSampleCorpus() {
  Log("Iteration %d started.\n", iteration_);
  HPOpt_Initialize();
}

void PlainGibbsSampler::PostSampleCorpus() {
  HPOpt_Optimize();

  if (iteration_ > burnin_iteration_
      && iteration_ % log_likelyhood_interval_ == 0) {
    Log("LogLikelyhood=%lg\n", LogLikelyhood());
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
  Hist& N_m = N_mk_[m];
  double talpha = doc.N - 1 + hp_total_alpha_;

  for (int n = 0; n < doc.N; n++, word++) {
    int v = word->v;
    int k = word->k;
    const Hist& N_v = N_vk_[v];

    N_m[k]--;
    N_vk_[v][k]--;
    N_k_[k]--;

    topic_cdf_[0] = 0.0;
    for (k = 0; k < K_ - 1; k++) {
      topic_cdf_[k] += (N_v[k] + hp_beta_)
                       / (N_k_[k] + hp_total_beta_)
                       * (N_m[k] + hp_alpha_k_[k])
                       / talpha;
      topic_cdf_[k + 1] = topic_cdf_[k];
    }
    topic_cdf_[K_ - 1] += (N_v[K_ - 1] + hp_beta_)
                          / (N_k_[K_ - 1] + hp_total_beta_)
                          * (N_m[K_ - 1] + hp_alpha_k_[K_ - 1])
                          / talpha;

    double r = genrand64_real3() * topic_cdf_[K_ - 1];
    for (k = 0; k < K_; k++) {
      if (topic_cdf_[k] >= r) {
        break;
      }
    }

    N_m[k]++;
    N_vk_[v][k]++;
    N_k_[k]++;
    word->k = k;
  }
}

void PlainGibbsSampler::HPOpt_Initialize() {
  if (!HPOpt_Enabled()) {
    return;
  }

  Log("Hyper optimization will be carried out in this iteration.\n");
  hp_opt_topic_doc_count_.clear();
  hp_opt_topic_doc_count_.resize(K_);
  hp_opt_doc_len_count_.clear();
  hp_opt_word_topic_count_.clear();
  hp_opt_topic_len_count_.clear();
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
    for (int j = 1, size = (int)hp_opt_doc_len_count_.size();
         j < size; j++) {
      diff_digamma += 1.0 / (j - 1 + hp_total_alpha_);
      denom += hp_opt_doc_len_count_[j] * diff_digamma;
    }
    denom -= 1.0 / hp_opt_alpha_scale_;

    hp_total_alpha_ = 0.0;
    for (int k = 0, size = (int)hp_opt_topic_doc_count_.size();
         k < size; k++) {
      double num = 0.0;
      diff_digamma = 0.0;
      for (int j = 1, size = (int)hp_opt_topic_doc_count_[k].size();
           j < size; j++) {
        diff_digamma += 1.0 / (j - 1 + hp_alpha_k_[k]);
        num += hp_opt_topic_doc_count_[k][j] *  diff_digamma;
      }
      hp_alpha_k_[k] = (hp_alpha_k_[k] * num + hp_opt_alpha_shape_) / denom;
      hp_total_alpha_ += hp_alpha_k_[k];
    }
  }
}

void PlainGibbsSampler::HPOpt_PrepareOptimizeBeta() {
  for (int m = 0; m < M_; m++) {
    Hist& N_m = N_mk_[m];
    for (int k = 0; k < K_; k++) {
      int count = N_m[k];
      if (count == 0) {
        continue;
      }
      if ((int)hp_opt_word_topic_count_.size() <= count) {
        hp_opt_word_topic_count_.resize(count + 1);
      }
      hp_opt_word_topic_count_[count]++;
    }
  }

  for (int k = 0; k < K_; k++) {
    int count = N_k_[k];
    if (count == 0) {
      continue;
    }
    if ((int)hp_opt_topic_len_count_.size() <= count) {
      hp_opt_topic_len_count_.resize(count + 1);
    }
    hp_opt_topic_len_count_[count]++;
  }
}

void PlainGibbsSampler::HPOpt_OptimizeBeta() {
  for (int i = 0; i < hp_opt_beta_iteration_; i++) {
    double num = 0.0;
    double diff_digamma = 0.0;
    for (int j = 1, size = (int)hp_opt_word_topic_count_.size();
         j < size; j++) {
      diff_digamma += 1.0 / (j - 1 + hp_beta_);
      num += diff_digamma * hp_opt_word_topic_count_[j];
    }

    double denom = 0.0;
    diff_digamma = 0.0;
    for (int j = 1, size = (int)hp_opt_topic_len_count_.size();
         j < size; j++) {
      diff_digamma += 1.0 / (j - 1 + hp_total_beta_);
      denom += diff_digamma * hp_opt_topic_len_count_[j];
    }
    hp_total_beta_ = hp_beta_ * num / denom;
    hp_beta_ = hp_total_beta_ / V_;
  }
}

void PlainGibbsSampler::HPOpt_PostSampleDocument(int m) {
  if (!HPOpt_Enabled()) {
    return;
  }

  if (hp_opt_alpha_iteration_ > 0) {
    const Doc& doc = docs_[m];
    Hist& N_m = N_mk_[m];
    for (int k = 0; k < K_; k++) {
      int count = N_m[k];
      if (count == 0) {
        continue;
      }
      std::vector<int>& doc_count = hp_opt_topic_doc_count_[k];
      if ((int)doc_count.size() <= count) {
        doc_count.resize(count + 1);
      }
      doc_count[count]++;
    }

    if (doc.N) {
      if ((int)hp_opt_doc_len_count_.size() <= doc.N) {
        hp_opt_doc_len_count_.resize(doc.N + 1);
      }
      hp_opt_doc_len_count_[doc.N]++;
    }
  }
}
