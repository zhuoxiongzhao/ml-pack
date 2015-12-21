// Copyright (c) 2015 Tencent Inc.
// Author: Yafei Zhang (zhangyafeikimi@gmail.com)
//
// id-count histogram
//

#ifndef SRC_LDA_HIST_H_
#define SRC_LDA_HIST_H_

#include <assert.h>
#include "lda/sorted-vector.h"

struct IdCount {
  int id;
  int count;
};

class SparseHist {
 private:
  struct IdCountPred {
    bool operator()(const IdCount& a, const IdCount& b) const {
      return a.id < b.id;
    }
  };

  typedef sorted_vector<IdCount, false, IdCountPred> Histgram;
  Histgram hist_;
  mutable int curr_;

 public:
  SparseHist() : curr_(0) {}

  void inc(int id, int count = 1) {
    IdCount target = {id, count};
    Histgram::iterator it = hist_.lower_bound(target);
    if (it != hist_.end() && it->id == id) {
      it->count += count;
    } else {
      hist_.insert(it, target);
    }
  }

  void dec(int id, int count = 1) {
    IdCount target = {id, count};
    Histgram::iterator it = hist_.find(target);
    if (it != hist_.end()) {
      assert(it->count >= count);
      it->count -= count;
      if (it->count == 0) {
        hist_.erase(it);
      }
    } else {
      assert(0);
    }
  }

  int count(int id) const {
    IdCount target = {id, 0};
    Histgram::const_iterator it = hist_.find(target);
    if (it != hist_.end()) {
      return it->count;
    }
    return 0;
  }

  void reset_cache() {
    curr_ = 0;
  }

  // this function is friendly to cache
  // under the circumstances of sequence calling like:
  // reset_cache()
  // cached_count(0)
  // cached_count(1)
  // cached_count(2)
  // ...
  // cached_count(N)
  int cached_count(int id) const {
    if (curr_ >= (int)hist_.size()) {
      return 0;
    }

    for (;;) {
      const IdCount& id_count = hist_[curr_];
      if (id_count.id > id) {
        return 0;
      } else if (id_count.id == id) {
        curr_++;
        return id_count.count;
      } else {
        curr_++;
      }
    }
  }

  int size() const {
    return (int)hist_.size();
  }

  IdCount& operator[](int i) {
    return hist_[i];
  }

  const IdCount& operator[](int i) const {
    return hist_[i];
  }
};

#endif  // SRC_LDA_HIST_H_
