// Copyright (c) 2015 Tencent Inc.
// Author: Yafei Zhang (zhangyafeikimi@gmail.com)
//

#include <algorithm>
#include "lda/hist.h"

void DenseHist::Inc(int id, int count) {
  storage_[id] += count;
}
void DenseHist::Dec(int id, int count) {
  storage_[id] -= count;
  assert(storage_[id] >= 0);
}
int DenseHist::Count(int id) const {
  return storage_[id];
}
int DenseHist::NextNonZeroCountIndex(int id) const {
  while (storage_[id] == 0) {
    id++;
  }
  return id;
}
int DenseHist::Size() const {
  return (int)storage_.size();
}
int DenseHist::GetId(int i) const {
  return i;
}
int DenseHist::GetCount(int i) const {
  return storage_[i];
}

void ArrayBufHist::Inc(int id, int count) {
  storage_[id] += count;
}
void ArrayBufHist::Dec(int id, int count) {
  storage_[id] -= count;
  assert(storage_[id] >= 0);
}
int ArrayBufHist::Count(int id) const {
  return storage_[id];
}
int ArrayBufHist::NextNonZeroCountIndex(int id) const {
  while (storage_[id] == 0) {
    id++;
  }
  return id;
}
int ArrayBufHist::Size() const {
  return size_;
}
int ArrayBufHist::GetId(int i) const {
  return i;
}
int ArrayBufHist::GetCount(int i) const {
  return storage_[i];
}

void SparseHist::Inc(int id, int count) {
  std::vector<IdCount>::iterator it =
    std::lower_bound(storage_.begin(), storage_.end(), id, compare_);
  if (it != storage_.end() && it->id == id) {
    it->count += count;
  } else {
    IdCount target = {id, count};
    storage_.insert(it, target);
  }
}
void SparseHist::Dec(int id, int count) {
  std::vector<IdCount>::iterator it =
    std::lower_bound(storage_.begin(), storage_.end(), id, compare_);
  if (it != storage_.end() && it->id == id) {
    assert(it->count >= count);
    it->count -= count;
    if (it->count == 0) {
      storage_.erase(it);
    }
  } else {
    assert(0);
  }
}
int SparseHist::Count(int id) const {
  std::vector<IdCount>::const_iterator it =
    std::lower_bound(storage_.begin(), storage_.end(), id, compare_);
  if (it != storage_.end() && it->id == id) {
    assert(it->count > 0);
    return it->count;
  }
  return 0;
}
int SparseHist::NextNonZeroCountIndex(int id) const {
  return id;
}
int SparseHist::Size() const {
  return (int)storage_.size();
}
int SparseHist::GetId(int i) const {
  return storage_[i].id;
}
int SparseHist::GetCount(int i) const {
  return storage_[i].count;
}
