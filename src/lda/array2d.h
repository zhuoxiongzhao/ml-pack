// Copyright (c) 2015 Tencent Inc.
// Author: Yafei Zhang (zhangyafeikimi@gmail.com)
//
// 2d array
//

#ifndef SRC_LDA_ARRAY2D_H_
#define SRC_LDA_ARRAY2D_H_

#include <vector>

template <class T>
class Array2D {
 private:
  Array2D(const Array2D& right);
  Array2D& operator=(const Array2D& right);

 private:
  int d1_;
  int d2_;
  std::vector<T> storage_;

 public:
  Array2D() : d1_(0), d2_(0) {}

  void Resize(int d1, int d2, T t = 0) {
    d1_ = d1;
    d2_ = d2;
    storage_.resize(d1 * d2, t);
  }

  T* operator[](int i) {
    return &storage_[0] + d2_ * i;
  }

  const T* operator[](int i) const {
    return &storage_[0] + d2_ * i;
  }
};

#endif  // SRC_LDA_ARRAY2D_H_
