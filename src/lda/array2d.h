// Copyright (c) 2015 Tencent Inc.
// Author: Yafei Zhang (zhangyafeikimi@gmail.com)
//
// 2d array
//

#ifndef SRC_LDA_ARRAY2D_H_
#define SRC_LDA_ARRAY2D_H_

#include <vector>

template <class T>
class Array1D {
 private:
  T* a_;
 public:
  Array1D() {}

  explicit Array1D(T* a) : a_(a) {}

  T& operator[](int i) {
    return a_[i];
  }

  T operator[](int i) const {
    return a_[i];
  }
};

template <class T>
class ConstArray1D {
 private:
  const T* a_;
 public:
  ConstArray1D() {}

  explicit ConstArray1D(const T* a) : a_(a) {}

  T operator[](int i) {
    return a_[i];
  }

  T operator[](int i) const {
    return a_[i];
  }
};

template <class T>
class Array2D {
 private:
  int d1_;
  int d2_;
  std::vector<T> a_;

 public:
  void resize(int d1, int d2) {
    d1_ = d1;
    d2_ = d2;
    a_.resize(d1 * d2);
  }

  Array1D<T> operator[](int i) {
    return Array1D<T>(&a_[0] + d2_ * i);
  }

  ConstArray1D<T> operator[](int i) const {
    return ConstArray1D<T>(&a_[0] + d2_ * i);
  }
};

#endif  // SRC_LDA_ARRAY2D_H_
