// Copyright (c) 2015 Tencent Inc.
// Author: Yafei Zhang (zhangyafeikimi@gmail.com)
//
// id-count histogram
//

#ifndef SRC_LDA_HIST_H_
#define SRC_LDA_HIST_H_

#include <assert.h>
#include <vector>

class IHist {
 private:
  IHist(const IHist& right);
  IHist& operator=(const IHist& right);

 public:
  IHist() {}
  virtual ~IHist() {}
  virtual void Inc(int id, int count) = 0;
  virtual void Dec(int id, int count) = 0;
  virtual int Count(int id) const = 0;
  virtual int NextNonZeroCountIndex(int id) const = 0;
  virtual int Size() const = 0;
  virtual int GetId(int i) const = 0;
  virtual int GetCount(int i) const = 0;
};

class DenseHist : public IHist {
 private:
  std::vector<int> storage_;

 public:
  explicit DenseHist(int size) {
    storage_.resize(size, 0);
  }
  virtual void Inc(int id, int count);
  virtual void Dec(int id, int count);
  virtual int Count(int id) const;
  virtual int NextNonZeroCountIndex(int id) const;
  virtual int Size() const;
  virtual int GetId(int i) const;
  virtual int GetCount(int i) const;
};

class ArrayBufHist : public IHist {
 private:
  int* storage_;
  int size_;

 public:
  ArrayBufHist(int* storage, int size)
    : storage_(storage), size_(size) {}
  virtual void Inc(int id, int count);
  virtual void Dec(int id, int count);
  virtual int Count(int id) const;
  virtual int NextNonZeroCountIndex(int id) const;
  virtual int Size() const;
  virtual int GetId(int i) const;
  virtual int GetCount(int i) const;
};

struct IdCount {
  int id;
  int count;
};

class SparseHist : public IHist {
 private:
  struct IdCountCompare {
    bool operator()(const IdCount& a, const IdCount& b) const {
      return a.id < b.id;
    }
    bool operator()(const IdCount& a, int b) const {
      return a.id < b;
    }
    bool operator()(int a, const IdCount& b) const {
      return a < b.id;
    }
  };
  const IdCountCompare compare_;
  std::vector<IdCount> storage_;

 public:
  SparseHist() : compare_() {}
  virtual void Inc(int id, int count);
  virtual void Dec(int id, int count);
  virtual int Count(int id) const;
  virtual int NextNonZeroCountIndex(int id) const;
  virtual int Size() const;
  virtual int GetId(int i) const;
  virtual int GetCount(int i) const;
};

class Hist {
 private:
  IHist* impl_;

 public:
  Hist() : impl_(NULL) {}

  Hist(const Hist& right) : impl_(NULL) {
    // disable actual copy
    assert(right.impl_ == NULL);
  }

  Hist& operator=(const Hist& right) {
    // disable actual copy
    assert(impl_ == NULL);
    assert(right.impl_ == NULL);
    return *this;
  }

  ~Hist() {
    delete impl_;
  }

  void InitDense(int size) {
    impl_ = new DenseHist(size);
  }

  void InitArrayBuf(int* storage, int size) {
    impl_ = new ArrayBufHist(storage, size);
  }

  void InitSparse() {
    impl_ = new SparseHist();
  }

 public:
  class const_iterator {
   private:
    const IHist* impl_;
    int index_;

   public:
    const_iterator(const IHist* impl, int index)
      : impl_(impl), index_(index) {}

    int id() const {
      return impl_->GetId(index_);
    }

    int count() const {
      return impl_->GetCount(index_);
    }

    bool operator==(const const_iterator& right) const {
      return impl_ == right.impl_ && index_ == right.index_;
    }

    bool operator!=(const const_iterator& right) const {
      return !(impl_ == right.impl_ && index_ == right.index_);
    }

    const_iterator& operator++() {
      index_++;
      index_ = impl_->NextNonZeroCountIndex(index_);
      return *this;
    }
  };

  class __Proxy {
   private:
    IHist* impl_;
    int id_;

   public:
    __Proxy(IHist* impl, int id) : impl_(impl), id_(id) {}

    void operator++() {
      impl_->Inc(id_, 1);
    }

    void operator++(int) {
      impl_->Inc(id_, 1);
    }

    void operator+=(int count) {
      impl_->Inc(id_, count);
    }

    void operator--() {
      impl_->Dec(id_, 1);
    }

    void operator--(int) {
      impl_->Dec(id_, 1);
    }

    void operator-=(int count) {
      impl_->Dec(id_, count);
    }

    operator int() {
      return impl_->Count(id_);
    }
  };

  const_iterator begin() const {
    return const_iterator(impl_, 0);
  }

  const_iterator end() const {
    return const_iterator(impl_, impl_->Size());
  }

  __Proxy operator[](int id) {
    return __Proxy(impl_, id);
  }

  int operator[](int id) const {
    return impl_->Count(id);
  }
};

enum HistType {
  kDenseHist = 1,
  kArrayBufHist,
  kSparseHist
};

class Hists {
 private:
  Hists(const Hists& right);
  Hists& operator=(const Hists& right);

 private:
  int d1_;
  int d2_;
  std::vector<Hist> matrix_;
  std::vector<int> array_buf_;

 public:
  Hists() : d1_(0), d2_(0) {}

  void Init(int d1, int d2, int type) {
    d1_ = d1;
    d2_ = d2;
    matrix_.resize(d1);

    if (type == kDenseHist) {
      for (int i = 0; i < d1_; i++) {
        matrix_[i].InitDense(d2_);
      }
    } else if (type == kArrayBufHist) {
      array_buf_.resize(d1_ * d2_, 0);
      for (int i = 0; i < d1_; i++) {
        matrix_[i].InitArrayBuf(&array_buf_[0] + i * d2_, d2_);
      }
    } else {
      for (int i = 0; i < d1_; i++) {
        matrix_[i].InitSparse();
      }
    }
  }

  Hist& operator[](int i) {
    return matrix_[i];
  }

  const Hist& operator[](int i) const {
    return matrix_[i];
  }
};

#endif  // SRC_LDA_HIST_H_
