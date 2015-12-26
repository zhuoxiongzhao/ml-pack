// Copyright (c) 2015 Tencent Inc.
// Author: Yafei Zhang (zhangyafeikimi@gmail.com)
//
// array, table, tables
//

#ifndef SRC_LDA_ARRAY_H_
#define SRC_LDA_ARRAY_H_

#include <assert.h>
#include <algorithm>
#include <vector>

// A cache-friendly array,
// in which bytes are coherent in memory.
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

  void Init(int d1, int d2, T t = 0) {
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

template <class T>
class ITable {
 private:
  ITable(const ITable& right);
  ITable& operator=(const ITable& right);

 public:
  ITable() {}
  virtual ~ITable() {}
  virtual T Inc(int id, T count) = 0;
  virtual T Dec(int id, T count) = 0;
  virtual T Count(int id) const = 0;
  virtual int NextNonZeroCountIndex(int id) const = 0;
  virtual int Size() const = 0;
  virtual int GetId(int i) const = 0;
  virtual T GetCount(int i) const = 0;
};

template <class T>
class DenseTable : public ITable<T> {
 private:
  typedef ITable<T> ITableT;
  std::vector<T> storage_;

 public:
  DenseTable() {}

  void Init(int size) {
    storage_.resize(size);
  }

  virtual T Inc(int id, T count) {
    T& r = storage_[id];
    r += count;
    return r;
  }
  virtual T Dec(int id, T count) {
    T& r = storage_[id];
    r -= count;
    assert(r >= 0);
    return r;
  }
  virtual T Count(int id) const {
    return storage_[id];
  }
  virtual int NextNonZeroCountIndex(int id) const {
    while (storage_[id] == 0) {
      id++;
    }
    return id;
  }
  virtual int Size() const {
    return (int)storage_.size();
  }
  virtual int GetId(int i) const {
    return i;
  }
  virtual T GetCount(int i) const {
    return storage_[i];
  }

#ifdef ENABLE_TABLE_PROXY
  class __Proxy {
   private:
    ITableT* impl_;
    int id_;

   public:
    __Proxy(ITableT* impl, int id) : impl_(impl), id_(id) {}

    T operator++() {
      return impl_->Inc(id_, 1);
    }

    T operator--() {
      return impl_->Dec(id_, 1);
    }

    operator T() {
      return impl_->Count(id_);
    }
  };

  __Proxy operator[](int id) {
    return __Proxy(this, id);
  }
#else
  T& operator[](int id) {
    return storage_[id];
  }
#endif
  T operator[](int id) const {
    return storage_[id];
  }
};

template <class T>
class ArrayBufTable : public ITable<T> {
 private:
  typedef ITable<T> ITableT;
  T* storage_;
  int size_;

 public:
  ArrayBufTable(T* storage, int size)
    : storage_(storage), size_(size) {}

  virtual T Inc(int id, T count) {
    T& r = storage_[id];
    r += count;
    return r;
  }
  virtual T Dec(int id, T count) {
    T& r = storage_[id];
    r -= count;
    assert(r >= 0);
    return r;
  }
  virtual T Count(int id) const {
    return storage_[id];
  }
  virtual int NextNonZeroCountIndex(int id) const {
    while (storage_[id] == 0) {
      id++;
    }
    return id;
  }
  virtual int Size() const {
    return size_;
  }
  virtual int GetId(int i) const {
    return i;
  }
  virtual T GetCount(int i) const {
    return storage_[i];
  }
};

template <class T>
class SparseTable : public ITable<T> {
 private:
  typedef ITable<T> ITableT;
  struct IdCount {
    int id;
    T count;
  };

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
  SparseTable() : compare_() {}

  virtual T Inc(int id, T count) {
    typename std::vector<IdCount>::iterator it =
      std::lower_bound(storage_.begin(), storage_.end(), id, compare_);
    if (it != storage_.end() && it->id == id) {
      it->count += count;
      return it->count;
    } else {
      IdCount target = {id, count};
      storage_.insert(it, target);
      return count;
    }
  }
  virtual T Dec(int id, T count) {
    typename std::vector<IdCount>::iterator it =
      std::lower_bound(storage_.begin(), storage_.end(), id, compare_);
    if (it != storage_.end() && it->id == id) {
      assert(it->count >= count);
      it->count -= count;
      if (it->count == 0) {
        storage_.erase(it);
        return 0;
      } else {
        return it->count;
      }
    } else {
      assert(0);
      return -1;
    }
  }
  virtual T Count(int id) const {
    typename std::vector<IdCount>::const_iterator it =
      std::lower_bound(storage_.begin(), storage_.end(), id, compare_);
    if (it != storage_.end() && it->id == id) {
      assert(it->count > 0);
      return it->count;
    }
    return 0;
  }
  virtual int NextNonZeroCountIndex(int id) const {
    return id;
  }
  virtual int Size() const {
    return (int)storage_.size();
  }
  virtual int GetId(int i) const {
    return storage_[i].id;
  }
  virtual T GetCount(int i) const {
    return storage_[i].count;
  }
};

template <class T>
class Table {
 private:
  typedef ITable<T> ITableT;
  typedef DenseTable<T> DenseTableT;
  typedef ArrayBufTable<T> ArrayBufTableT;
  typedef SparseTable<T> SparseTableT;
  ITableT* impl_;

 public:
  Table() : impl_(NULL) {}

  Table(const Table& right) : impl_(NULL) {
    // disable actual copy
    assert(right.impl_ == NULL);
  }

  Table& operator=(const Table& right) {
    // disable actual copy
    assert(impl_ == NULL);
    assert(right.impl_ == NULL);
    return *this;
  }

  ~Table() {
    delete impl_;
  }

  void InitDense(int size) {
    DenseTableT* hist = new DenseTableT();
    hist->Init(size);
    impl_ = hist;
  }

  void InitArrayBuf(T* storage, int size) {
    impl_ = new ArrayBufTableT(storage, size);
  }

  void InitSparse() {
    impl_ = new SparseTableT();
  }

 public:
  class const_iterator {
   private:
    const ITableT* impl_;
    int index_;

   public:
    const_iterator(const ITableT* impl, int index)
      : impl_(impl), index_(index) {}

    int id() const {
      return impl_->GetId(index_);
    }

    T count() const {
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

  const_iterator begin() const {
    return const_iterator(impl_, 0);
  }

  const_iterator end() const {
    return const_iterator(impl_, impl_->Size());
  }

  class __Proxy {
   private:
    ITableT* impl_;
    int id_;

   public:
    __Proxy(ITableT* impl, int id) : impl_(impl), id_(id) {}

    T operator++() {
      return impl_->Inc(id_, 1);
    }

    T operator+=(int count) {
      return impl_->Inc(id_, count);
    }

    T operator--() {
      return impl_->Dec(id_, 1);
    }

    T operator-=(int count) {
      return impl_->Dec(id_, count);
    }

    operator T() {
      return impl_->Count(id_);
    }
  };

  __Proxy operator[](int id) {
    return __Proxy(impl_, id);
  }

  T operator[](int id) const {
    return impl_->Count(id);
  }
};

enum TableType {
  kDenseHist = 1,
  kArrayBufHist,
  kSparseHist
};

template <class T>
class Tables {
 public:
  typedef Table<T> TableT;

 private:
  Tables(const Tables& right);
  Tables& operator=(const Tables& right);

 private:
  int d1_;
  int d2_;
  std::vector<TableT> matrix_;
  std::vector<int> array_buf_;

 public:
  Tables() : d1_(0), d2_(0) {}

  void Init(int d1, int d2, int type) {
    d1_ = d1;
    d2_ = d2;
    matrix_.resize(d1);

    if (type == kDenseHist) {
      for (int i = 0; i < d1_; i++) {
        matrix_[i].InitDense(d2_);
      }
    } else if (type == kArrayBufHist) {
      array_buf_.resize(d1_ * d2_);
      for (int i = 0; i < d1_; i++) {
        matrix_[i].InitArrayBuf(&array_buf_[0] + i * d2_, d2_);
      }
    } else {
      for (int i = 0; i < d1_; i++) {
        matrix_[i].InitSparse();
      }
    }
  }

  TableT& operator[](int i) {
    return matrix_[i];
  }

  const TableT& operator[](int i) const {
    return matrix_[i];
  }
};

typedef DenseTable<int> IntDenseTable;
typedef ArrayBufTable<int> IntArrayBufTable;
typedef SparseTable<int> IntSparseTable;
typedef Table<int> IntTable;
typedef Tables<int> IntTables;

#endif  // SRC_LDA_ARRAY_H_
