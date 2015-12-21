// Copyright (c) 2015 Tencent Inc.
// Author: Yafei Zhang (zhangyafeikimi@gmail.com)
//
// sorted vector
//

#ifndef SRC_LDA_SORTED_VECTOR_H_
#define SRC_LDA_SORTED_VECTOR_H_

#include <algorithm>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

template<class K,
         bool Duplicate = false,
         class Pred = std::less<K>,
         class A = std::allocator<K> >
class sorted_vector {
 public:
  typedef Pred                                          key_compare;
  typedef Pred                                          value_compare;
  typedef sorted_vector<K, Duplicate, Pred, A>          self_type;
  typedef std::vector<K, A>                             container_type;
  typedef typename container_type::allocator_type       allocator_type;
  typedef typename container_type::size_type            size_type;
  typedef typename container_type::difference_type      difference_type;
  typedef typename container_type::reference            reference;
  typedef typename container_type::const_reference      const_reference;
  typedef typename container_type::value_type           key_type;
  typedef typename container_type::value_type           value_type;
  typedef typename container_type::iterator             iterator;
  typedef typename container_type::const_iterator       const_iterator;
  typedef typename container_type::const_reverse_iterator
  const_reverse_iterator;
  typedef typename container_type::reverse_iterator     reverse_iterator;

  typedef std::pair<iterator, iterator> pairii;
  typedef std::pair<const_iterator, const_iterator> paircc;
  typedef std::pair<iterator, bool> pairib;

 public:
  explicit sorted_vector(const Pred& pred = Pred(), const A& al = A())
    : key_compare_(pred), container_(al) {
  }

  template<class It>
  sorted_vector(It first, It beyond,
                const Pred& pred = Pred(), const A& al = A())
    : key_compare_(pred), container_(first, beyond, al) {
    stable_sort();
  }

  explicit sorted_vector(const self_type& x)
    : key_compare_(x.key_compare_), container_(x.container_) {
  }

  ~sorted_vector() {
  }

  self_type& operator=(const self_type& x) {
    container_.operator = (x.container_);
    key_compare_ = x.key_compare_;
    return *this;
  }

  self_type& operator=(const container_type& x) {
    container_.operator = (x);
    sort();
    return *this;
  }

  void reserve(size_type n) {
    container_.reserve(n);
  }
  iterator begin() {
    return container_.begin();
  }
  const_iterator begin() const {
    return container_.begin();
  }
  iterator end() {
    return container_.end();
  }
  const_iterator end() const {
    return container_.end();
  }
  reverse_iterator rbegin() {
    return container_.rbegin();
  }
  const_reverse_iterator rbegin() const {
    return container_.rbegin();
  }
  reverse_iterator rend() {
    return container_.rend();
  }
  const_reverse_iterator rend() const {
    return container_.rend();
  }

  size_type size() const {
    return container_.size();
  }
  size_type max_size() const {
    return container_.max_size();
  }
  bool empty() const {
    return container_.empty();
  }
  A get_allocator() const {
    return container_.get_allocator();
  }
  const_reference at(size_type p) const {
    return container_.at(p);
  }
  reference at(size_type p) {
    return container_.at(p);
  }
  const_reference operator[](size_type p) const {
    return container_.operator[](p);
  }
  reference operator[](size_type p) {
    return container_.operator[](p);
  }
  reference front() {
    return container_.front();
  }
  const_reference front() const {
    return container_.front();
  }
  reference back() {
    return container_.back();
  }
  const_reference back() const {
    return container_.back();
  }
  void pop_back() {
    container_.pop_back();
  }

  void assign(const_iterator first, const_iterator beyond) {
    container_.assign(first, beyond);
  }

  void assign(size_type n, const key_type& x = key_type()) {
    container_.assign(n, x);
  }

  pairib insert(const key_type& x) {
    if (Duplicate) {
      iterator p = lower_bound(x);
      if (p == end() || key_compare_(x, *p)) {
        return pairib(__insert(p, x), true);
      } else {
        return pairib(p, false);
      }
    } else {
      iterator p = upper_bound(x);
      return pairib(__insert(p, x), true);
    }
  }

  iterator insert(iterator it, const key_type& x) {
    if (it != end()) {
      if (Duplicate) {
        if (key_compare_(*it, x)) {
          if ((it + 1) == end() || key_great(*(it + 1), x)) {
            // use hint
            return __insert(it + 1, x);
          } else if (key_great_equal(*(it + 1), x)) {
            return end();
          }
        }
      } else {
        if (key_less_equal(*it, x) && ((it + 1) == end()
                                       || key_great_equal(*(it + 1), x))) {
          return __insert(it + 1, x);
        }
      }
    }
    return insert(x).first;
  }

  template<class It>
  void insert(It first, It beyond) {
    size_type n = std::distance(first, beyond);
    reserve(size() + n);
    for ( ; first != beyond; ++first) {
      insert(*first);
    }
  }

  iterator erase(iterator p) {
    return container_.erase(p);
  }
  iterator erase(iterator first, iterator beyond) {
    return container_.erase(first, beyond);
  }
  size_type erase(const key_type& key) {
    pairii begin_end = equal_range(key);
    size_type n = std::distance(begin_end.first, begin_end.second);
    erase(begin_end.first, begin_end.second);
    return n;
  }

  void clear() {
    return container_.clear();
  }

  bool equal(const self_type& x) const {
    return (size() == x.size() && std::equal(begin(), end(), x.begin()));
  }

  bool less_than(const self_type& x) const {
    return (std::lexicographical_compare(begin(), end(), x.begin(), x.end()));
  }

  void swap(self_type& x) {
    container_.swap(x.container_);
    std::swap(key_compare_, x.key_compare_);
  }

  friend void swap(self_type& x, self_type& y) {
    x.swap(y);
  }

  key_compare key_comp() const {
    return key_compare_;
  }
  value_compare value_comp() const {
    return (key_comp());
  }

  iterator find(const key_type& k) {
    iterator p = lower_bound(k);
    return (p == end() || key_compare_(k, *p)) ? end() : p;
  }

  const_iterator find(const key_type& k) const {
    const_iterator p = lower_bound(k);
    return (p == end() || key_compare_(k, *p)) ? end() : p;
  }

  size_type count(const key_type& k) const {
    paircc begin_end = equal_range(k);
    size_type n = std::distance(begin_end.first, begin_end.second);
    return (n);
  }

  iterator lower_bound(const key_type& k) {
    return std::lower_bound(begin(), end(), k, key_compare_);
  }
  const_iterator lower_bound(const key_type& k) const {
    return std::lower_bound(begin(), end(), k, key_compare_);
  }
  iterator upper_bound(const key_type& k) {
    return std::upper_bound(begin(), end(), k, key_compare_);
  }
  const_iterator upper_bound(const key_type& k) const {
    return std::upper_bound(begin(), end(), k, key_compare_);
  }
  pairii equal_range(const key_type& k) {
    return std::equal_range(begin(), end(), k, key_compare_);
  }
  paircc equal_range(const key_type& k) const {
    return std::equal_range(begin(), end(), k, key_compare_);
  }

  // functions for use with direct std::vector-access
  container_type& get_container() {
    return container_;
  }

 private:
  void sort() {
    std::sort(container_.begin(), container_.end(), key_compare_);
    if (Duplicate) {
      container_.erase(unique(), container_.end());
    }
  }

  void stable_sort() {
    std::stable_sort(container_.begin(), container_.end(), key_compare_);
    if (Duplicate) {
      erase(unique(), end());
    }
  }

  iterator unique() {
    iterator front = container_.begin();
    iterator out = container_.end();
    iterator end = container_.end();
    bool copy = false;
    for (iterator prev; (prev = front) != end && ++front != end; ) {
      if (key_compare_(*prev, *front)) {
        if (copy) {
          *out = *front;
          out++;
        }
      } else {
        if (!copy) {
          out = front;
          copy = true;
        }
      }
    }
    return out;
  }

  iterator __insert(iterator p, const key_type& x) {
    return container_.insert(p, x);
  }
  bool key_less_equal(const key_type& ty0, const key_type& ty1) {
    return !key_compare_(ty1, ty0);
  }
  bool key_great_equal(const key_type& ty0, const key_type& ty1) {
    return !key_compare_(ty0, ty1);
  }
  bool key_great(const key_type& ty0, const key_type& ty1) {
    return key_compare_(ty1, ty0);
  }

  key_compare key_compare_;
  container_type container_;
};


template<class K, bool Duplicate, class Pred, class A> inline
bool operator==(const sorted_vector<K, Duplicate, Pred, A>& x,
                const sorted_vector<K, Duplicate, Pred, A>& y) {
  return x.equal(y);
}
template<class K, bool Duplicate, class Pred, class A> inline
bool operator!=(const sorted_vector<K, Duplicate, Pred, A>& x,
                const sorted_vector<K, Duplicate, Pred, A>& y) {
  return !(x == y);
}
template<class K, bool Duplicate, class Pred, class A> inline
bool operator<(const sorted_vector<K, Duplicate, Pred, A>& x,
               const sorted_vector<K, Duplicate, Pred, A>& y) {
  return x.less_than(y);
}
template<class K, bool Duplicate, class Pred, class A> inline
bool operator>(const sorted_vector<K, Duplicate, Pred, A>& x,
               const sorted_vector<K, Duplicate, Pred, A>& y) {
  return y < x;
}
template<class K, bool Duplicate, class Pred, class A> inline
bool operator<=(const sorted_vector<K, Duplicate, Pred, A>& x,
                const sorted_vector<K, Duplicate, Pred, A>& y) {
  return !(y < x);
}
template<class K, bool Duplicate, class Pred, class A> inline
bool operator>=(const sorted_vector<K, Duplicate, Pred, A>& x,
                const sorted_vector<K, Duplicate, Pred, A>& y) {
  return (!(x < y));
}

#endif  // SRC_LDA_SORTED_VECTOR_H_
