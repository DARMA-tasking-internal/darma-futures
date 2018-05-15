#ifndef mpi_async_ref_h
#define mpi_async_ref_h

#include <vector>

struct mpi_async_ref {

  const std::vector<int>& pendingRequests() const {
    return requests_;
  }

 private:
  std::vector<int> requests_;
};

template <class T>
struct async_ref_base {
  template <class... Args>
  async_ref_base(Args&&... args) {
    t_ = new T(std::forward<Args>(args)...);
  }

  async_ref_base(T* t) : t_(t) {}

  async_ref_base(const async_ref_base&) = delete;

  async_ref_base(async_ref_base&&) = default;
  async_ref_base& operator=(async_ref_base&& t) = default;

  async_ref_base() : t_(nullptr) {}

  T& operator*(){
    return *t_;
  }

  T* operator->(){
    return t_;
  }

  operator T&() {
    return *t_;
  }

  operator const T&() const {
    return *t_;
  }

 private:
  T* t_;

};

#endif

