#ifndef mpi_async_ref_h
#define mpi_async_ref_h

#include <vector>
#include <iostream>
#include "mpi_phase.h"
#include "mpi_collection.h"

struct mpi_async_ref {
  mpi_async_ref(const mpi_async_ref&) = delete;

  mpi_async_ref(mpi_async_ref&&) = default;

  mpi_async_ref& operator=(mpi_async_ref&& t) = default;

  mpi_async_ref() : terminateID_(-1) {}

  const std::vector<int>& pendingRequests() const {
    return requests_;
  }

  void clearRequests() {
    requests_.clear();
  }

  void addRequest(int id) {
    requests_.push_back(id);
  }

  void setTerminateID(int termID) {
    terminateID_ = termID;
  }

 protected:
  enum empty_tag { Empty };

 private:
  std::vector<int> requests_;
  int terminateID_;
};

template <class T, class Imm, class Sched> class async_ref;

template <class T>
struct async_ref_base : public mpi_async_ref {
  async_ref_base(T* t) : t_(t), parent_(nullptr) {}

  async_ref_base(async_ref_base&& in) :
    t_(in.t_), parent_(in.parent_),
    mpi_async_ref(std::move(in))
  {
  }

  async_ref_base& operator=(async_ref_base&& t) = default;

  template <class Imm, class Sched>
  async_ref_base(async_ref<T,Imm,Sched>&& parent);

  static async_ref_base<T> make(){
    return async_ref_base<T>();
  }

  bool hasParent() const {
    return parent_;
  }

  template <class Idx>
  collection<T,Idx>* getParent() const {
    return static_cast<collection<T,Idx>*>(parent_);
  }

  void setParent(collection_base* parent) {
    parent_ = parent;
  }

  T* get() const {
    return t_;
  }

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

 protected:
  friend class MpiBackend;

  async_ref_base(empty_tag tag) : t_(nullptr), parent_(nullptr){}

  async_ref_base(async_ref_base* old) : //ptr to delete copy constructor
    parent_(old->parent_),
    t_(old->t_)
  {
  }

  template <class... Args>
  async_ref_base(Args&&... args) : parent_(nullptr) {
    //only backend can call this ctor
    t_ = new T(std::forward<Args>(args)...);
  }

 private:
  collection_base* parent_;
  T* t_;

};

#endif

