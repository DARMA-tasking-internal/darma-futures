#ifndef mpi_be_collection_h
#define mpi_be_collection_h

#include <map>
#include <vector>
#include <utility>
#include "mpi_phase.h"

template <class Idx>
struct Linearization {};

template <>
struct Linearization<int> {
  int toLinear(int rank, int size){
    return rank;
  }
  int fromLinear(int rank, int size){
    return rank;
  }
};

struct collection_base {
  void setId(int id){
    id_ = id;
  }
  
  int id() const {
    return id_;
  }

 private:
  int id_;
};

template <class T, class Idx> struct collection;

template <class T, class Idx>
struct mpi_collection {

  mpi_collection(int size) : referenced_(nullptr), size_(size) {}

  mpi_collection(mpi_collection&&) = default;

  bool referencesDarmaCollection() const {
    return referenced_;
  }

  collection<T,Idx>* darmaCollection() const {
    return referenced_;
  }

  const Idx& size() const {
    return size_;
  }

  const auto& localElements() const {
    return local_elements_;
  }

  T& getLocal(const Idx& idx){
    auto iter = local_elements_.find(idx);
    if (iter == local_elements_.end()){
      abort();
    }
    return *(iter->second);
  }

  template <class... Args>
  T& emplaceLocal(const Idx& idx, Args&&... args){
    T* elem = new T(std::forward<Args>(args)...);
    local_elements_[idx] = elem;
    return *elem;
  }

 private:
  std::map<int, T*> local_elements_;
  collection<T,Idx>* referenced_;
  int size_;

};

template <class T, class Idx>
using mpi_collection_ptr = std::unique_ptr<mpi_collection<T,Idx>>;

template <class T, class Idx>
struct collection : public collection_base {
  collection(int size) :
    size_(size),
    initialized_(false)
  {}

  collection(int size, const std::map<int,T*>& elements) :
    initialized_(true),
    size_(size),
    local_elements_(elements)
  {
  }

  T* getElement(int idx){
    auto iter = local_elements_.find(idx);
    return iter == local_elements_.end() ? nullptr : iter->second;
  }

  void setElement(int idx, T* t){
    local_elements_[idx] = t;
  }

  int size() const {
    return size_;
  }

  int getRank(int index){
    return index_mapping_[index].rank;
  }

  void assignMpi(mpi_collection_ptr<T,Idx>&& coll){
    mpi_parent_ = std::move(coll);
  }

  bool hasMpiParent() const {
    return bool(mpi_parent_);
  }

  auto&& moveMpiParent(){
    return std::move(mpi_parent_);
  }

  bool initialized() const {
    return initialized_;
  }

  void setInitialized() {
    initialized_ = true;
  }

  void initPhase(Phase<int>& ph){
    index_mapping_ = ph.mapping();
  }

  T* emplaceNew(const Idx& idx){
    T* t = new T;
    local_elements_[idx] = t;
    return t;
  }

  const IndexInfo& getIndexInfo(int index){
    return index_mapping_[index];
  }

  const std::map<int,T*> localElements() const {
    return local_elements_;
  }

  void remove(const Idx& idx){
    auto iter = local_elements_.find(idx);
    T* t = iter->second;
    delete t;
    local_elements_.erase(iter);
  }

  std::vector<IndexInfo> index_mapping_;
  std::map<int, T*> local_elements_;
  int size_;
  bool initialized_;
  std::unique_ptr<mpi_collection<T,Idx>> mpi_parent_;

};



#endif

