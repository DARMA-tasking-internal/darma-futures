#ifndef mpi_be_collection_h
#define mpi_be_collection_h

#include <map>
#include <vector>

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

template <class T, class Idx>
struct collection : public collection_base {
  collection(int size) : size_(size) {}

  T* getElement(int idx){
    return local_elements_[idx];
  }

  void setElement(int idx, T* t){
    local_elements_[idx] = t;
  }

  int getRank(int index){
    return index_mapping_[index].rank;
  }

  struct EntryInfo {
    int rank;
    int rankUniqueId;
  };

  const EntryInfo& getEntryInfo(int index){
    return index_mapping_[index];
  }

  std::vector<EntryInfo> index_mapping_;
  std::map<int, T*> local_elements_;
  int size_;

};


template <class T, class Idx>
struct mpi_collection {

  mpi_collection(int size) : coll_(size) {}

  mpi_collection(collection<T,Idx>&& coll) : coll_(std::move(coll)) {}

  collection<T,Idx>& getCollection(){
    return coll_;
  }

 private:
  collection<T,Idx> coll_;

};
#endif

