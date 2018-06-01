#ifndef mpi_be_phase_h
#define mpi_be_phase_h

#include <memory>
#include <cstdint>
#include "mpi_index_entry.h"

struct PerformanceCounter {
  uint64_t counter; //just one for now
  PerformanceCounter() : counter(0){}
};

struct LocalIndex {
  PerformanceCounter counters;
  int index;

  LocalIndex(int i) : index(i){} 
};

namespace detail
{
  /**
   * Returns the beginning and end of an element-to-rank assignment in an array.
   * 
   * @tparam IndexType  The index type (deduced)
   * @param rank        The rank to get the index range for
   * @param nranks      The total number of ranks
   * @param begin       The beginning index of the array (e.g. 0)
   * @param end         The ending index of the array (e.g. the array size)
   * @return            A pair where first is the beginning of the range for rank and second is
   *                    the end of the range.
   */
  template<typename IndexType>
  std::pair<IndexType, IndexType>
  range_for_rank(int rank, int nranks, IndexType begin, IndexType end)
  {
    auto size = end - begin;
    
    auto count = size / static_cast< IndexType >(nranks);
    auto rem = size % static_cast< IndexType >(nranks);
    
    std::pair<IndexType, IndexType> ret;
    if (rank < rem) {
      ret.first = begin + rank * (count + 1);
      ret.second = ret.first + count + 1;
    } else {
      ret.first = begin + rem + rank * count;
      ret.second = ret.first + count;
    }
    
    return ret;
  };
}

struct PhaseData {
 friend struct MpiBackend;

 PhaseData(int size) : size_(size){}

 int getSize() const {
  return size_;
 }

 auto index_begin() {
  return local_.begin();
 }

 auto index_end() {
  return local_.end();
 }

 int getRank(int idx) const {
  return index_to_rank_mapping_[idx].rank;
 }

 const std::vector<LocalIndex>& local() const {
   return local_;
 }

 const std::vector<IndexInfo>& mapping() const {
   return index_to_rank_mapping_;
 }

 private:
  int size_;
  std::vector<IndexInfo> index_to_rank_mapping_;
  std::vector<LocalIndex> local_;
};

template <class Idx>
struct Phase {

 Phase(){}

 Phase(const Idx& size) :
  data_(std::make_shared<PhaseData>(size))
 {}

 bool active() const {
  return bool(data_);
 }

 const std::vector<IndexInfo>& mapping() const {
   return data_->mapping();
 }

 PhaseData* operator->() const {
  return data_.get();
 }

 std::shared_ptr<PhaseData> data_;
};

template <class Idx>
struct Window {
  Window(){}

  void setId(int id){
    id_ = id;
  }

  int id() const {
    return id_;
  }

 private:
  int id_;
};


#endif

