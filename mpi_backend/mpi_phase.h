#ifndef mpi_be_phase_h
#define mpi_be_phase_h

#include <memory>
#include <cstdint>

struct PerformanceCounter {
  std::vector<uint64_t> counters;
};

struct LocalIndex {
  PerformanceCounter counters;
  int index;

  LocalIndex(int i) : index(i){} 
};


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

 private:
  int size_;
  std::vector<int> index_to_rank_mapping_;
  std::vector<LocalIndex> local_;
};

template <class Idx>
struct Phase {

 Phase(){}

 Phase(Idx& size) : 
  data_(std::make_shared<PhaseData>(size))
 {}

 bool active() const {
  return bool(data_);
 }

 PhaseData* operator->() const {
  return data_.get();
 }

 std::shared_ptr<PhaseData> data_;
};

#endif

