#ifndef mpi_be_phase_h
#define mpi_be_phase_h

template <class Idx>
struct Phase {
 friend struct MpiBackend;

 Phase(Idx& size) : size_(size){}

 const Idx& getSize() const {
  return size_;
 }

 const std::vector<Idx> indices() const {
  return local_;
 }

 private:
  Idx size_;
  std::vector<int> index_to_rank_mapping_;
  std::vector<Idx> local_;
};

#endif

