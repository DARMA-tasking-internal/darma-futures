#ifndef DARMA_BACKEND_GATHER_H
#define DARMA_BACKEND_GATHER_H

#include <async_ref.h>
#include "mpi_helpers.h"
#include <mpi.h>
#include <vector>

namespace darma_backend {
  namespace detail {
    serialization_buffer
    gather_internal(const serialization_buffer &buff,
                    int root, MPI_Comm comm = MPI_COMM_WORLD);
  }
  
  template<typename T, typename IndexType>
  async_ref<std::vector<T>, None, Modify>
  gather(async_collection<T, IndexType> &&_collection, int root,
         MPI_Comm comm = MPI_COMM_WORLD) {
    auto &outcoll = *_collection;
    
    auto sar = serializer::make_packing_archive(sizeof(std::size_t) + outcoll.local_elements_.size() * sizeof(T));
    sar << outcoll.local_elements_.size();
    for (auto &&kv : outcoll.local_elements_)
      sar << *kv.second;
    
    auto sendbuff = serializer::extract_buffer(std::move(sar));
    
    auto retbuff = detail::gather_internal(sendbuff, root, comm);
    
    int nranks;
    MPI_Comm_size(comm, &nranks);
  
    int rank;
    MPI_Comm_rank(comm, &rank);
  
    std::vector<T> retvec;
    if ( rank == root ) {
      auto ar = serializer::make_unpacking_archive(retbuff);
  
      for (int i = 0; i < nranks; ++i) {
        auto sz = ar.template unpack_next_item_as<std::size_t>();
    
        retvec.reserve(retvec.size() + sz);
    
        for (std::size_t j = 0; j < sz; ++j)
          retvec.emplace_back(ar.template unpack_next_item_as<T>());
      }
    }
    
    return async_ref<std::vector<T>, None, Modify>(std::move(retvec));
  }
}

#endif  // DARMA_BACKEND_GATHER_H
