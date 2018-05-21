#ifndef DARMA_BACKEND_BROADCAST_H
#define DARMA_BACKEND_BROADCAST_H

#include <async_ref.h>
#include "mpi_helpers.h"
#include <mpi.h>

namespace darma_backend {
  namespace detail {
    /**
     * Broadcast contents of buff from root to other processes, writing to buff.
     * After this operation, buff will contain a copy of the data in buff from root.
     * 
     * \pre The size of buff must be the same on all processes
     * 
     * @param buff The send/recv buffer
     * @param root The rank to broadcast from
     * @param comm The MPI communicator
     */
    void broadcast_internal(serialization_buffer &buff, int root, MPI_Comm comm = MPI_COMM_WORLD);
  }
  
  /**
   * Perform a broadcast operation. Data broadcast by the root rank will be vailable on
   * all ranks in the communicator. This yields a collection which represents data
   * that may not be on-rank.
   * @tparam    IndexType   The index type for the resulting collection
   * @tparam    T           The type of the data to broadcast
   * @param     ref         The data to broadcast
   * @param     root        The rank of the broadcasting process
   * @param     comm        The MPI communicator to use for the broadcast
   * @return                The collection containing the broadcasted data
   */
  template<typename IndexType, typename T>
  async_collection<T, IndexType>
  broadcast(async_ref<T, None, Modify> &&ref, int root, MPI_Comm comm = MPI_COMM_WORLD) {
    int nranks;
    MPI_Comm_size(comm, &nranks);
    
    int rank;
    MPI_Comm_rank(comm, &rank);
  
    collection<T, IndexType> ret(static_cast<IndexType>(nranks));
    
    if (rank == root) {
      auto buff = serializer::serialize(*ref);
      auto datasizebuff = serializer::serialize(buff.capacity());
      
      // Broadcast size of data
      detail::broadcast_internal(datasizebuff, root, comm);
      
      // Broadcast actual data buffer
      detail::broadcast_internal(buff, root, comm);
      
      // TODO: figure out how collection should store things to avoid a memory leak here
      ret.setElement(rank, new T (std::move(*ref)));
    } else {
      auto datasizebuff = serialization_buffer(sizeof(std::size_t));
  
      detail::broadcast_internal(datasizebuff, root, comm);
      auto data_size = serializer::deserialize<std::size_t>(datasizebuff);
      
      auto buff = serialization_buffer(data_size);
      
      // Get data buffer
      detail::broadcast_internal(buff, root, comm);
  
      // TODO: figure out how collection should store things to avoid a memory leak here
      ret.setElement(rank, new T (serializer::deserialize<T>(buff)));
    }
  
    return async_collection<T, IndexType>(std::move(ret));
  };
}

#endif  // DARMA_BACKEND_BROADCAST_H
