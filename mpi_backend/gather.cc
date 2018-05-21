#include <mpi.h>
#include "mpi_helpers.h"
#include <vector>
#include <numeric>
#include <iostream>

namespace darma_backend {
  namespace detail {
    serialization_buffer
    gather_internal(const serialization_buffer &buff, int root, MPI_Comm comm) {
      // Get size of all data being transferred, in bytes
      auto local_size = static_cast< int >( buff.capacity());
    
      int size;
      MPI_Comm_size(comm, &size);
    
      int rank;
      MPI_Comm_rank(comm, &rank);
    
      // Size and offsets used by root and ignored by everything else
      std::vector<int> sizes;
      std::vector<int> offsets;
      if (rank == root) {
        sizes.resize(static_cast< std::size_t >(size));
        offsets.resize(static_cast< std::size_t >(size), 0ULL);
      }
    
      // Transfer all the sizes to the root
      MPI_Gather(&local_size, 1, get_mpi_type(local_size), &sizes[0],
                 1, get_mpi_type(local_size), root, comm);
  
      int total_size = 0;
      if ( rank == root ) {
        for (std::size_t i = 0; i < offsets.size(); ++i) {
          // sizes and offsets have the same lengths
          offsets[i] = total_size;
          total_size += sizes[i];
        }
      }
  
      serialization_buffer recvbuff(static_cast< std::size_t >(total_size));
      MPI_Gatherv(buff.data(), static_cast< int >( buff.capacity()), MPI_BYTE,
                  recvbuff.data(),
                  sizes.data(), offsets.data(), MPI_BYTE, root, comm);
    
      return recvbuff;
    }
  }
}
