#include <mpi.h>
#include "mpi_helpers.h"

namespace darma_backend {
  namespace detail {
    void 
    broadcast_internal(serialization_buffer &buff, int root, MPI_Comm comm) {
      // Get size of data being transferred, only matters for root
      auto local_size = static_cast< int >( buff.capacity() );
      
      int rank;
      MPI_Comm_rank(comm, &rank);
      
      MPI_Bcast(buff.data(), local_size, MPI_BYTE, root, comm);
    }
  }
}