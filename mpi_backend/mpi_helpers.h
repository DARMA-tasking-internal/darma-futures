#ifndef DARMA_BACKEND_MPI_HELPERS_H
#define DARMA_BACKEND_MPI_HELPERS_H

#include <async_ref.h>
#include <mpi_collection.h>
#include <utility>
#include <darma/serialization/simple_handler.h>
#include <darma/serialization/serializers/all.h>
#include <mpi.h>

namespace darma_backend
{
  using serializer = darma::serialization::SimpleSerializationHandler<>;
  using serialization_buffer = darma::serialization::DynamicSerializationBuffer<std::allocator<char>>;
  
  template< typename T, typename ScheduledPermissions >
  serializer::serialization_buffer_t
  get_sendbuff( const async_ref< T, ReadOnly, ScheduledPermissions > &_ref )
  {
    return serializer::serialize( *_ref );
  }
  
  template<typename T>
  struct type_to_mpi;

  template<>
  struct type_to_mpi<int> {
    static constexpr auto datatype = MPI_INT;
  };

  template<>
  struct type_to_mpi<long> {
    static constexpr auto datatype = MPI_LONG;
  };

  template<>
  struct type_to_mpi<long long> {
    static constexpr auto datatype = MPI_LONG_LONG;
  };
  template<>
  struct type_to_mpi<unsigned> {
    static constexpr auto datatype = MPI_UNSIGNED;
  };

  template<>
  struct type_to_mpi<unsigned long> {
    static constexpr auto datatype = MPI_UNSIGNED_LONG;
  };

  template<>
  struct type_to_mpi<unsigned long long> {
    static constexpr auto datatype = MPI_UNSIGNED_LONG_LONG;
  };
  
  template< typename T >
  constexpr auto get_mpi_type( T )
  {
    return type_to_mpi<T>::datatype;
  }
}

#endif  // DARMA_BACKEND_MPI_HELPERS_H
