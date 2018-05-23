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
  
  template<typename T>
  struct type_to_mpi;

  template<>
  struct type_to_mpi<int> {
    static auto datatype()
    {
      return MPI_INT;
    }
  };

  template<>
  struct type_to_mpi<long> {
    static auto datatype()
    {
      return MPI_LONG;
    }
  };

  template<>
  struct type_to_mpi<long long> {
    static auto datatype()
    {
      return MPI_LONG_LONG;
    }
  };
  template<>
  struct type_to_mpi<unsigned> {
    static auto datatype()
    {
      return MPI_UNSIGNED;
    }
  };

  template<>
  struct type_to_mpi<unsigned long> {
    static auto datatype()
    {
      return MPI_UNSIGNED_LONG;
    }
  };

  template<>
  struct type_to_mpi<unsigned long long> {
    static auto datatype()
    {
      return MPI_UNSIGNED_LONG_LONG;
    }
  };
  
  // TODO: rest of conversions
  
  template< typename T >
  constexpr auto get_mpi_type( T )
  {
    return type_to_mpi<T>::datatype();
  }
}

#endif  // DARMA_BACKEND_MPI_HELPERS_H
