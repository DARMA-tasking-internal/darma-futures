#include <gtest/gtest.h>
#include <mpi_backend.h>
#include <broadcast.h>
#include <mpi_helpers.h>

TEST(mpi_broadcast_test, BroadcastInternal) { // NOLINT
  const std::vector<int> element_vector = {5, 9, 3, 7, 11, 2, 1, 20};
  
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int nranks;
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);
  
  if ( rank == 0 ) {
    auto buff = darma_backend::serializer::serialize(element_vector);
    
    darma_backend::detail::broadcast_internal(buff, 0);
  
    auto ar = darma_backend::serializer::make_unpacking_archive(buff);
    auto retvec = ar.template unpack_next_item_as<std::vector<int>>();
    EXPECT_TRUE(std::equal(element_vector.begin(), element_vector.end(),
                           retvec.begin(), retvec.end()));
    
  } else {
    auto buff = darma_backend::serialization_buffer{sizeof(std::size_t) + element_vector.size() * sizeof(int)};
    darma_backend::detail::broadcast_internal(buff, 0);
  
    auto ar = darma_backend::serializer::make_unpacking_archive(buff);
    auto retvec = ar.template unpack_next_item_as<std::vector<int>>();
    EXPECT_TRUE(std::equal(element_vector.begin(), element_vector.end(), 
                           retvec.begin(), retvec.end()));
  }
}

TEST(mpi_broadcast_test, BroadcastAsyncRef) { // NOLINT
  constexpr int value = 17;
  
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
  async_ref<int, None, Modify> ref;
  if (rank == 0) { 
    ref = async_ref<int, None, Modify>{value};
  }
  
  auto ret = darma_backend::broadcast< int >(std::move(ref), 0);
  
  EXPECT_EQ(*ret->getElement(rank), value);
  
  delete ret->getElement(rank);
}