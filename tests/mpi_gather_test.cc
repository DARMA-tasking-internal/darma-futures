#include <gtest/gtest.h>
#include <mpi_backend.h>
#include <gather.h>
#include <mpi_collection.h>

TEST(mpi_gather_test, GatherInternal) { // NOLINT
  const std::vector<int> element_vector = {5, 9, 3, 7, 11, 2, 1, 20};
  
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  
  auto buff = darma_backend::serializer::serialize(element_vector[rank]);
  
  auto retbuff = darma_backend::detail::gather_internal(buff, 0);
  
  
  auto *intbuff = reinterpret_cast<int *>(retbuff.data());
  
  if (rank == 0) {
    for (int i = 0; i < size; ++i) {
      EXPECT_EQ(element_vector[i], intbuff[i]);
    }
  }
}

TEST(mpi_gather_test, GatherAsyncRef) { // NOLINT
  // TODO: make sure collection can take const ptrs so this can be const
  std::vector<int> element_vector = {5, 9, 3, 7, 11, 2, 1, 20};
  
  // Build a collection manually for now
  collection<int, int> coll(static_cast< int >(element_vector.size()));
  
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
  int nranks;
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);
  
  int start, end;
  std::tie(start, end) = range_for_rank(rank, nranks, 0, static_cast<int>(element_vector.size()));
  
  for (int i = start; i < end; ++i)
    coll.setElement(i, &element_vector[i]);
  
  auto ref = async_ref<collection<int, int>, ReadOnly, None>::make(coll);
  
  auto retref = darma_backend::gather(std::move(ref), 0);
  
  const auto &v = *retref;
  
  if (rank == 0)
    EXPECT_TRUE(std::equal(element_vector.begin(), element_vector.end(), v.begin(), v.end()));
}
