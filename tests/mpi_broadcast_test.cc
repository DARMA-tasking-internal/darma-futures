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
  
  auto ref = async_ref<int, ReadOnly, None>::empty();
  if (rank == 0) { 
    ref = async_ref<int, ReadOnly, None>::make(value);
  }
  
  auto ret = darma_backend::broadcast< int >(std::move(ref), 0);
  
  EXPECT_EQ(*ret->getElement(rank), value);
}

constexpr int g_testval = 17;

struct init_broadcast_val
{
  void operator()(Frontend<MpiBackend> *ctx, async_ref<int, Modify, Modify> ref, int val)
  {
    *ref = val;
  }
};

struct test_broadcasted
{
  void operator()(Frontend<MpiBackend> *ctx, int index,
                  async_ref< int, Modify, Modify > ref)
  {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    EXPECT_EQ(*ref, g_testval);
  }
};

TEST(mpi_broadcast_test, BroadcastFrontend) { // NOLINT
  int nranks;
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);
  
  auto dc = allocate_context(MPI_COMM_WORLD, 0, nullptr);
  
  auto val = dc->make_async_ref< int >();
  auto phase = dc->make_phase(nranks);
  
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
  auto c = dc->make_collection<int>(nranks);
  
  auto newval = dc->create_work<init_broadcast_val>(std::move(val), g_testval);
  
  std::tie(c) = dc->phase_broadcast< int >(phase, 0, std::move(std::get<0>(newval)));
  
  dc->create_phase_work< test_broadcasted >(phase, std::move(c));
  
  dc->flush();
}
