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
  std::tie(start, end) = detail::range_for_rank(rank, nranks, 0, static_cast<int>(element_vector.size()));
  
  for (int i = start; i < end; ++i)
    coll.setElement(i, &element_vector[i]);
  
  auto ref = async_ref<collection<int, int>, ReadOnly, None>::make(coll);
  
  auto retref = darma_backend::gather(std::move(ref), 0);
  
  const auto &v = *retref;
  
  if (rank == 0)
    EXPECT_TRUE(std::equal(element_vector.begin(), element_vector.end(), v.begin(), v.end()));
}

const
std::vector<int> g_element_vector = {5, 9, 3, 7, 11, 2, 1, 20};

struct init_test_work
{
  void operator()(Frontend<MpiBackend> *ctx, int index,
                  async_ref< int, Modify, Modify > ref)
  {
    *ref = g_element_vector[index];
  }
};

struct check_init
{
  void operator()(Frontend<MpiBackend> *ctx, int index,
                  async_ref<int, ReadOnly, Modify> ref)
  {
    EXPECT_EQ(*ref, g_element_vector[index]);
  }
};

struct check_vecs
{
  void operator()(Frontend<MpiBackend> *ctx,
                  async_ref< std::vector<int>, Modify, Modify > ref)
  {
    const auto &vec = *ref;
    EXPECT_TRUE(std::equal(g_element_vector.begin(), g_element_vector.end(),
                           vec.begin(), vec.end()));
  }
};

TEST(mpi_gather_test, GatherFrontend)
{
  auto dc = allocate_context(MPI_COMM_WORLD);
  
  auto c = dc->make_collection<int>(static_cast<int>(g_element_vector.size()));
  auto phase = dc->make_phase(static_cast<int>(g_element_vector.size()));
  
  std::tie(c) = dc->create_phase_work<init_test_work>(phase, std::move(c));
  
  std::tie(c) = dc->create_phase_work<check_init>(phase, std::move(c));
  
  auto vec = dc->phase_gather(phase, 0, std::move(c));
  
  if ( dc->is_root() ) {
    dc->create_work<check_vecs>(std::move(std::get<0>(vec)));
  }
  
  dc->flush();
}


const std::vector<std::vector<int>> g_elements = {{5,3}, {21, 4, 9}, {3}, {7, 10, 54},
                                                {11, 22, 33}, {2, 4, 8, 16}, {1, 1, 2, 3, 5},
                                                {20, 10, 5}};

struct init_elements
{
  void operator()(Frontend<MpiBackend> *ctx, int index,
                  async_ref< std::vector<int>, Modify, Modify > ref)
  {
    *ref = g_elements[index];
  }
};

struct check_elements
{
  void operator()(Frontend<MpiBackend> *ctx,
                  async_ref<std::vector<std::vector<int>>, Modify, Modify > ref)
  {
    const auto &vec = *ref;
    EXPECT_TRUE(std::equal(g_elements.begin(), g_elements.end(),
                           vec.begin(), vec.end()));
  }
};

TEST(mpi_gather_test, GatherFrontendComplexSerialization)
{
  auto dc = allocate_context(MPI_COMM_WORLD);
  
  auto c = dc->make_collection<std::vector<int>>(static_cast<int>(g_elements.size()));
  auto phase = dc->make_phase(static_cast<int>(g_elements.size()));
  
  std::tie(c) = dc->create_phase_work<init_elements>(phase, std::move(c));
  auto vec = dc->phase_gather(phase, 0, std::move(c));
  
  if ( dc->is_root() ) {
    dc->create_work<check_elements>(std::move(std::get<0>(vec)));
  }
  
  dc->flush();
  
}
