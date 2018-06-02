#include "mpi_backend.h"

using Context=Frontend<MpiBackend>;

struct BagOfTasks {

  struct Compute {
    void operator()(Context* ctx, int index, int rank, async_ref_mm<int> localSleepMs){
      std::cout << "Index " << index << " sleeping for " << localSleepMs
                << " on MPI rank " << rank << std::endl;
      struct timespec sleepTS;
      sleepTS.tv_sec = 0;
      //convert ms to ns
      sleepTS.tv_nsec = *localSleepMs * 1000000;
      struct timespec remainTS;
      while (nanosleep(&sleepTS, &remainTS) == EINTR){
        sleepTS = remainTS;
      }
    }
  };

  struct Migrate {
    template <class Archive>
    static void pack(int& x, Archive& ar){
      ar | x;
    }

    template <class Archive>
    static void unpack(int& x, Archive& ar){
      ar | x;
    }

    template <class Archive>
    static void compute_size(int& x, Archive& ar){
      pack(x,ar);
    }
  };

};

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);
  int od_factor = 2;
  int rank; MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int size; MPI_Comm_size(MPI_COMM_WORLD, &size);
  int darma_size = size*od_factor;

  auto dc = allocate_context(MPI_COMM_WORLD);
  auto phase = dc->make_phase(darma_size);

  //this object IS valid to be accessed now
  auto mpi_coll = dc->make_local_collection<int>(darma_size);
  for (int i=0; i < od_factor; ++i){
    int initialValue = 5*(rank+1); //sleep rank*5 ms
    mpi_coll->emplaceLocal(rank*od_factor + i, initialValue);
  }

  auto coll = dc->from_mpi<BagOfTasks::Migrate>(std::move(mpi_coll));

  if (dc->run_root()){
    int niter = 2;
    for (int i=0; i < niter; ++i){
      std::tie(coll) = dc->create_phase_work<BagOfTasks::Compute>(phase,rank,std::move(coll));
      if (i % 1 == 0){
        dc->rebalance(phase);
        coll = dc->rebalance<BagOfTasks::Migrate>(phase,std::move(coll));
      }
    }
  } else {
    dc->run_worker();
  }
  MPI_Finalize();
}
