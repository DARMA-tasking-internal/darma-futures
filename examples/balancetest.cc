#include "mpi_backend.h"
#include <sstream>
#include <sys/time.h>
#include <iomanip>

double get_time(){
  timeval t; gettimeofday(&t, nullptr);
  return t.tv_sec + 1e-6*t.tv_usec;
}

using Context=Frontend<MpiBackend>;

static const int primes[] = {
  1031, 857, 1811, 283, 941, 1019, 1153, 1583
};
static const int numPrimes = sizeof(primes) / sizeof(int);

struct BagOfTasks {

  struct Compute {
    void operator()(Context* ctx, int index, int rank, int localSleepMs){
      darmaDebug(Task, "Index {} sleeping for {}ms on MPI Rank={}",
            index, localSleepMs, rank);
      struct timespec sleepTS;
      sleepTS.tv_sec = 0;
      //convert ms to ns
      sleepTS.tv_nsec = localSleepMs * 1000000;
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

void usage(std::ostream& os){
  os << "Usage: ./run <niter> <od_factor> <time-multiplier> <seed> <lb_interval> <mpi_interop_interval>";
}

int run(int argc, char** argv)
{
  int rank; MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int size; MPI_Comm_size(MPI_COMM_WORLD, &size);

  auto dc = allocate_context(MPI_COMM_WORLD, argc, argv);
  int app_argc = dc->split_argv(argc,argv);

  if (app_argc != 7){
    if (rank == 0){
      std::cerr << "Invalid number of arguments: need 6\n";
      usage(std::cerr);
      std::cerr << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    return 0;
  }

  int niter = atoi(argv[1]);
  int od_factor = atoi(argv[2]);
  int time_multiplier = atoi(argv[3]);
  int random_seed = atoi(argv[4]);
  int lb_interval = atoi(argv[5]);
  int mpi_interval = atoi(argv[6]);

  int darma_size = size*od_factor;
  auto phase = dc->make_phase(darma_size);

  //this object IS valid to be accessed now
  auto mpi_coll = dc->make_local_collection<int>(darma_size);
  for (int i=0; i < od_factor; ++i){
    int workload = time_multiplier*(rank+1);
    if (random_seed != 0){ //scramble!
      int prime = primes[random_seed%numPrimes];
      workload = time_multiplier*(((rank+1)*prime)%size)*od_factor + i*time_multiplier;
    }
    mpi_coll->emplaceLocal(rank*od_factor + i, workload);
  }

  auto coll = dc->from_mpi<BagOfTasks::Migrate>(std::move(mpi_coll));

  if (dc->run_root()){
    for (int i=1; i <= niter; ++i){
      double t_start = get_time();
      std::tie(coll) = dc->create_phase_work<BagOfTasks::Compute>(phase,rank,std::move(coll));
      if (lb_interval && i % lb_interval == 0){
        if (rank == 0){
          std::cout << "Running load balancer on iteration... " << i
                    << std::endl;
        }
        dc->rebalance(phase);
        coll = dc->rebalance<BagOfTasks::Migrate>(phase,std::move(coll));
      }
      if (mpi_interval && i % mpi_interval == 0){
        if (rank == 0){
          std::cout << "MPI-interop on iteration... " << i
                    << std::endl;
        }
        mpi_coll = dc->to_mpi<BagOfTasks::Migrate>(std::move(coll));
        /**
        std::stringstream sstr;
        sstr << "MPI Check: Rank " << rank << " = {";
        for (auto& pair : mpi_coll->localElements()){
          BagOfTasks::Compute()(nullptr, pair.first, rank, *pair.second);
          sstr << " " << pair.first << ":" << *pair.second;
        }
        sstr << "}\n";
        std::cout << sstr.str();
        */
        coll = dc->from_mpi<BagOfTasks::Migrate>(std::move(mpi_coll));
      }
      double t_stop = get_time();
      if (rank == 0){
        double t_ms = (t_stop - t_start)*1e3;
        std::cout << "Iteration " << i << " took "
              << std::setprecision(3) << t_ms << "ms" << std::endl;
      }
    }
  } else {
    dc->run_worker();
  }
  return 0;
}

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);
  int rc = run(argc, argv);
  MPI_Finalize();
  return rc;
}
