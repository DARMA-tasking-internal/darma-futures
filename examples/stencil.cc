#include "mpi_backend.h"
#include <vector>
#include <iostream>

using Context=Frontend<MpiBackend>;

struct Patch {
 public:
  void timestep(int index, int iter){
    std::cout << "Running timestep "
              << iter << " on index " << index
              << std::endl;
  }

  const std::vector<int>& boundaries() const {
    return boundaries_;
  }
  
 private:
  std::vector<int> boundaries_;

};


struct DarmaPatch {
  struct GhostAccessor {
    template <class Archive>
    static void pack(Patch& p, int index, Archive& ar){
      ar | index;
      //pack a vector or something
    }

    template <class Archive>
    static void unpack(Patch& p, Archive& ar){
      int neighbor;
      std::vector<double> values;
      ar | neighbor;
      //ar | values;
      //loop incoming values from that neighbor and put them in correct location
    }

    template <class Archive>
    static void compute_size(Patch& p, int index, Archive& ar){
      pack(p,index,ar);
    }
  };

  struct Timestep {
    auto operator()(Context* ctx, int index, int iter, async_ref_mm<Patch> patch){
      patch->timestep(index, iter);
      auto patch_rm = ctx->read(std::move(patch));
      for (int bnd : patch_rm->boundaries()){
        patch_rm = ctx->send<GhostAccessor>(index,bnd,std::move(patch_rm));
        patch_rm = ctx->recv<GhostAccessor>(index,bnd,std::move(patch_rm));
      }
      return std::make_tuple(std::move(patch_rm));
    }
  };

};

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);
  Patch patch;
  int od_factor = 4;
  int rank; MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int size; MPI_Comm_size(MPI_COMM_WORLD, &size);
  int darma_size = size*od_factor;

  auto dc = allocate_context(MPI_COMM_WORLD);
  auto coll = dc->make_collection<Patch>(darma_size);
  auto phase = dc->make_phase(darma_size);

  if (dc->run_root()){
    int niter = 10;
    for (int i=0; i < niter; ++i){
      std::tie(coll) = dc->create_phase_work<DarmaPatch::Timestep>(phase,i,std::move(coll));
      if (i % 5 == 0) dc->balance(phase);
    }
  } else {
    dc->run_worker();
  }

  MPI_Finalize();
}


