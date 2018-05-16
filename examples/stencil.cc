#include "mpi_backend.h"
#include <vector>

using Context=Frontend<MpiBackend>;

struct Patch {
 public:
  void timestep(){}

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
    auto operator()(Context* ctx, async_ref_mm<Patch> patch){
      patch->timestep();
      auto patch_rm = patch.read();
      for (auto& bnd : patch->boundaries()){
        patch_rm = ctx->send<GhostAccessor>(bnd,std::move(patch_rm));
        patch_rm = ctx->recv<GhostAccessor>(bnd,std::move(patch_rm));
      }
      
      async_ref_mm<Patch> ugh(patch);

      return std::make_tuple(patch_rm);
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
      std::tie(coll) = dc->create_phase_work<DarmaPatch::Timestep>(phase,coll);
      if (i % 5 == 0) dc->balance(phase);
    }
  } else {
    dc->run_worker();
  }

  MPI_Finalize();
}


