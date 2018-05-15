#include "mpi_backend.h"
#include <vector>

using Context=Frontend<MpiBackend>;

struct Swarm {
 public:
  void move(){}
  void solveFields(){}

  const std::vector<int>& boundaries() const {
    return boundaries_;
  }

 private:
  std::vector<int> boundaries_;
};

struct DarmaSwarm {
 struct Migrate { };

 struct MpiOut { };

 struct MpiIn { };

 struct Move {
  template <class Accessor>
  auto operator()(Context* ctx, async_ref_ii<Swarm> swarm){
    swarm->move();
    for (auto& bnd : swarm->boundaries()){
      swarm = ctx->put_task<DarmaSwarm::Migrate,Move>(bnd,std::move(swarm));
    }
    return std::make_tuple(swarm);
  }
 };
};

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);
  Swarm mainPatch;
  int od_factor = 4;
  int rank; MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int size; MPI_Comm_size(MPI_COMM_WORLD, &size);
  int darma_size = size*od_factor;

  auto dc = allocate_context(MPI_COMM_WORLD);
  auto ph = dc->make_phase(darma_size);
  auto mpi_swarm = dc->make_local_collection<Swarm>(ph);

  int niter = 10;
  for (int i=0; i < niter; ++i){
    //overdecompose
    //for (auto& pair : coll){
    // int idx = pair.first;
    // Swarm& patch = pair.second;
    // overdecompose(rank,mainPatch,idx,patch);
    //}
    auto part_coll = dc->darma_collection(mpi_swarm);
    std::tie(part_coll) = dc->to_mpi<DarmaSwarm::MpiIn>(std::move(mpi_swarm));
    std::tie(part_coll) = dc->create_phase_idempotent_work<DarmaSwarm::Move>(ph,std::move(part_coll));
    std::tie(mpi_swarm) = dc->from_mpi<DarmaSwarm::MpiOut>(std::move(part_coll));
    //un-overdecompose
    //for (auto& pair : coll){
    // int idx = pair.first;
    // Swarm& patch = pair.second;
    // unOverdecompose(rank,mainPatch,idx,patch);
    //}
    mainPatch.solveFields();
  }
  MPI_Finalize();
}



