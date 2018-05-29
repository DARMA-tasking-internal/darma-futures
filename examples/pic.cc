#include "mpi_backend.h"
#include <vector>

using Context=Frontend<MpiBackend>;

//struct Particle {};
using Particle=int;


struct Swarm {
  friend struct DarmaSwarm;
 public:
  void move(bool local, std::vector<Particle>& parts){}

  void moveLocal(){
    move(true, parts_);
  }

  void solveFields(){}

  const std::vector<int>& boundaries() const {
    return boundaries_;
  }

 private:
  std::vector<Particle> parts_;
  std::vector<std::vector<Particle>> migrants_;
  std::vector<int> boundaries_;
};

struct DarmaSwarm {
 struct Migrate {
   template <class Archive>
   static void pack(Swarm& p, Archive& ar, int nbr){
     ar | p.migrants_[nbr];
   }

   template <class Archive>
   static void compute_size(Swarm& p, Archive& ar, int nbr){
     pack(p,ar,nbr);
   }

   static auto sendMigrants(Context* ctx, async_ref_ii<Swarm> swarm){
     auto& bounds = swarm->boundaries();
     for (int b=0; b < bounds.size(); ++b){
       swarm = ctx->put_task<DarmaSwarm::Migrate>(bounds[b],std::move(swarm),b);
     }
     return std::make_tuple(std::move(swarm));
   }

   template <class Archive>
   static void unpack(Context* ctx, async_ref_ii<Swarm> swarm, Archive& ar){
     std::vector<Particle> incoming;
     ar | incoming;
     swarm->move(false, incoming);
     sendMigrants(ctx, std::move(swarm));
   }

 };

 struct MpiOut { };

 struct MpiIn { };

 struct Move {
  auto operator()(Context* ctx, int index, async_ref_ii<Swarm> swarm){
    swarm->moveLocal();
    return Migrate::sendMigrants(ctx, std::move(swarm));
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
    std::tie(part_coll) = dc->from_mpi<DarmaSwarm::MpiIn>(std::move(mpi_swarm));
    std::tie(part_coll) = dc->create_phase_window<DarmaSwarm::Move>(ph, std::move(part_coll));
    std::tie(mpi_swarm) = dc->to_mpi<DarmaSwarm::MpiOut>(std::move(part_coll));
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

//register the task with the runtime system
static auto move_task = recv_task_id<DarmaSwarm::Migrate,Swarm,int>();

