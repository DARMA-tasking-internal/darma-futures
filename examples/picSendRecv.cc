#include "mpi_backend.h"

using Context=Frontend<MpiBackend>;

struct Swarm {
 public:
  /**
    @return The number moved outside patch
  */
  int move(){
    return 0;
  }

  void solveFields(){}

  const std::vector<int>& boundaries() const {
    return boundaries_;
  }

 private:
  std::vector<int> boundaries_;
};

struct DarmaSwarm {
 struct MigrateAccessor {
    template <class Archive>
    static void pack(Swarm& p, int index, Archive& ar){
      ar | index;
      //pack a vector or something
    }

    template <class Archive>
    static void unpack(Swarm& p, Archive& ar){
      int neighbor;
      std::vector<double> values;
      ar | neighbor;
      //ar | values;
      //loop incoming values from that neighbor and put them in correct location
    }

    template <class Archive>
    static void compute_size(Swarm& p, int index, Archive& ar){
      pack(p,index,ar);
    }
 };

 struct MpiIn {};
 struct MpiOut {};

 struct Move {
  auto operator()(Context* ctx, int index, async_ref_mm<Swarm> swarm, async_ref_mm<int> nmoved){
    *nmoved = swarm->move();
    auto swarm_nm = ctx->modify(std::move(swarm));
    for (auto& bnd : swarm->boundaries()){
      swarm_nm = ctx->send<MigrateAccessor>(index,bnd,std::move(swarm_nm));
      swarm_nm = ctx->recv<MigrateAccessor>(index,bnd,std::move(swarm_nm));
    }
    return std::make_tuple(std::move(swarm_nm),std::move(nmoved));
  }
 };

};

struct GreaterThanZero {
  bool operator()(Context* ctx, async_ref_rr<int> value){
    return (*value) > 0;
  }
};

template <class T>
struct Add {
  static T identity(){
    return 0;
  }

  static T* mpiBuffer(T& t){
    return &t;
  }

  static int mpiSize(T& t){
    return 1;
  }

  static MPI_Datatype mpiType(int& i){ return MPI_INT; }

  static MPI_Op mpiOp(int& i){ return MPI_SUM; }

  void operator()(const T& in, T& out){
    out += in;
  }

};


struct CollectiveMove {
  auto operator()(Context* ctx, Phase<int> ph, 
                  async_collection<Swarm,int> swarm, 
                  async_collection<int,int> nmoved){

    auto swarm_ret = ctx->modify(std::move(swarm));
    auto nmoved_ret = ctx->modify(std::move(nmoved));
    //std::tie(swarm_ret, nmoved_ret) = ctx->create_phase_work<DarmaSwarm::Move>(
    //          ph,std::move(swarm),std::move(nmoved));

    auto total_ret = ctx->make_async_ref<int>();
    std::tie(total_ret, nmoved_ret) = ctx->phase_reduce<Add<int>>(ph, std::move(nmoved_ret));

    //this will be so much cleaner in c++17
    auto tupleRet = ctx->make_predicate<GreaterThanZero>(std::move(total_ret));
    auto terminate = std::move(std::get<0>(tupleRet));
    std::tie(total_ret) = std::move(std::get<1>(tupleRet));

    //std::tie(swarm_ret,nmoved_ret) = 
    //  ctx->create_work_if<CollectiveMove>(std::move(terminate), ph,
    //     std::move(swarm_ret), std::move(nmoved_ret));
    return std::make_tuple(std::move(swarm_ret), std::move(nmoved_ret));
  }
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
  auto phase = dc->make_phase(darma_size);

  auto nmoved_coll = dc->make_collection<int>(darma_size);

  //this object IS valid to be accessed now
  auto mpi_swarm = dc->make_local_collection<Swarm>(phase);
  //need an mpi init function here
  
  int niter = 10;
  for (int i=0; i < niter; ++i){
    //overdecompose
    //for (auto& pair : mpi_swarm){
    //  int idx = pair.first;
    //  Swarm& patch = pair.second;
    //  //overdecompose(rank,mainPatch,idx,patch);
    //}

    auto part_coll = dc->darma_collection(mpi_swarm);
    std::tie(part_coll) = dc->to_mpi<DarmaSwarm::MpiIn>(std::move(mpi_swarm));
    std::tie(part_coll,nmoved_coll) = 
      dc->create_work<CollectiveMove>(phase,std::move(part_coll),std::move(nmoved_coll));
    std::tie(mpi_swarm) = dc->from_mpi<DarmaSwarm::MpiOut>(std::move(part_coll));

    //un-overdecompose
    //for (auto& pair : mpi_swarm){
    // int idx = pair.first;
    // Swarm& patch = pair.second;
    //// unOverdecompose(rank,mainPatch,idx,patch);
    //}
    mainPatch.solveFields();
  }
  MPI_Finalize();
}


