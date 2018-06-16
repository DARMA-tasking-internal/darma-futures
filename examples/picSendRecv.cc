#include "mpi_backend.h"

using Context=Frontend<MpiBackend>;

struct Swarm {
 public:
  Swarm() : remaining_(0){}

  void init(int x, int y, int nx, int ny){
    myX_ = x;
    myY_ = y;
    myIndex_ = linearize(x,y);
    sizeX_ = nx;
    sizeY_ = ny;

    boundaries_.resize(4);
    boundaries_[0] = linearize(nbrX(1),myY_);
    boundaries_[1] = linearize(nbrX(-1),myY_);
    boundaries_[2] = linearize(myX_,nbrY(1));
    boundaries_[3] = linearize(myX_,nbrY(-1));
  }

  int nbrX(int disp){
    return (myX_ + disp - sizeX_) % sizeX_;
  }

  int nbrY(int disp){
    return (myY_ + disp - sizeY_) % sizeY_;
  }

  int linearize(int x, int y){
    return y*sizeX_ + x;
  }

  /**
    @return The number moved outside patch
  */
  int initialMove(){
    remaining_ = 3;
    return remaining_;
  }

  int move(){
    if (remaining_ > 0) --remaining_;
    return remaining_;
  }

  void solveFields(){}

  const std::vector<int>& boundaries() const {
    return boundaries_;
  }

 private:
  int remaining_;
  int sizeX_;
  int sizeY_;
  int myX_;
  int myY_;
  int myIndex_;
  std::vector<int> boundaries_;
};

struct DarmaSwarm {
 struct MigrateAccessor {
    template <class Archive>
    static void pack(Swarm& p, int local, int remote, Archive& ar){
      ar | local;
      ar | remote;
    }

    template <class Archive>
    static void unpack(Context* ctx, Swarm& p, Archive& ar){
      int remote;
      int local;
      std::vector<double> values;
      ar | remote;
      ar | local;
    }

    template <class Archive>
    static void compute_size(Swarm& p, int local, int remote, Archive& ar){
      pack(p,local,remote,ar);
    }
 };

struct MpiIn {
  template <class Archive>
  static void compute_size(Swarm& s, Archive& ar){}

  template <class Archive>
  static void pack(Swarm& s, Archive& ar){}

  template <class Archive>
  static void unpack(Swarm& s, Archive& ar){}
};

struct MpiOut {
  template <class Archive>
  static void compute_size(Swarm& s, Archive& ar){}

  template <class Archive>
  static void pack(Swarm& s, Archive& ar){}

  template <class Archive>
  static void unpack(Swarm& s, Archive& ar){}
};

 struct Move {
  void operator()(Context* ctx, int index, int iter,
                  async_ref_mm<Swarm> swarm, async_ref_mm<int> nmoved){

    if (iter == 0) *nmoved = swarm->initialMove();
    else *nmoved = swarm->move();

    auto swarm_sent = ctx->to_send(std::move(swarm));
    for (auto& bnd : swarm->boundaries()){
      swarm_sent = ctx->send<MigrateAccessor>(index,bnd,std::move(swarm_sent));
    }
    auto swarm_recvd = ctx->to_recv(std::move(swarm_sent));
    for (auto& bnd : swarm->boundaries()){
      swarm_recvd = ctx->recv<MigrateAccessor>(index,bnd,std::move(swarm_recvd));
    }
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
                  int iter, int microIter,
                  async_collection<Swarm,int> swarm, 
                  async_collection<int,int> nmoved){

    if (ctx->is_root()){
      if (microIter == 0){
        std::cout << "Starting iteration " << iter << std::endl;
      }
      std::cout << " ...running micro-iteration " << microIter << std::endl;
    }

    auto swarm_ret = ctx->modify(std::move(swarm));
    auto nmoved_ret = ctx->modify(std::move(nmoved));
    std::tie(swarm_ret, nmoved_ret) = ctx->create_phase_work<DarmaSwarm::Move>(
              ph,microIter,std::move(swarm),std::move(nmoved));

    auto total_ret = ctx->make_async_ref<int>();
    std::tie(total_ret, nmoved_ret) = ctx->phase_reduce<Add<int>>(ph, std::move(nmoved_ret));

    std::tie(total_ret) = ctx->create_work_inline([](Context* ctx, auto total){
      std::cout << "Moved a total of " << *total << std::endl;
    }, std::move(total_ret));

    //this will be so much cleaner in c++17
    auto tupleRet = ctx->make_predicate<GreaterThanZero>(std::move(total_ret));
    auto terminate = std::move(std::get<0>(tupleRet));
    std::tie(total_ret) = std::move(std::get<1>(tupleRet));

    std::tie(swarm_ret,nmoved_ret) =
      ctx->create_work_if<CollectiveMove>(std::move(terminate), ph,
         iter, microIter+1,
         std::move(swarm_ret), std::move(nmoved_ret));
    return std::make_tuple(std::move(swarm_ret), std::move(nmoved_ret));
  }
};

void run(int argc, char** argv)
{
  Swarm mainPatch;

  int od_factor = 4;
  int rank; MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int size; MPI_Comm_size(MPI_COMM_WORLD, &size);
  int darma_size = size*od_factor;

  auto dc = allocate_context(MPI_COMM_WORLD, argc, argv);

  auto nmoved_coll = dc->make_collection<int>(darma_size);

  //this object IS valid to be accessed now
  auto mpi_swarm = dc->make_local_collection<Swarm>(darma_size);

  for (int i=0; i < od_factor; ++i){
    mpi_swarm->emplaceLocal(rank*od_factor + i);
  }

  auto phase = dc->make_phase(mpi_swarm);

  //need an mpi init function here
  int niter = 10;
  for (int i=0; i < niter; ++i){
    auto part_coll = dc->from_mpi<DarmaSwarm::MpiIn>(std::move(mpi_swarm));
    std::tie(part_coll,nmoved_coll) =
      dc->create_work<CollectiveMove>(phase,i,0,std::move(part_coll),std::move(nmoved_coll));
    mpi_swarm = dc->to_mpi<DarmaSwarm::MpiOut>(std::move(part_coll));
  }
}

#define sstmac_app_name pic
int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);
  run(argc, argv);
  MPI_Finalize();
  return 0;
}


