#include "mpi_backend.h"
#include <vector>
#include <iostream>
#include <cmath>

using Context=Frontend<MpiBackend>;

struct Patch {
 public:
  friend struct DarmaPatch;

  Patch(){}

  double timestep(int index, int iter){
    double residual = 0;
    for (int i=1; i < nelems_-1; ++i){
      double uxx = values_[i+1] + values_[i-1] - 2*values_[i];
      residual += uxx;
      values_[i] += alpha_ * uxx;
    }
    return fabs(residual);
  }

  void init(int index, int size, int nelems, double alpha){
    nelems_ = nelems;
    myIndex_ = index;
    values_.resize(nelems);
    for (int i=1; i < nelems-1; ++i){
      values_[i] = myIndex_;
    }
    alpha_ = alpha;

    int left = (index - 1 + size) % size;
    int right = (index + 1) % size;
    boundaries_.resize(2);
    boundaries_[0] = left;
    boundaries_[1] = right;
  }

  const std::vector<int>& boundaries() const {
    return boundaries_;
  }

  int index() const {
    return myIndex_;
  }
  
 private:
  std::vector<int> boundaries_;
  std::vector<double> values_;
  int myIndex_;
  int size_;
  int nelems_;
  double alpha_;

};


struct DarmaPatch {
  struct GhostAccessor {
    template <class Archive>
    static void pack(Patch& p, int local, int remote, Archive& ar, int nbr){
      ar | local;
      ar | remote;
      if (nbr == 0) ar | p.values_[1]; //the beginning is a ghost
      else ar | p.values_[p.nelems_-2]; //the end is a ghost
      //pack a vector or something
    }

    template <class Archive>
    static void unpack(Context* ctx, Patch& p, Archive& ar, int localNbr){
      int myGlobalIdx;
      int remoteGlobalIdx;
      std::vector<double> values;
      ar | remoteGlobalIdx;
      ar | myGlobalIdx;
      if (localNbr == 0) ar | p.values_[0];
      else ar | p.values_[p.nelems_-1];
      //loop incoming values from that neighbor and put them in correct location
    }

    template <class Archive>
    static void compute_size(Patch& p, int local, int remote, Archive& ar, int nbr){
      pack(p,local,remote,ar,nbr);
    }
  };

  struct SendRecv {
    auto operator()(Context* ctx, async_ref_mm<Patch>&& patch){
      auto& neighbors = patch->boundaries();
      auto patch_sent = ctx->to_send(std::move(patch));
      for (int n=0; n < neighbors.size(); ++n){
        patch_sent = ctx->send<GhostAccessor>(patch->index(),neighbors[n],std::move(patch_sent),n);
      }

      auto patch_recvd = ctx->to_recv(std::move(patch_sent));
      for (int n=0; n < neighbors.size(); ++n){
        //sent index as extra argument
        patch_recvd = ctx->recv<GhostAccessor>(patch->index(),neighbors[n],std::move(patch_recvd),n);
      }
      return patch_recvd;
    }
  };

  struct Timestep {
    void operator()(Context* ctx, int index, int iter,
                    async_ref_mm<Patch> patch,
                    async_ref_mm<double> residual){
      *residual = patch->timestep(index, iter);
      auto finalPatch = SendRecv()(ctx, std::move(patch));
    }
  };

  struct Init {
    void operator()(Context* ctx, int index, int nelems, int size, double alpha,
                    async_ref_mm<Patch> patch){
      patch->init(index,size,nelems,alpha);
      auto finalPatch = SendRecv()(ctx, std::move(patch));
    }
  };


  struct Print {
    void operator()(Context* ctx, int iter, async_ref_mm<double> residual){
      std::cout << "Iter " << iter << " Residual=" << residual << std::endl;
    }
  };
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
  static MPI_Datatype mpiType(double& i){ return MPI_DOUBLE; }
  static MPI_Op mpiOp(double& i){ return MPI_SUM; }

  void operator()(const T& in, T& out){
    out += in;
  }

};

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);
  int od_factor = 4;
  int rank; MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int size; MPI_Comm_size(MPI_COMM_WORLD, &size);
  int darma_size = size*od_factor;
  int nelems = 10;
  double alpha = 0.01;

  auto dc = allocate_context(MPI_COMM_WORLD);
  auto coll = dc->make_collection<Patch>(darma_size);
  auto residuals = dc->make_collection<double>(darma_size);
  auto phase = dc->make_phase(darma_size);

  if (dc->run_root()){
    std::tie(coll) = dc->create_phase_work<DarmaPatch::Init>(phase,nelems,darma_size,alpha,std::move(coll));
    int niter = 10;
    for (int i=0; i < niter; ++i){
      std::tie(coll,residuals) = dc->create_phase_work<DarmaPatch::Timestep>(phase,i,
                                                  std::move(coll),std::move(residuals));
      auto residual = dc->make_async_ref<double>();
      std::tie(residual,residuals) = dc->phase_reduce<Add<double>>(phase, std::move(residuals));
      std::tie(residual) = dc->create_work<DarmaPatch::Print>(i,std::move(residual));
      if (i % 5 == 0) dc->balance(phase);
    }
  } else {
    dc->run_worker();
  }

  MPI_Finalize();
}


