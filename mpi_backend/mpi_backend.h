#ifndef mpi_backend_h
#define mpi_backend_h

#include "mpi_async_ref.h"
#include "mpi_task.h"
#include "mpi_send_recv.h"
#include "mpi_phase.h"
#include "mpi_predicate.h"
#include <mpi.h>
#include <list>

/**
  allocate_ -> implies a pointer return
  make_ -> implies a class return
*/

struct MpiBackend {
  using Context=Frontend<MpiBackend>;
  using task=TaskBase<Context>;

  MpiBackend(MPI_Comm comm) : comm_(comm) {}

  Context& frontend() {
    return *static_cast<Context*>(this);
  }

  template <class T, class... Args>
  auto make_async_ref(Args&&... args){
    return async_ref<T,Modify,Modify>(std::forward<Args>(args)...);
  }

  template <class T, class Idx>
  auto make_collection(Idx idx){
    return async_ref<collection<T,Idx>,None,Modify>(idx);
  }

  template <class Idx>
  auto make_phase(Idx idx){
    return Phase<Idx>(idx);
  }

  template <class FrontendTask> 
  auto allocate_task(FrontendTask&& task){
    return new Task<Context,FrontendTask>(std::move(task));
  }

  template <class ReduceFunctor>
  auto allocate_reduce_task(){
  }

  template <class PredicateOp, class DependentOp>
  auto allocate_predicate_task(PredicateOp&& pred_op, DependentOp&& dep_op){
    return new PredicateTask<PredicateOp,DependentOp,Context>(std::move(pred_op), std::move(dep_op));
  }

  void register_dependency(task* t, mpi_async_ref& in){
    for (int reqId : in.pendingRequests()){
      t->increment_join_counter();
      listeners_[reqId] = t;
    }
  }

  bool run_root() const {
    return true;
  }

  void run_worker(){}

  template <class Idx>
  void balance(Phase<Idx>& idx){}

  template <class T>
  void register_dependency(task*, T&&){
    //don't register dependencies that aren't async_refs
  }
  
  template <class T> //no ops if not async refs
  void register_pred_cond_dependency(task*, T&&){}

  void register_pred_cond_dependency(task* t, mpi_async_ref& in){
    register_dependency(t,in);
  }

  template <class T> //no ops if not async refs
  void register_pred_body_dependency(task*, T&&){}

  void register_pred_body_dependency(task* t, mpi_async_ref& in){
    register_dependency(t,in);
  }

  void register_task(task* t){
    if (t->join_counter() == 0) taskQueue_.push_back(t);
  }

  void register_predicated_task(task* t){
    //don't do anything special for predicate tasks
    register_task(t);
  }

  //template <class PackFunctor, class UnpackFunctor, class TaskFunctor,
  //          template <class> Ref, class T, class Index, class... Args>
  //auto make_active_send_op(Ref<T>&& ref, idempotent_task_base<T>& acc, Index&& idx, Args&&... args){
  //  SendOp<Ref,T> op(std::move(ref));
  //  //MPI_Isend(..., op.getArgument().allocateRequest());
  //  return op;
  //}

  //todo - make this a set of variadic args
  //todo - have the frontend do most of the work for this
  template <class Accessor, class T, class Index>
  auto to_mpi(mpi_collection<T,Index>&& coll){
    //no load balancing yet, so this does nothing
    return std::make_tuple(async_ref<collection<T,Index>,None,Modify>(std::move(coll.getCollection())));
  }

  template <class Accessor, class T, class Index>
  auto from_mpi(async_ref_base<collection<T,Index>>&& arg){
    //this is a fully blocking call
    clear_tasks();
    return std::make_tuple(mpi_collection<T,Index>(std::move(*arg)));
  }

  template <class Accessor, class T,
            class Index, class... Args>
  auto make_send_op(async_ref_base<T>&& ref, 
                    Index&& idx, Args&&... args){
    //size
    //allocate a send buffer
    //post the MPI request with a tag from att
    SendOp<T> op(std::move(ref));
    //MPI_Isend(..., op.getArgument().allocateRequest());
    return op;
  };

  template <class Accessor, class Task, class T,
            class Index, class... Args>
  auto make_active_send_op(async_ref_base<T>&& ref, 
                    Index&& idx, Args&&... args){
    //size
    //allocate a send buffer
    //post the MPI request with a tag from att
    SendOp<T> op(std::move(ref));
    //MPI_Isend(..., op.getArgument().allocateRequest());
    //update my termination detection info
    return op;
  };

  template <class Accessor, class T, class Index, class... Args>
  auto make_recv_op(async_ref_base<T>&& ref, 
                    Index&& idx, Args&&... args){
    //size
    //allocate a send buffer
    //post the MPI request with a tag from att
    RecvOp<T> op(std::move(ref));
    //MPI_Irecv(..., op.getArgument().allocateRequest());
    return op;
  };

  template <class SendOp>
  auto register_send_op(SendOp&& op){
    //already done
  };

  template <class SendOp>
  auto register_active_send_op(SendOp&& op){
    //already done
  };

  template <class RecvOp>
  auto register_recv_op(RecvOp&& op){
    //already done
  };

  template <class Phase, class GeneratorTask>
  void register_phase_collection(Phase& ph, GeneratorTask&& gen){
    clear_tasks();
    //ensure that all of these tasks 
    for (int idx : ph.indices()){
      auto* be_task = gen.generate(static_cast<Context*>(this),idx);
      //these rigorously cannot have any dependencies
      //frontend().register_dependencies(be_task);
      taskQueue_.push_back(be_task);
    }
  }

  template <class Phase, class Terminator, class GeneratorTask>
  void register_phase_idempotent_collection(Phase& ph, Terminator&& term, GeneratorTask&& gen){
    register_phase_collection(ph, std::move(gen));
    //add a task that will keep looping until termination detection is achieved
    //taskQueue_.push_back(terminate_task);
  }

  template <class Functor, class Phase, class T, class Idx>
  auto register_phase_reduce(Phase& ph, async_ref_base<collection<T,Idx>>&& collIn,
                                        async_ref_base<collection<T,Idx>>& collOut)
  {
    clear_tasks();
    auto identity = Functor::identity();
    //ensure that all of these tasks 
    auto& coll = collIn;
    for (int idx : ph.indices()){
      Functor()(*coll->getElement(idx), identity);
    }
    MPI_Allreduce(MPI_IN_PLACE,
                  Functor::mpiBuffer(identity), 
                  Functor::mpiSize(identity), 
                  Functor::mpiType(identity),
                  Functor::mpiOp(identity),
                  comm_);
    return async_ref_base<T>(std::move(identity));
  }

  template <class Index>
  void local_init_phase(Phase<Index>& ph, std::vector<Index>& idx){
    ph.local_ = idx;
    make_rank_mapping(ph.size_, ph.index_to_rank_mapping_, idx);
  }

  template <class T, class Index>
  auto make_local_collection(Phase<Index>& p){
    mpi_collection<T,Index> coll(p.size_);    
    for (auto idx : p.local_){
      coll.getCollection().setElement(idx, new T);
    }
    return coll;
  }

  template <class Index, class T>
  auto getElement(Index& idx, async_ref_base<collection<T,Index>>& coll){
    return async_ref_base<T>(coll->getElement(idx));
  }

  template <class Op, class T, class U>
  void sequence(Op&& op, T&& t, U&& u){}

 private:
  void inform_listener(int idx);
  void progress_dependencies();
  void progress_tasks();
  void progress_engine();
  void clear_dependencies();
  void clear_tasks();
  void clear_queues();
  void make_rank_mapping(int total_size, std::vector<int>& mapping, const std::vector<int>& local);

 private:
  std::vector<task*> listeners_;
  std::vector<int> indices_;
  std::vector<MPI_Request> requests_;
  std::list<task*> taskQueue_;
  MPI_Comm comm_;

};

static inline Frontend<MpiBackend>* allocate_context(MPI_Comm comm){
  return new Frontend<MpiBackend>(comm);
}

#endif
