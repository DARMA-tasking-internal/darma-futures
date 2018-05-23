#ifndef mpi_backend_h
#define mpi_backend_h

#include "mpi_async_ref.h"
#include "mpi_task.h"
#include "mpi_send_recv.h"
#include "mpi_phase.h"
#include "mpi_predicate.h"
#include "mpi_pending_recv.h"
#include "gather.h"
#include "broadcast.h"

#include <darma/serialization/simple_handler.h>
#include <darma/serialization/serializers/all.h>

#include <mpi.h>
#include <list>
#include <vector>
#include <map>

/**
  allocate_ -> implies a pointer return
  make_ -> implies a class return
*/

struct MpiBackend {
  using Context=Frontend<MpiBackend>;
  using task=TaskBase<Context>;

  static constexpr uintptr_t REQUEST_CLEAR = 0x1;

  MpiBackend(MPI_Comm comm);

  void error(const char* fmt, ...);

  void debug(const char* fmt, ...);

  Context& frontend() {
    return *static_cast<Context*>(this);
  }

  template <class T, class... Args>
  auto make_async_ref(Args&&... args){
    return async_ref<T,Modify,Modify>::make(std::forward<Args>(args)...);
  }

  template <class T, class Idx>
  auto make_collection(Idx size){
    auto ret = async_ref<collection<T,Idx>,None,Modify>::make(size);
    ret->setId(collIdCtr_++);
    return ret;
  }

  template <class Idx>
  auto make_phase(Idx idx){
    Phase<Idx> ret(idx);
    local_init_phase(ret);
    return ret;
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

  void register_dependency(task* t, mpi_async_ref& in);

  bool run_root() const {
    return true;
  }

  bool is_root() const {
    return rank_ == 0;
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
    return std::make_tuple(async_ref<collection<T,Index>,None,Modify>::make(coll.getCollection()));
  }

  template <class Accessor, class T, class Index>
  auto from_mpi(async_ref_base<collection<T,Index>>&& arg){
    //this is a fully blocking call
    clear_tasks();
    return std::make_tuple(mpi_collection<T,Index>(std::move(*arg)));
  }

  // Note: if you want to allocate the buffer using a custom allocator, make it
  //       the template parameter of this type. (If you need a stateful allocator,
  //       talk to me and I'll add that).
  using non_local_handler_t = darma::serialization::SimpleSerializationHandler<>;
  // TODO update the type of local_handler_t once the copy_constructor_archive example
  //      gets moved to a header file in DARMA serialization and organized into a handler
  using local_handler_t = darma::serialization::SimpleSerializationHandler<>;

  int makeUniqueTag(int collId, int dstId, int srcId);

  template <class Accessor, class T, class LocalIndex, class RemoteIndex, class... Args>
  auto make_send_op(async_ref_base<T>&& ref,
                    LocalIndex&& local, RemoteIndex&& remote,
                    Args&&... args){
    using index_t = std::decay_t<LocalIndex>;
    if (!ref.hasParent()){
      error("sending object with no parent collection");
    }

    auto* parent = ref.template getParent<index_t>();
    auto& dst = parent->getIndexInfo(remote);
    auto& src = parent->getIndexInfo(local);

    bool is_local = false; //push everything through MPI for now
    if(is_local) {
      //extra work needed here to put a local listener in the list
    } else {
      // The templated methods below operate on an instance, in case you need
      // something like a stateful allocator at some point in the future.
      // (All SerializationHandlers that are currently implemented, though,
      // use static methods for everything).
      auto buffer = make_packed_buffer<Accessor>(
        non_local_handler_t{}, ref,
        std::forward<LocalIndex>(local),
        std::forward<RemoteIndex>(remote),
        std::forward<Args>(args)...
      );
      send_data(ref, parent->id(), src, dst, buffer.data(), buffer.capacity());
    }

    //size
    //allocate a send buffer
    //post the MPI request with a tag from att
    SendOp<T> op(std::move(ref));
    return op;
  }

  template <class Accessor, class SerializationHandler,
            class T, class LocalIndex, class RemoteIndex, class... Args>
  auto make_packed_buffer(SerializationHandler&& handler,
                          async_ref_base<T>& ref,
                          LocalIndex&& local, RemoteIndex&& remote,
                          Args&&... args){
    auto s_ar = handler.make_sizing_archive();
    // TODO pass idx to the Accessor (if that's part of the concept?)
    Accessor::compute_size(*ref, local, remote, s_ar, std::forward<Args>(args)...);
    auto p_ar = handler.make_packing_archive(std::move(s_ar));
    // TODO forward idx to the Accessor (if that's part of the concept?)
    Accessor::pack(*ref, local, remote, p_ar, std::forward<Args>(args)...);
    return std::forward<SerializationHandler>(handler).extract_buffer(std::move(p_ar));
  }


  template <class Accessor, class Task, class T,
            class Index>
  auto make_active_send_op(async_ref_base<T>&& ref, 
                    Index&& idx){
    //size
    //allocate a send buffer
    //post the MPI request with a tag from att
    SendOp<T> op(std::move(ref));
    //MPI_Isend(..., op.getArgument().allocateRequest());
    //update my termination detection info
    return op;
  };

  template <class Accessor, class T, class LocalIndex, class RemoteIndex, class... Args>
  auto make_recv_op(async_ref_base<T>&& ref, LocalIndex&& local, RemoteIndex&& remote, Args&&... args){
    using index_t = std::decay_t<LocalIndex>;
    auto* parent = ref.template getParent<index_t>();
    auto& localEntry = parent->getIndexInfo(local);
    auto& remoteEntry = parent->getIndexInfo(remote);

    using MyRecv = NonLocalPendingRecv<Accessor,T,index_t,std::remove_reference_t<Args>...>;
    auto* pending = new MyRecv(std::forward<Args>(args)...);
    pending->setObject(ref.get());
    add_pending_recv(pending, parent->id(), localEntry, remoteEntry);
    RecvOp<T> op(std::move(ref));
    return op;
  }

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
    for (auto iter=ph->index_begin(); iter != ph->index_end(); ++iter){
      auto& local = *iter;
      auto* be_task = gen.generate(static_cast<Context*>(this),local.index);
      //these rigorously cannot have any dependencies
      //frontend().register_dependencies(be_task);
      be_task->setCounters(&local.counters);
      taskQueue_.push_back(be_task);
    }
    //flush all tasks created by this collection
    //run "bulk-synchronously" for now
    clear_tasks();
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
    auto& coll = *collIn;
    for (auto iter=ph->index_begin(); iter != ph->index_end(); ++iter){
      auto& local = *iter;
      Functor()(*coll.getElement(local.index), identity);
    }
    MPI_Allreduce(MPI_IN_PLACE,
                  Functor::mpiBuffer(identity), 
                  Functor::mpiSize(identity), 
                  Functor::mpiType(identity),
                  Functor::mpiOp(identity),
                  comm_);
    return async_ref_base<T>(in_place_construct, std::move(identity));
  }
  
  template <class Phase, class T, class Idx>
  auto register_phase_gather(Phase& ph, int root,
                                        async_ref_base<collection<T, Idx>>&& coll_in)
  {
    // Finish all pending tasks
    clear_tasks();
    return darma_backend::gather(std::move(coll_in), root, comm_);
  }
  
  template <class Idx, class Phase, class T>
  auto register_phase_broadcast(Phase& ph, int root, 
                                async_ref_base<T>&& ref_in)
  {
    // Finish all pending tasks
    clear_tasks();
    return darma_backend::broadcast<Idx>(std::move(ref_in), root, comm_);
  }

  template <class T, class Index>
  auto make_local_collection(Phase<Index>& p){
    mpi_collection<T,Index> coll(p->size_);    
    for (auto& local : p->local_){
      coll.getCollection().setElement(local.index, new T);
    }
    return coll;
  }

  template <class Index, class T>
  auto get_element(const Index& idx, async_ref_base<collection<T,Index>>& coll){
    T* t = coll->getElement(idx);
    if (t){
      async_ref_base<T> ret(coll->getElement(idx));
      ret.setParent(coll.get());
      return ret;
    } else if (!coll->initialized()){
      async_ref_base<T> ret = async_ref_base<T>::make();
      coll->setElement(idx, ret.get());
      ret.setParent(coll.get());
      return ret;
    } else {
      error("do not yet support remote get_element from collections");
      return async_ref_base<T>::make();
    }
  }

  template <class Op, class T, class U>
  void sequence(Op&& op, T&& t, U&& u){}

  template <class Op, class T, class Index>
  void sequence(Op&& op,
                async_ref_base<collection<T,Index>>& closure,
                async_ref_base<collection<T,Index>>& continuation){
    continuation->setInitialized();
  }

  void* allocate_temp_buffer(int size);
  void free_temp_buffer(void* buf, int size);
  
  void flush()
  {
    clear_tasks();
  }

 private:
  void inform_listener(int idx);
  void progress_dependencies();
  void progress_tasks();
  void progress_engine();
  void clear_dependencies();
  void clear_tasks();
  void clear_queues();
  void make_rank_mapping(int total_size, std::vector<IndexInfo>& mapping, std::vector<int>& local);
  int allocate_request();
  void create_pending_recvs();
  void add_pending_recv(PendingRecvBase* recv, int collId,
                        const IndexInfo& local, const IndexInfo& remote);
  void send_data(mpi_async_ref& in, int collId,
                 const IndexInfo& src, const IndexInfo& dst,
                 void* data, int size);

  template <class Index>
  void local_init_phase(Phase<Index>& ph){
    std::vector<int> localIndices;
    make_rank_mapping(ph->size_, ph->index_to_rank_mapping_, localIndices);
    for (int idx : localIndices){
      ph->local_.emplace_back(idx);
    }
  }

 private:
  std::vector<Listener*> listeners_;
  std::vector<int> indices_;
  std::vector<MPI_Request> requests_;
  std::list<task*> taskQueue_;
  std::map<int,std::map<int,PendingRecvBase*>> pendingRecvs_;
  MPI_Comm comm_;
  int rank_;
  int size_;
  int collIdCtr_;
  int numPendingProbes_;

};

static inline Frontend<MpiBackend>* allocate_context(MPI_Comm comm){
  return new Frontend<MpiBackend>(comm);
}

#endif

