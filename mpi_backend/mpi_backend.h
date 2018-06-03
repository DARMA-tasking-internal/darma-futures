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
#include <set>

template <class Accessor, class T, class Index>
int recv_task_id();

static inline uint64_t rdtsc(void)
{
  uint32_t hi, lo;
  __asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
  return uint64_t( (uint64_t)lo | (uint64_t)hi<<32);
}

struct MpiBackend {
  struct PerfCtrReduce {
    uint64_t total;
    uint64_t max;
    uint64_t min;
    uint64_t maxLocalTasks;
  };

  using Context=Frontend<MpiBackend>;
  using task=TaskBase<Context>;

  static constexpr uintptr_t REQUEST_CLEAR = 0x1;

  MpiBackend(MPI_Comm comm);

  ~MpiBackend(){
    clear_tasks();
  }

  void error(const char* fmt, ...);

  void debug(const char* fmt, ...);

  Context& frontend() {
    return *static_cast<Context*>(this);
  }

  Context* frontendPtr() {
    return static_cast<Context*>(this);
  }

  template <class T, class... Args>
  auto make_async_ref(Args&&... args){
    return async_ref<T,Modify,Modify>::make(std::forward<Args>(args)...);
  }

  template <class T, class Idx>
  auto make_collection(Idx size){
    auto ret = async_ref<collection<T,Idx>,None,Modify>::make(size);
    ret->setId(collIdCtr_++);
    collections_[ret->id()] = ret.get();
    return ret;
  }

  template <class T, class Idx>
  auto make_phase(mpi_collection_ptr<T,Idx>& coll){
    Phase<Idx> ret(coll->size());
    std::vector<int> local;
    for (auto& pair : coll->localElements()){
      local.push_back(pair.first);
      ret->local_.emplace_back(pair.first);
    }
    make_global_mapping_from_local(coll->size(), local, ret->index_to_rank_mapping_);
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
  void rebalance(Phase<Idx>& ph){
    std::vector<pair64> newConfig = balance(ph->local());
    for (auto& pair : newConfig){
      std::cout << "Rank=" << rank_ << " now has index=" << pair.second
                << std::endl;
    }
    reset_phase(newConfig, ph->local_, ph->index_to_rank_mapping_);
  }

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
  auto from_mpi(mpi_collection_ptr<T,Index>&& mpi_coll){
    //no load balancing yet, so this does nothing
    if (mpi_coll->referencesDarmaCollection()){
      collection<T,Index>* darma_coll = mpi_coll->darmaCollection();
      //this was remapped from a previous collection
      darma_coll->assignMpi(std::move(mpi_coll));
      return async_ref<collection<T,Index>,None,Modify>::make(darma_coll);
    } else {
      //no collection ever existed, so better make it now
      auto ref = async_ref<collection<T,Index>,None,Modify>::make(mpi_coll->size(), mpi_coll->localElements());
      ref->setId(collIdCtr_++);
      std::cout << "Making DARMA collection from MPI" << std::endl;
      ref->assignMpi(std::move(mpi_coll));
      return ref;
    }
  }

  template <class Accessor, class T, class Index>
  auto to_mpi(async_ref_base<collection<T,Index>>&& arg){
    //this is a fully blocking call
    clear_tasks();
    if (!arg->hasMpiParent())
      error("darma collection cannot return an MPI collection if no MPI collection was originally moved in");

    return arg->moveMpiParent();
  }

  // Note: if you want to allocate the buffer using a custom allocator, make it
  //       the template parameter of this type. (If you need a stateful allocator,
  //       talk to me and I'll add that).
  using non_local_handler_t = darma::serialization::SimpleSerializationHandler<>;
  // TODO update the type of local_handler_t once the copy_constructor_archive example
  //      gets moved to a header file in DARMA serialization and organized into a handler
  using local_handler_t = darma::serialization::SimpleSerializationHandler<>;

  int makeUniqueTag(int collId, int dstId, int srcId, int taskId = 0 /*zero means no task*/);

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

  template <class Accessor, class SerializationHandler, class T, class... Args>
  auto make_packed_buffer(SerializationHandler&& handler,
                          async_ref_base<T>& ref, Args&&... args){
    auto s_ar = handler.make_sizing_archive();
    // TODO pass idx to the Accessor (if that's part of the concept?)
    Accessor::compute_size(*ref, s_ar, std::forward<Args>(args)...);
    auto p_ar = handler.make_packing_archive(std::move(s_ar));
    // TODO forward idx to the Accessor (if that's part of the concept?)
    Accessor::pack(*ref, p_ar, std::forward<Args>(args)...);
    return std::forward<SerializationHandler>(handler).extract_buffer(std::move(p_ar));
  }


  template <class Accessor, class T, class Index, class... Args>
  auto make_active_send_op(async_ref_base<T>&& ref, Index&& idx, Args&&... args){
    using index_t = std::decay_t<LocalIndex>;
    if (!ref.hasParent()){
      error("sending object with no parent collection");
    }

    auto* parent = ref.template getParent<index_t>();
    auto& dst = parent->getIndexInfo(idx);
    bool is_local = false; //push everything through MPI for now
    if(is_local) {
      //extra work needed here to put a local listener in the list
    } else {
      // The templated methods below operate on an instance, in case you need
      // something like a stateful allocator at some point in the future.
      // (All SerializationHandlers that are currently implemented, though,
      // use static methods for everything).
      auto buffer = make_packed_buffer<Accessor>(non_local_handler_t{}, ref,
                                                 std::forward<Args>(args)...);
      IndexInfo src; //the source doesn't actuall matter here
      src.rank = rank_;
      src.rankUniqueId = 0;
      send_data(ref, parent->id(), src, dst, buffer.data(), buffer.capacity(),
                recv_task_id<Accessor,T,Index>());
    }

    //size
    //allocate a send buffer
    //post the MPI request with a tag from att
    SendOp<T> op(std::move(ref));
    return op;
  }

  template <class Accessor, class T, class LocalIndex, class RemoteIndex, class... Args>
  auto make_recv_op(async_ref_base<T>&& ref, LocalIndex&& local, RemoteIndex&& remote, Args&&... args){
    using index_t = std::decay_t<LocalIndex>;
    auto* parent = ref.template getParent<index_t>();
    auto& localEntry = parent->getIndexInfo(local);
    auto& remoteEntry = parent->getIndexInfo(remote);

    using MyRecv = NonLocalPendingRecv<Accessor,T,index_t,std::remove_reference_t<Args>...>;
    auto* pending = new MyRecv(std::move(ref), std::forward<Args>(args)...);
    add_pending_recv(pending, parent->id(), localEntry, remoteEntry);
    RecvOp<T> op{};
    return op;
  }

  template <class Accessor, class Index, class T>
  auto rebalance(Phase<Index>& ph, async_collection<T,Index>&& coll){
    int rebalance_info_tag = 444;
    int rebalance_data_tag = 445;
    int currentInfoReq = 0;
    int currentDataReq = 0;

    static async_ref_base<T>* dummy;
    using pack_buf_t = decltype(make_packed_buffer<Accessor>(non_local_handler_t{}, *dummy));
    std::vector<pack_buf_t> packers;

    int numSends = 0;
    int numRecvs = 0;
    for (auto& pair : coll->localElements()){
      int index = pair.first;
      int newLoc = ph->getRank(index);
      if (newLoc != rank_){
        ++numSends;
      }
    }
    for (const LocalIndex& lidx : ph->local()){
      int oldLoc = coll->getRank(lidx.index);
      if (oldLoc != rank_){
        ++numRecvs;
      }
    }

    std::vector<MPI_Request> sendDataReqs(numSends);
    std::vector<MPI_Request> sendInfoReqs(numSends);
    std::vector<MPI_Request> recvInfoReqs(numRecvs);
    std::vector<MPI_Request> recvDataReqs(numRecvs);
    std::vector<int> sendInfos(numSends*2);
    std::vector<Index> toDel(numSends);
    std::vector<int> recvInfos(numRecvs*2);
    std::vector<int> sources(numRecvs);

    for (auto& pair : coll->localElements()){
      int index = pair.first;
      int newLoc = ph->getRank(index);
      std::cout << "Index=" << index << " lives on " << rank_
                << " but will live on " << newLoc << std::endl;
      if (newLoc != rank_){
        std::cout << "Rank=" << rank_ << " has " << index
                  << " but rank=" << newLoc << " needs it" << std::endl;
        //I have to send data
        T* elem = pair.second;
        async_ref_base<T> toRecv = async_ref_base<T>::make(elem);
        auto buffer = make_packed_buffer<Accessor>(non_local_handler_t{}, toRecv);
        int* info = &sendInfos[2*currentInfoReq];
        toDel[currentInfoReq] = index;
        info[0] = buffer.capacity();
        info[1] = index;
        send_data(newLoc, info, sizeof(int)*2, rebalance_info_tag, &sendInfoReqs[currentInfoReq++]);
        send_data(newLoc, buffer.data(), buffer.capacity(), rebalance_data_tag, &sendDataReqs[currentDataReq++]);
        packers.emplace_back(std::move(buffer));
      }
    }

    MPI_Waitall(numSends, sendInfoReqs.data(), MPI_STATUSES_IGNORE);


    //add validation pass ensure the incoming data is what we expect
    currentInfoReq = 0;
    for (const LocalIndex& lidx : ph->local()){
      int oldLoc = coll->getRank(lidx.index);
      if (oldLoc != rank_){
        int* info = &recvInfos[2*currentInfoReq];
        sources[currentInfoReq] = oldLoc;
        std::cout << "Rank=" << rank_
                  << " wants to receive index " << lidx.index << " from " << oldLoc
                  << " wrote " << oldLoc << " to " << &sources[currentInfoReq]
                  << std::endl;
        recv_data(oldLoc, info, sizeof(int)*2, rebalance_info_tag, &recvInfoReqs[currentInfoReq++]);
      }
    }

    MPI_Waitall(numRecvs, recvInfoReqs.data(), MPI_STATUSES_IGNORE);

    std::vector<void*> recvBufs(numRecvs);
    for (int i=0; i < numRecvs; ++i){
      int* info = &recvInfos[2*i];
      int index = info[1];
      int size = info[0];
      void* buf = allocate_temp_buffer(size);
      recvBufs[i] = buf;
      std::cout << "Rank=" << rank_
                << " receiving " << size << " bytes "
                << " in buffer=" << buf
                << " from " << sources[i] << " at " << &sources[i]
                << " to transfer index " << index << std::endl;
      recv_data(sources[i], buf, size, rebalance_data_tag, &recvDataReqs[i]);
    }

    MPI_Waitall(numRecvs, recvDataReqs.data(), MPI_STATUSES_IGNORE);
    MPI_Waitall(numSends, sendDataReqs.data(), MPI_STATUSES_IGNORE);

    for (auto& idx : toDel){
      std::cout << "Rank " << rank_ << " would like to delete " << idx << std::endl;
      coll->remove(idx);
    }

    for (int i=0; i < numRecvs; ++i){
      int* info = &recvInfos[2*i];
      int index = info[1];
      int size = info[0];
      void* buf = recvBufs[i];
      non_local_handler_t handler{};
      T* newT = coll->emplaceNew(index);
      std::cout << "Rank=" << rank_
                << " unpacking " << size << " bytes "
                << " in buffer=" << buf
                << " to complete index " << index << std::endl;
      auto u_ar = handler.make_unpacking_archive(
        darma::serialization::NonOwningSerializationBuffer(buf, size));
      Accessor::unpack(*newT, u_ar);
      free_temp_buffer(buf, size);
    }

    async_collection<T,Index> ret(std::move(coll));
    return ret;
  }

  template <class SendOp>
  auto register_send_op(SendOp&& op){
    //already done
  }

  template <class SendOp>
  auto register_active_send_op(SendOp&& op){
    //already done
  }

  template <class RecvOp>
  auto register_recv_op(RecvOp&& op){
    //already done
  }

  template <class Phase, class GeneratorTask>
  void register_phase_collection(Phase& ph, GeneratorTask&& gen){
    clear_tasks();
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
  auto make_local_collection(Index size){
    return std::make_unique<mpi_collection<T,Index>>(size);
  }

  template <class T>
  auto get_collection_element(int id, int idx){
    //hope this is an int
    collection<T,int>* coll = static_cast<collection<T,int>*>(collections_[idx]);
    return get_element(idx, coll);
  }

  template <class Index, class T>
  auto get_element(const Index& idx, collection<T,Index>* coll){
    T* t = coll->getElement(idx);
    if (t){
      async_ref_base<T> ret(coll->getElement(idx));
      ret.setParent(coll);
      return ret;
    } else if (!coll->initialized()){
      async_ref_base<T> ret = async_ref_base<T>::make();
      coll->setElement(idx, ret.get());
      ret.setParent(coll);
      return ret;
    } else {
      error("do not yet support remote get_element from collections: index %d on rank %d", idx, rank_);
      return async_ref_base<T>::make();
    }
  }

  template <class Index, class T>
  auto get_element(const Index& idx, async_ref_base<collection<T,Index>>& coll){
    return get_element<Index,T>(idx, coll.get());
  }

  template <class Op, class T, class U>
  void sequence(Op&& op, T&& t, U&& u){}

  template <class Op, class T, class Index>
  void sequence(Op&& op,
                async_ref_base<collection<T,Index>>& closure,
                async_ref_base<collection<T,Index>>& continuation){
    continuation->setInitialized();
  }

  template <class Accessor, class T, class Index>
  static int register_recv_generator(){
    int id = taskIdCtr_++;
    generators_.push_back(new RecvOpGenerator<Context,Accessor,T,Index>);
    return id;
  }

  void* allocate_temp_buffer(int size);
  void free_temp_buffer(void* buf, int size);
  
  void flush()
  {
    clear_tasks();
  }

 private:
  using pair64 = std::pair<uint64_t,uint64_t>;
  struct sortByWeight {
    bool operator()(const pair64& lhs, const pair64& rhs) const {
      return lhs.first < rhs.first;
    }
  };

  void inform_listener(int idx);
  void progress_dependencies();
  void progress_tasks();
  void progress_engine();
  void clear_dependencies();
  void clear_tasks();
  void clear_queues();
  void make_global_mapping_from_local(int total_size, const std::vector<int>& local,
                                      std::vector<IndexInfo>& mapping);
  void make_rank_mapping(int total_size, std::vector<IndexInfo>& mapping, std::vector<int>& local);
  int allocate_request();
  void create_pending_recvs();
  void add_pending_recv(PendingRecvBase* recv, int collId,
                        const IndexInfo& local, const IndexInfo& remote);
  void send_data(mpi_async_ref& in, int collId,
                 const IndexInfo& src, const IndexInfo& dst,
                 void* data, int size, int taskId = 0 /*zero means no task*/);

  void send_data(int dest, void* data, int size, int tag, MPI_Request* req);
  void recv_data(int src, void* data, int size, int tag, MPI_Request* req);

  void reset_phase(const std::vector<pair64>& config,
                   std::vector<LocalIndex>& local,
                   std::vector<IndexInfo>& indices);

  template <class Index>
  void local_init_phase(Phase<Index>& ph){
    std::vector<int> localIndices;
    make_rank_mapping(ph->size_, ph->index_to_rank_mapping_, localIndices);
    for (int idx : localIndices){
      ph->local_.emplace_back(idx);
    }
  }

  uint64_t tradeTasks(uint64_t desiredDelta,
                  const std::vector<pair64>& bigger,
                  const std::vector<pair64>& smaller,
                  int& biggerIdx, int& smallerIdx);

  std::set<int> takeTasks(uint64_t desiredDelta, const std::vector<pair64>& giver);

  /**
   * @brief balance
   * @param local
   * @return The new local configuraiton
   */
  std::vector<pair64> balance(const std::vector<LocalIndex>& local);

  /**
   * @brief balance
   * @param localConfig
   * @return The new local configuraiton
   */
  std::vector<pair64> balance(std::vector<pair64>&& localConfig);

  /**
   * @brief runBalancer
   * @param localConfig
   * @param newLocalConfig in-out return of new local config after moving tasks
   * @param localWork The total amount of work currently local
   * @param globalWork  The total amount of work globally
   * @param maxNumLocalTasks  The max number of tasks on any given node
   * @param allowGiveTake whether to rigorously enforce only "exchanging" tasks
   *         or to allow giving/taking tasks that change num local
   */
  void runBalancer(std::vector<pair64>&& localConfig,
      std::vector<pair64>& newLocalConfig,
      uint64_t localWork, uint64_t globalWork,
      int maxNumLocalTasks, bool allowTrades, bool allowGiveTake);

 private:
  std::vector<Listener*> listeners_;
  std::vector<int> indices_;
  std::vector<MPI_Request> requests_;
  std::vector<MPI_Status> statuses_;
  std::list<task*> taskQueue_;
  std::map<int,collection_base*> collections_;
  std::map<int,std::map<int,std::list<PendingRecvBase*>>> pendingRecvs_;
  MPI_Comm comm_;
  int rank_;
  int size_;
  int collIdCtr_;
  int numPendingProbes_;
  static int taskIdCtr_;
  static std::vector<RecvOpGeneratorBase<Context>*> generators_;

  //for idempotent task regions
  int activeWindow_;

  MPI_Op perfCtrOp_;
  MPI_Datatype perfCtrType_;

  MPI_Op perfCtrBalanceOp_;
  MPI_Datatype perfCtrBalanceType_;

};

template <class Accessor, class T, class Index>
int recv_task_id(){
  return MpiBackend::register_recv_generator<Accessor,T,Index>();
}

static inline auto allocate_context(MPI_Comm comm){
  return std::make_unique<Frontend<MpiBackend>>(comm);
}

#endif

