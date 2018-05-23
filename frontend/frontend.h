#ifndef darma_frontend_h
#define darma_frontend_h

#include "backend_fwd.h"
#include "async_ref.h"
#include "sequencer.h"
#include "frontend_task.h"
#include "frontend_tuple_mpl.h"
#include "frontend_task_mpl.h"

template <class Backend>
struct Frontend : public Backend {

  using Context=Frontend<Backend>;

  template <class... Args>
  Frontend(Args&&... args) : Backend(std::forward<Args>(args)...)
  {}


  template <class Accessor, class LocalIndex, class RemoteIndex,
            class T, class Imm, class Sched, class... Args>
  auto send(LocalIndex&& local, RemoteIndex&& remote, async_ref<T,Imm,Sched>&& input, Args&&... args){
    using retType = typename DefaultSequencer::NewPermissions<ReadOnly,T,Imm,Sched>::type_t;
    auto ret = retType::clone(&input);
    //Backend::sequence(op, input, ret);
    auto op = Backend::template make_send_op<Accessor>(std::move(input), 
      std::forward<LocalIndex>(local), std::forward<RemoteIndex>(remote),
      std::forward<Args>(args)...);
    Backend::register_send_op(std::move(op));
    return ret;
  }

  /**
   * Given a static accessor, perform the necessary sequencing operation,
   * post a recv (or recv descriptor)
   * @param args An optional list of arguments to be returned with the finish recv, e.g.
   *  used for stashing neighbor info or other identifiers so that the accessor can know
   *  what is being received
   */
  template <class Accessor, class LocalIndex, class RemoteIndex,
            class T, class Imm, class Sched, class... Args>
  auto recv(LocalIndex&& local, RemoteIndex&& remote, async_ref<T,Imm,Sched>&& input, Args&&... args){
    using retType = typename DefaultSequencer::NewPermissions<None,T,Imm,Sched>::type_t;
    retType ret = retType::clone(&input);
    auto op = Backend::template make_recv_op<Accessor>(std::move(input), 
      std::forward<LocalIndex>(local), std::forward<RemoteIndex>(remote),
      std::forward<Args>(args)...);
    Backend::sequence(op, op.getArgument(), ret);
    Backend::register_recv_op(std::move(op));
    return ret;
  }

  template <class Accessor, class Task, class Index,
            class T, class Imm, class Sched>
  auto put_task(Index&& idx, async_ref<T,Imm,Sched>&& input){
    auto ret = async_ref<T,typename min_permissions<Idempotent,Imm>::type_t,Sched>::clone(&input);
    //auto op = Backend::template make_active_send_op<Accessor,Task>(std::move(input), std::forward<Index>(idx));
    //Backend::sequence(op, op.getArgument(), ret);
    //Backend::register_active_send_op(std::move(op));
    return ret;
  }

  template <class T, class Imm, class Sched>
  auto modify(async_ref<T,Imm,Sched>&& in){
    //recv is forcibly deferred
    return async_ref<T,None,Sched>(std::move(in));
  }

  template <class T, class Imm, class Sched>
  auto to_recv(async_ref<T,Imm,Sched>&& in){
    //recv is forcibly deferred
    return async_ref<T,typename min_permissions<Imm,ReadOnly>::type_t,Sched>(std::move(in));
  }
  
  template <class T, class Imm, class Sched>
  auto to_send(async_ref<T,Imm,Sched>&& in){
    //send is not forcibly deferred
    return async_ref<T,typename min_permissions<Imm,ReadOnly>::type_t,Sched>(std::move(in));
  }


  template <class Functor, class Predicate, class... Args>
  auto create_work_if(Predicate&& pred, Args&&... args){
    auto out = output_tuple_selector<mod_return_type_selector,sizeof...(Args),
                                     std::remove_reference_t<Args>...>()(args...);

    using body_task_t = typename task_type_selector<
        FrontendTask, Context, Functor,
        sizeof...(Args),
        std::remove_reference_t<Args>...>::type_t;
    body_task_t body(std::forward<Args>(args)...);

    using cond_task_t = std::remove_reference_t<Predicate>;

    tuple_sequencer<sizeof...(Args),0,0,body_task_t,decltype(out)>()(this,body,out);

    auto* be_task = Backend::allocate_predicate_task(std::move(pred), std::move(body));
    constexpr int predNargs = cond_task_t::nArgs;
    constexpr int bodyNargs = body_task_t::nArgs;
    tuple_register_pred_cond<predNargs,0>()(this,be_task);
    tuple_register_pred_body<bodyNargs,0>()(this,be_task);
    Backend::register_predicated_task(be_task);
    return out;
  }

  template <class Functor, class... Args>
  auto predicate(Args&&... args){
    using cond_task_t = typename task_type_selector<
        FrontendTask, Context, Functor,
        sizeof...(Args),
        std::remove_reference_t<Args>...>::type_t;
    cond_task_t t(std::forward<Args>(args)...);
    return t;
  }

  template <class Functor, class... Args>
  auto make_predicate(Args&&... args){
    auto ret = output_tuple_selector<ro_return_type_selector,
                         sizeof...(Args),
                         std::remove_reference_t<Args>...>()(args...);
    auto pred_task = predicate<Functor>(std::forward<Args>(args)...);
    tuple_sequencer<sizeof...(Args),0,0,decltype(pred_task),decltype(ret)>()(this,pred_task,ret);
    return std::make_tuple(std::move(pred_task), std::move(ret));
  }

  template <class Functor, class Phase,  class T, class Idx>
  auto phase_reduce(Phase& ph, async_ref<collection<T,Idx>,None,Modify>&& coll){
    async_ref<collection<T,Idx>,None,Modify> coll_ret(std::move(coll));

    //Backend::sequence(coll, coll_ret);

    //some sort of registration
    auto red_ret = Backend::template register_phase_reduce<Functor>(ph, std::move(coll), coll_ret);
    return std::make_tuple(std::move(red_ret), std::move(coll_ret));
  }
  
  /**
   * Perform a gather operation with the current phase on the specified collection.
   * This operation will collect all items in the collection coll onto the rank specified
   * by root.
   * 
   * @tparam Phase  The type of phase (deduced)
   * @tparam Idx    The index type used in the collection (deduced)
   * @tparam T      The type stored in the collection (deduced)
   * @param ph      The phase object of type Phase
   * @param root    The root rank used for the gather operation
   * @param coll    The collection to gather
   * @return        A tuple containing a vector of all elements in the collection, in rank order
   */
  template <class Phase, class Idx, class T>
  auto phase_gather(Phase& ph, int root, async_ref<collection<T, Idx>, None, Modify>&& coll) {
    
    // TODO: sequence params
    
    auto registered = Backend::template register_phase_gather(ph, root, std::move(coll));
    
    return std::make_tuple(async_ref<std::vector<T>, None, Modify>::make(std::move(registered)));
  }
  
  /**
   * Perform a broadcast operation from the root onto all ranks in a phase.
   * THe operation returns a collection that distributes a copy of the broadcast
   * element to all ranks in phase ph.
   * 
   * @tparam Idx    The index type of the returned collection
   * @tparam Phase  The type of phase (deduced)
   * @tparam T      The type of element to broadcast
   * @param ph      The phase of type Phase over which to broadcast
   * @param root    The rank that will broadcast ref to all other ranks
   * @param ref     The async_ref to the element to broadcast. For all ranks not root, this parameter is ignored.
   * @return        The resulting collection with a copy of the broadcasted element on each rank.
   */
  template <class Idx, class Phase, class T>
  auto phase_broadcast(Phase& ph, int root, async_ref<T, None, Modify>&& ref) {
    
    // TODO: sequence params
    
    auto registered = Backend::template register_phase_broadcast<Idx>(ph, root, std::move(ref));
    
    return std::make_tuple(async_ref<collection<T, Idx>, None, Modify>(std::move(registered)));
  }

  template <class T, class Idx>
  auto darma_collection(mpi_collection<T,Idx>& coll){
    return async_ref<collection<T,Idx>,None,Modify>::empty();
  }

  template <class BeTask>
  void register_dependencies(BeTask* be_task){
    using be_task_t = std::remove_reference_t<decltype(*be_task)>;
    tuple_register<BeTask::nArgs,0,be_task_t>()(this,be_task);
  }

  template <class Functor, class... Args>
  auto create_work(Args&&... args){
    auto out = output_tuple_selector<mod_return_type_selector,sizeof...(Args),
                                     std::remove_reference_t<Args>...>()(args...);

    using fe_task_t = typename task_type_selector<
        FrontendTask, Context, Functor,
        sizeof...(Args),
        std::remove_reference_t<Args>...>::type_t;

    fe_task_t fe_task(std::forward<Args>(args)...);
    tuple_sequencer<sizeof...(Args),0,0,fe_task_t,decltype(out)>()(this,fe_task,out);

    //allocate functions return pointers
    auto* be_task = Backend::allocate_task(std::move(fe_task));

    register_dependencies(be_task);

    Backend::register_task(be_task);

    return out;
  }

  template <class Functor, class Idx, class Imm, class Sched, class... Args>
  auto create_concurrent_work(Idx sizes, Args&&... args){
    auto out = output_tuple_selector<mod_return_type_selector,sizeof...(Args),
                                     std::remove_reference_t<Args>...>()(args...);

    using fe_task_t = GeneratorTask<Context,Functor,std::remove_reference_t<Args>...>;

    fe_task_t fe_task(std::forward<Args>(args)...);
    tuple_sequencer<sizeof...(Args),0,0,fe_task_t,decltype(out)>()(this,fe_task,out);

    //some sort of registration

    return out;
  }

  template <class Phase>
  struct InitIndexing {
    InitIndexing(Phase& ph) : ph_(ph) {}

    template <class Collection>
    void operator()(Collection& coll){
      coll->initPhase(ph_);
    }

    Phase& ph_;
  };

  template <class Functor, class Phase, class... Args>
  auto create_phase_work(Phase& ph, Args&&... args){
    auto out = output_tuple_selector<mod_return_type_selector,sizeof...(Args),
                                     std::remove_reference_t<Args>...>()(args...);

    using fe_task_t = typename task_type_selector<
        GeneratorTask, Context, Functor,
        sizeof...(Args),
        std::remove_reference_t<Args>...>::type_t;

    fe_task_t fe_task(std::forward<Args>(args)...);
    tuple_sequencer<sizeof...(Args),0,0,fe_task_t,decltype(out)>()(this,fe_task,out);

    tuple_apply_all_collection<sizeof...(Args), 0>()(fe_task.getArgs(), InitIndexing<Phase>(ph));

    //some sort of registration
    Backend::register_phase_collection(ph, std::move(fe_task));

    return out;
  }

  template <class Functor, class Phase, class... Args>
  auto create_phase_idempotent_work(Phase& ph, Args&&... args){
    auto out = output_tuple_selector<mod_return_type_selector,sizeof...(Args),
                                     std::remove_reference_t<Args>...>()(args...);

    using fe_task_t = typename task_type_selector<
        GeneratorTask, Context, Functor,
        sizeof...(Args),
        std::remove_reference_t<Args>...>::type_t;

    fe_task_t fe_task(std::forward<Args>(args)...);
    tuple_sequencer<sizeof...(Args),0,0,fe_task_t,decltype(out)>()(this,fe_task,out);

    //auto terminator = Backend::make_termination_detection();
    //add the termination detection info to all elements in the task tuple

    //some sort of registration
    //Backend::register_phase_idempotent_collection(ph, std::move(terminator), std::move(fe_task));

    return out;
  }
  
  void flush()
  {
    Backend::flush();
  }

};

#endif

