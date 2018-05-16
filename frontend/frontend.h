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


  template <class Accessor, class Index,
            class T, class Imm, class Sched>
  auto send(Index&& idx, async_ref<T,Imm,Sched>&& input){
    using retType = typename DefaultSequencer::NewPermissions<ReadOnly,T,Imm,Sched>::type_t;
    retType ret;
    auto op = Backend::template make_send_op<Accessor>(std::move(input), std::forward<Index>(idx));
    Backend::sequence(op, op.getArgument(), ret);
    Backend::register_send_op(std::move(op));
    return ret;
  }

  template <class Accessor, class Index,
            class T, class Imm, class Sched>
  auto recv(Index&& idx, async_ref<T,Imm,Sched>&& input){
    using retType = typename DefaultSequencer::NewPermissions<None,T,Imm,Sched>::type_t;
    retType ret;
    auto op = Backend::template make_recv_op<Accessor>(std::move(input), std::forward<Index>(idx));
    Backend::sequence(op, op.getArgument(), ret);
    Backend::register_recv_op(std::move(op));
    return ret;
  }

  template <class Accessor, class Task, class Index,
            class T, class Imm, class Sched>
  auto put_task(Index&& idx, async_ref<T,Imm,Sched>&& input){
    async_ref<T,typename min_permissions<Idempotent,Imm>::type_t,Sched> ret;
    auto op = Backend::template make_active_send_op<Accessor,Task>(std::move(input), std::forward<Index>(idx));
    Backend::sequence(op, op.getArgument(), ret);
    Backend::register_active_send_op(std::move(op));
    return ret;
  }

  template <class T, class Imm, class Sched>
  auto modify(async_ref<T,Imm,Sched>&& in){
    return async_ref<T,None,Sched>(std::move(in));
  }
  
  template <class T, class Imm, class Sched>
  auto read(async_ref<T,Imm,Sched>&& in){
    return async_ref<T,typename min_permissions<Imm,ReadOnly>::type_t,Sched>(std::move(in));
  }


  template <class Functor, class Predicate, class... Args>
  auto create_work_if(Predicate&& pred, Args&&... args){
    using out_tuple = typename tuple_return_type_selector<mod_return_type_selector,
                         sizeof...(Args),
                         std::remove_reference_t<Args>...>::type_t;
    out_tuple out;

    using body_task_t = typename task_type_selector<
        FrontendTask, Context, Functor,
        sizeof...(Args),
        std::remove_reference_t<Args>...>::type_t;
    body_task_t body(std::forward<Args>(args)...);

    using cond_task_t = std::remove_reference_t<Predicate>;

    tuple_sequencer<sizeof...(Args),0,0,body_task_t,out_tuple>()(this,body,out);

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

    using cond_tuple = typename tuple_return_type_selector<ro_return_type_selector,
                         sizeof...(Args),
                         std::remove_reference_t<Args>...>::type_t;
    cond_tuple ret;
    
    auto pred_task = predicate<Functor>(std::forward<Args>(args)...);

    tuple_sequencer<sizeof...(Args),0,0,decltype(pred_task),cond_tuple>()(this,pred_task,ret);

    return tuple_cat(std::make_tuple(std::move(pred_task)), std::move(ret));
  }

  template <class Functor, class Phase, class Idx, class T>
  auto phase_reduce(Phase& ph, async_ref<collection<Idx,T>,None,Modify>&& coll){
    async_ref<collection<Idx,T>,None,Modify> coll_ret; 

    //Backend::sequence(coll, coll_ret);

    //some sort of registration
    auto red_ret = Backend::template register_phase_reduce<Functor>(ph, std::move(coll), coll_ret);

    return std::make_tuple(async_ref<T,None,Modify>(std::move(red_ret)), coll_ret);
  }

  template <class T, class Idx>
  auto darma_collection(mpi_collection<T,Idx>& coll){
    return async_ref<collection<T,Idx>,None,Modify>();
  }

  template <class BeTask>
  void register_dependencies(BeTask* be_task){
    using be_task_t = std::remove_reference_t<decltype(*be_task)>;
    tuple_register<BeTask::nArgs,0,be_task_t>()(this,be_task);
  }

  template <class Functor, class... Args>
  auto create_work(Args&&... args){
    using out_tuple = typename tuple_return_type_selector<mod_return_type_selector,
                                  sizeof...(Args),
                                  std::remove_reference_t<Args>...>::type_t;
    out_tuple out;

    using fe_task_t = typename task_type_selector<
        FrontendTask, Context, Functor,
        sizeof...(Args),
        std::remove_reference_t<Args>...>::type_t;

    fe_task_t fe_task(std::forward<Args>(args)...);
    tuple_sequencer<sizeof...(Args),0,0,fe_task_t,out_tuple>()(this,fe_task,out);

    //allocate functions return pointers
    auto* be_task = Backend::allocate_task(std::move(fe_task));

    register_dependencies(be_task);

    Backend::register_task(be_task);

    return out;
  }

  template <class Functor, class Idx, class Imm, class Sched, class... Args>
  auto create_concurrent_work(Idx sizes, Args&&... args){
    using out_tuple = typename tuple_return_type_selector<mod_return_type_selector,
                                  sizeof...(Args),
                                  std::remove_reference_t<Args>...>::type_t;
    out_tuple ret;

    using fe_task_t = GeneratorTask<Context,Functor,std::remove_reference_t<Args>...>;

    fe_task_t fe_task(std::forward<Args>(args)...);
    tuple_sequencer<sizeof...(Args),0,0,fe_task_t,out_tuple>()(this,fe_task,ret);

    //some sort of registration

    return ret;
  }

  template <class Functor, class Phase, class... Args>
  auto create_phase_work(Phase& ph, Args&&... args){
    using out_tuple = typename tuple_return_type_selector<mod_return_type_selector,
                                  sizeof...(Args),
                                  std::remove_reference_t<Args>...>::type_t;
    out_tuple out;

    using fe_task_t = typename task_type_selector<
        GeneratorTask, Context, Functor,
        sizeof...(Args),
        std::remove_reference_t<Args>...>::type_t;

    fe_task_t fe_task(std::forward<Args>(args)...);
    tuple_sequencer<sizeof...(Args),0,0,fe_task_t,out_tuple>()(this,fe_task,out);

    //some sort of registration
    Backend::register_phase_collection(ph, std::move(fe_task));

    return out;
  }

  template <class Functor, class Phase, class... Args>
  auto create_phase_idempotent_work(Phase& ph, Args&&... args){
    using out_tuple = typename tuple_return_type_selector<mod_return_type_selector,
                                  sizeof...(Args),
                                  std::remove_reference_t<Args>...>::type_t;
    out_tuple out;

    using fe_task_t = typename task_type_selector<
        GeneratorTask, Context, Functor,
        sizeof...(Args),
        std::remove_reference_t<Args>...>::type_t;

    fe_task_t fe_task(std::forward<Args>(args)...);
    tuple_sequencer<sizeof...(Args),0,0,fe_task_t,out_tuple>()(this,fe_task,out);

    //auto terminator = Backend::make_termination_detection();
    //add the termination detection info to all elements in the task tuple

    //some sort of registration
    //Backend::register_phase_idempotent_collection(ph, std::move(terminator), std::move(fe_task));

    return out;
  }

};

#endif

