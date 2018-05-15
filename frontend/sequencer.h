#ifndef darma_default_sequencer_h
#define darma_default_sequencer_h

template <class Closure, class Return> 
struct ValidateSequencing {
  static constexpr bool valid = true;
};

template <class T, class ImmClosure, class ImmReturn> 
struct ValidateSequencing<async_ref<T,ImmClosure,ReadOnly>,async_ref<T,ImmReturn,None>> {
  static constexpr bool valid = false;
};

template <class T, class ImmClosure, class ImmReturn> 
struct ValidateSequencing<async_ref<T,ImmClosure,Idempotent>,async_ref<T,ImmReturn,ReadOnly>> {
  static constexpr bool valid = false;
};

template <class T, class ImmClosure, class ImmReturn> 
struct ValidateSequencing<async_ref<T,ImmClosure,Modify>,async_ref<T,ImmReturn,ReadOnly>> {
  static constexpr bool valid = false;
};

struct DefaultSequencer {

  template <class NewImm, class T, class Imm, class Sched>
  struct NewPermissions {
    using type_t = async_ref<T, typename min_permissions<ReadOnly,Imm>::type_t, Sched>;
  };
  
  template <class Context, class Operation, class Closure, class Return> 
    void sequence(Context* ctx, Operation&& op, Closure& cl, Return& ret){
    //pick_sequence_action<Closure,Return>()(ctx,std::forward<Operation>(op),cl,ret);
  }
};

#endif

