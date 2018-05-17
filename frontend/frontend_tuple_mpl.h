#ifndef frontend_tuple_mpl_h
#define frontend_tuple_mpl_h

#include "mpl.h"
#include "async_ref.h"

#include <type_traits>
#include <tuple>
#include <utility>

template <int Idx, class T>
struct increment_on_async_ref {
  static constexpr int index = Idx + 1;
};

template <int Idx, class T, class Imm, class Sched>
struct increment_on_async_ref<Idx,async_ref<T,Imm,Sched>> {
  static constexpr int index = Idx;
};

template <int Remainder, int Idx, class Task>
struct tuple_register {
  template <class Context>
  void operator()(Context* ctx, Task* in){
    ctx->register_dependency(in, std::get<Idx>(in->getArgs()));
    tuple_register<Remainder-1,Idx+1,Task>()(ctx,in);
  }
};

template <int Idx, class Task>
struct tuple_register<0, Idx, Task> {
  template <class Context>
  void operator()(Context* ctx, Task* in){ /*terminate*/ }
};

template <int Remainder, int Idx>
struct tuple_register_pred_cond {
  template <class Context, class Task>
  void operator()(Context* ctx, Task* in){
    ctx->register_pred_cond_dependency(in, std::get<Idx>(in->getConditionArgs()));
    tuple_register_pred_cond<Remainder-1,Idx+1>()(ctx,in);
  }
};

template <int Idx>
struct tuple_register_pred_cond<0, Idx> {
  template <class Context, class Task>
  void operator()(Context* ctx, Task* in){ /*terminate*/ }
};

template <int Remainder, int Idx>
struct tuple_register_pred_body {
  template <class Context, class Task>
  void operator()(Context* ctx, Task* in){
    ctx->register_pred_body_dependency(in, std::get<Idx>(in->getBodyArgs()));
    tuple_register_pred_body<Remainder-1,Idx+1>()(ctx,in);
  }
};

template <int Idx>
struct tuple_register_pred_body<0, Idx> {
  template <class Context, class Task>
  void operator()(Context* ctx, Task* in){ /*terminate*/ }
};

template <int Remainder, int SrcIdx, int DstIdx, class Task, class OutTuple>
struct tuple_sequencer {
  template <class Context>
  void operator()(Context* ctx, Task& in, OutTuple& out){
    ctx->sequence(in, std::get<SrcIdx>(in.getArgs()), std::get<DstIdx>(out));
    static constexpr int NextDstIdx = increment_on_async_ref<DstIdx,
      std::remove_reference_t<decltype(std::get<DstIdx>(out))>>::index;
    tuple_sequencer<Remainder-1,SrcIdx+1,NextDstIdx,Task,OutTuple>()(ctx,in,out);
  }
};

template <int SrcIdx, int DstIdx, class Task, class OutTuple>
struct tuple_sequencer<0,SrcIdx,DstIdx,Task,OutTuple> {
  template <class Context>
  void operator()(Context* ctx, Task& in, OutTuple& out){}
};

template <template <class> class Selector, int N, class T, class... Args>
struct tuple_return_type_selector {
  using type_t = typename tuple_return_type_selector<Selector,N-1,Args...>::type_t;
};

template <template <class> class Selector, int N, class T, class Idx, class... Args>
struct tuple_return_type_selector<Selector,N,collection<Idx,T>,Args...> {
  using type_t = typename tuple_return_type_selector<Selector,N-1,Args...,
    typename Selector<collection<Idx,T>>::type_t
  >::type_t;
};

template <template <class> class Selector, int N, class T, class Imm, class Sched, class... Args>
struct tuple_return_type_selector<Selector,N,async_ref<T,Imm,Sched>,Args...> {
  using type_t = typename tuple_return_type_selector<Selector,N-1,Args...,
    typename Selector<async_ref<T,Imm,Sched>>::type_t
  >::type_t;
};

template <template <class> class Selector, class T, class... Args>
struct tuple_return_type_selector<Selector,1,T,Args...> {
  using type_t = typename reverse_tuple<sizeof...(Args),Args...>::type_t;
};

template <template <class> class Selector, class T, class Idx, class... Args>
struct tuple_return_type_selector<Selector,1,collection<Idx,T>,Args...> {
  using type_t = typename reverse_tuple<sizeof...(Args)+1,Args...,collection<Idx,T>>::type_t;
};

template <template <class> class Selector, class T, class Imm, class Sched, class... Args>
struct tuple_return_type_selector<Selector,1,async_ref<T,Imm,Sched>,Args...> {
  using type_t = typename reverse_tuple<sizeof...(Args)+1,Args...,async_ref<T,Imm,Sched>>::type_t;
};

template <template <class> class Selector, int N, class T, class... Args>
struct output_tuple_selector {
  auto operator()(T& t, Args&... args){
    return output_tuple_selector<Selector,N-1,Args...>()(args...);
  }
};

template <template <class> class Selector, int N, class T, class Imm, class Sched, class... Args>
struct output_tuple_selector<Selector,N,async_ref<T,Imm,Sched>,Args...> {
  auto operator()(async_ref<T,Imm,Sched>& t, Args&... args){
    using next_t = typename Selector<async_ref<T,Imm,Sched>>::type_t;
    next_t ret = next_t::clone(&t);
    auto front = std::make_tuple<next_t>(std::move(ret));
    auto back =  output_tuple_selector<Selector,N-1,Args...>()(args...);
    return std::tuple_cat(std::move(front), std::move(back));
  }
};

template <template <class> class Selector, class T>
struct output_tuple_selector<Selector,1,T> {
  auto operator()(T& t){
    return std::make_tuple(); //send back empty
  }
};

template <template <class> class Selector, class T, class Imm, class Sched>
struct output_tuple_selector<Selector,1,async_ref<T,Imm,Sched>> {
  auto operator()(async_ref<T,Imm,Sched>& t){
    using next_t = typename Selector<async_ref<T,Imm,Sched>>::type_t;
    next_t ret = next_t::clone(&t);
    return std::make_tuple<next_t>(std::move(ret));
  }
};


template <class T>
struct ro_return_type_selector {
  using type_t=T;
};

template <class T, class Imm, class Sched>
struct ro_return_type_selector<async_ref<T,Imm,Sched>> {
  using type_t=async_ref<T,typename min_permissions<Imm,ReadOnly>::type_t,Sched>; 
};

template <class T>
struct mod_return_type_selector {
  using type_t=T;
};

template <class T, class Imm, class Sched>
struct mod_return_type_selector<async_ref<T,Imm,Sched>> {
  using type_t=async_ref<T,None,Sched>; 
};

template <class T>
struct is_async_ref_type {
  constexpr static bool value = false;
};

template <class T, class I, class S>
struct is_async_ref_type<async_ref<T, I, S>> {
  constexpr static bool value = true;
};

template <class T>
struct is_async_ref_type<async_ref_base<T>> {
  constexpr static bool value = true;
};

template <class T>
struct is_collection_type {
  constexpr static bool value = false;
};

template <class T, class Index>
struct is_collection_type<collection<T,Index>> {
  constexpr static bool value = true;
};

namespace detail {

template <class T, class Enable = void>
struct set_async_ref_terminate_id {
  void operator()(T&& object, int terminateID_) { /*do nothing*/ }
};

template <class T>
struct set_async_ref_terminate_id<T, std::enable_if_t<is_async_ref_type<std::decay_t<T>>::value>> {
  void operator()(T&& object, int terminateID_) { 
    object.setTerminateID(terminateID_);
  }
};

} // end namespace detail

template <int Remainder, int Idx>
struct tuple_set_async_refs_terminate_id {

  template <class ArgsTupleDeduced>
  void operator()(ArgsTupleDeduced&& args_tuple, int terminateID_) {

    detail::set_async_ref_terminate_id<decltype(std::get<Idx>(args_tuple))>()(
       std::forward<decltype(std::get<Idx>(args_tuple))>(std::get<Idx>(args_tuple)),
       terminateID_
    );
 
    tuple_set_async_refs_terminate_id<Remainder-1, Idx+1>()(std::forward<ArgsTupleDeduced>(args_tuple), terminateID_);
  }

};

template <int Idx>
struct tuple_set_async_refs_terminate_id<0, Idx> {
  template <class ArgsTupleDeduced>
  void operator()(ArgsTupleDeduced&& args_tuple, int terminateID_) { /*terminate*/ }
};

namespace detail {

template <class T, class Enable = void>
struct apply_collection {
  template <class Operator>
  void operator()(T&& object, Operator& op) { /*do nothing*/ }
};

template <class T>
struct apply_collection<T, std::enable_if_t<is_collection_type<std::decay_t<T>>::value>> {
  template <class Operator>
  void operator()(T&& object, Operator& op) {
    op(object);
  }
};

} // end namespace detail

template <int Remainder, int Idx>
struct tuple_apply_all_collection {

  template <class ArgsTupleDeduced, class Operator>
  void operator()(ArgsTupleDeduced&& args_tuple, Operator& op) {

    detail::apply_collection<decltype(std::get<Idx>(args_tuple))>()(
       std::forward<decltype(std::get<Idx>(args_tuple))>(std::get<Idx>(args_tuple)),
       op
    );

    tuple_apply_all_collection<Remainder-1, Idx+1>()(std::forward<ArgsTupleDeduced>(args_tuple), op);
  }

};

template <int Idx>
struct tuple_apply_all_collection<0, Idx> {
  template <class ArgsTupleDeduced, class Operator>
  void operator()(ArgsTupleDeduced&& args_tuple, Operator& op) { /*terminate*/ }
};

#endif

