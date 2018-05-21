#ifndef darma_frontend_task_h
#define darma_frontend_task_h

#include "backend_fwd.h"

template <class Context, class Functor, class... Args>
class FrontendTaskBase {
 public:
  template <class... InArgs>
  FrontendTaskBase(InArgs&&... args) : args_(std::forward<InArgs>(args)...){
  }

  std::tuple<Args...>& getArgs(){
    return args_;
  }

  static constexpr int nArgs = sizeof...(Args);

  template<typename Function, size_t ... I>
  auto call(Function f, Context* ctx, std::index_sequence<I ...>)
  {
    return f(ctx, std::move(std::get<I>(args_))...);
  }

  template<typename Function, typename Arg, size_t ... I>
  auto callWithArg(Function f, Context* ctx, Arg& arg, std::index_sequence<I ...>)
  {
    return f(ctx, arg, std::get<I>(args_) ...);
  }


 protected:
  std::tuple<Args...> args_;

};

template <class Context, class Functor, class... Args>
struct FrontendTask : public FrontendTaskBase<Context,Functor,Args...> {
  using Parent=FrontendTaskBase<Context,Functor,Args...>;

  template <class... InArgs>
  FrontendTask(InArgs&&... args) : 
    Parent(std::forward<InArgs>(args)...)
  {}

  void run(Context* ctx) {
    static constexpr auto size = sizeof...(Args);
    Parent::call(Functor(), ctx, std::make_index_sequence<size>{});
  }
};

template <class Functor, class Context, class... Args>
auto make_frontend_task(Context* ctx, Args&&... args){
  using task_t = FrontendTask<Context, Functor, std::remove_reference_t<Args>...>;
  task_t t(std::forward<Args>(args)...);
  return ctx->allocate_task(std::move(t));
}

template <class T>
struct GeneratorForwarder {
  template <class Context, class GenerateArg>
  T& operator()(Context* ctx, GenerateArg& arg, T& t){
    return t;
  }
};

template <class T, class Idx>
struct GeneratorForwarder<async_ref_base<collection<T,Idx>>> {
  template <class Context>
  auto operator()(Context* ctx, const Idx& idx, async_ref_base<collection<T,Idx>>& coll){
    auto ret = ctx->get_element(idx,coll);
    return ret;
  }
};

template<class Context, class GenerateArg, class T>
auto generatorForward(Context* ctx, GenerateArg& arg, T& t){
  return GeneratorForwarder<std::remove_reference_t<T>>()(ctx,arg,t);
}

template <class Functor, class Context, class GenerateArg>
struct TaskGenerateWrapper {
  template <class... Args>
  auto operator()(Context* ctx, GenerateArg& arg, Args&&... args){
    return make_frontend_task<Functor,Context>(ctx, arg,
            generatorForward<Context,GenerateArg,Args>(ctx,arg,args)...);
  }
};

template <class Context, class Functor, class... Args>
struct GeneratorTask : public FrontendTaskBase<Context,Functor,Args...> {

  using Parent=FrontendTaskBase<Context,Functor,Args...>;

  template <class... InArgs>
  GeneratorTask(InArgs&&... args) : 
    Parent(std::forward<InArgs>(args)...)
  {}

  template <class GeneratorArg>
  auto generate(Context* ctx, GeneratorArg& arg){
    static constexpr auto size = sizeof...(Args);
    return Parent::callWithArg(TaskGenerateWrapper<Functor,Context,GeneratorArg>(),
      ctx, arg, std::make_index_sequence<size>{}); 
  }

};

#endif

