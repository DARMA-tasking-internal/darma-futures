#ifndef task_mpl_h
#define task_mpl_h

#include "frontend_task.h"

template <template <class,class,class...> class Task, class Context, class Functor,
          int N, class T, class... Args>
struct task_reverse_args {
  using type_t = typename task_reverse_args<Task,Context,Functor,N-1,Args...,T>::type_t;
};

template <template <class,class,class...> class Task, class Context, class Functor,
          class T, class... Args>
struct task_reverse_args<Task,Context,Functor,1,T,Args...> {
  using type_t = Task<Context,Functor,Args...,T>; 
};

template <template <class,class,class...> class Task, class Context, class Functor,
          int N, class T, class... Args>
struct task_type_selector {
  using type_t = typename task_type_selector<Task,Context,Functor,N-1,Args...,T>::type_t;
};

template <template <class,class,class...> class Task, class Context, class Functor,
          int N, class T, class Imm, class Sched, class... Args>
struct task_type_selector<Task,Context,Functor,N,async_ref<T,Imm,Sched>,Args...> {
  using type_t = typename task_type_selector<Task,Context,Functor,N-1,Args...,async_ref_base<T>>::type_t;
};

template <template <class,class,class...> class Task, class Context, class Functor,
          class T, class Imm, class Sched, class... Args>
struct task_type_selector<Task,Context,Functor,1,async_ref<T,Imm,Sched>,Args...> {
  using type_t = typename task_reverse_args<Task,Context,Functor,sizeof...(Args)+1,Args...,async_ref_base<T>>::type_t;
};

template <template <class,class,class...> class Task, class Context, class Functor, class T, class... Args>
struct task_type_selector<Task,Context,Functor,1,T,Args...> {
  using type_t = typename task_reverse_args<Task,Context,Functor,sizeof...(Args)+1,Args...,T>::type_t;
};

#endif

