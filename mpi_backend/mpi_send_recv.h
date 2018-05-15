#ifndef mpi_send_recv_h
#define mpi_send_recv_h

#include "mpi_async_ref.h"

template <class T>
struct SendOp {
  SendOp(async_ref_base<T>&& ref) : toSend_(ref){}

  async_ref_base<T>& getArgument(){
    return toSend_;
  }

  async_ref_base<T> toSend_;
};

template <class T>
struct RecvOp {
  RecvOp(async_ref_base<T>&& ref) : toRecv_(ref){}

  async_ref_base<T>& getArgument(){
    return toRecv_;
  }

  async_ref_base<T> toRecv_;
};

template <class T, class Idx, class Accessor>
struct send_recv {
};

template <class T, class Idx, class Accessor>
struct mpi_accessor {
};

template <class T, class Idx, class Accessor, class Functor>
struct idempotent_task_accessor {
};


#endif

