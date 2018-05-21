#ifndef mpi_pending_recv_h
#define mpi_pending_recv_h

#include "mpi_listener.h"
#include <tuple>
#include <darma/serialization/simple_handler.h>
#include <darma/serialization/serializers/all.h>

struct MpiBackend;
struct PendingRecvBase : public Listener {
  PendingRecvBase() : listener_(nullptr) {}

  void configure(MpiBackend* be, int size, void* data){
    be_ = be;
    size_ = size; 
    data_ = data;
  }

  void setListener(Listener* listener){
    listener_ = listener;
  }

  void setId(int id){
    id_ = id;
  }

  int id() const {
    return id_;
  }

  void clear();

  using non_local_handler_t = darma::serialization::SimpleSerializationHandler<>;
  using local_handler_t = darma::serialization::SimpleSerializationHandler<>;

 protected:
  void* data_;
  int size_;
  int id_;
  Listener* listener_;
  MpiBackend* be_;
};

template <class Accessor, class T, class Index>
struct PendingRecv : public PendingRecvBase {


  template<class Archive, class Tuple, size_t ... I>
  void call(Archive&& ar, Tuple&& tuple, std::index_sequence<I ...>)
  {
    //do not forward archive - always put into unpack as l-value
    Accessor::unpack(*t_, ar, std::get<I>(std::move(tuple))...);
  }

  template <class Handler, class Tuple>
  void unpack(Handler&& handler, Tuple&& t) {
    auto u_ar = handler.make_unpacking_archive(
      darma::serialization::NonOwningSerializationBuffer(data_, size_));
    static constexpr auto size = std::tuple_size<std::remove_reference_t<Tuple>>::value;
    call(std::move(u_ar), std::forward<Tuple>(t), std::make_index_sequence<size>{});
  }


  void setObject(T* t){
    t_ = t;
  }

 private:
  T* t_;
};

template <class Accessor, class T, class Index, class... Args>
struct LocalPendingRecv : public PendingRecv<Accessor,T,Index> 
{
  template <class... InArgs>
  LocalPendingRecv(InArgs&&... args) : args_(std::forward<InArgs>(args)...){}

  using Parent=PendingRecv<Accessor,T,Index>;
  using typename Parent::local_handler_t;
  using Parent::unpack;
  using Parent::clear;
  using Parent::listener_;
  bool finalize() override {
    PendingRecv<Accessor,T,Index>::unpack(local_handler_t{}, std::move(args_));
    if (listener_) listener_->decrement_join_counter();
    clear();
    return true; //this is done
  }

  std::tuple<Args...> args_;
};

template <class Accessor, class T, class Index, class... Args>
struct NonLocalPendingRecv : public PendingRecv<Accessor,T,Index> 
{
  template <class... InArgs>
  NonLocalPendingRecv(InArgs&&... args) : args_(std::forward<InArgs>(args)...){}

  using Parent=PendingRecv<Accessor,T,Index>;
  using typename Parent::non_local_handler_t;
  using Parent::unpack;
  using Parent::clear;
  using Parent::listener_;

  bool finalize() override {
    PendingRecv<Accessor,T,Index>::unpack(non_local_handler_t{}, std::move(args_));
    if (listener_) listener_->decrement_join_counter();
    clear();
    return true; //this is done;
  }

  std::tuple<Args...> args_;
};

#endif

