#ifndef darma_async_ref_h_
#define darma_async_ref_h_

#include <utility>
#include <tuple>
#include "permissions.h"
#include "backend_fwd.h"

template <class T, class Imm, class Sched>
struct async_ref : public async_ref_base<T> {

  using typename async_ref_base<T>::empty_tag;
  using async_ref_base<T>::Empty;

  template <class NewImm, class NewSched>
  async_ref(async_ref<T,NewImm,NewSched>&& old) : async_ref_base<T>(std::move(old)){}

  async_ref(async_ref&& t) = default; 

  async_ref(async_ref_base<T>&& old) : async_ref_base<T>(std::move(old)){}

  async_ref& operator=(async_ref&& t) = default;

  template <class... Args>
  static async_ref<T,Imm,Sched> make(Args&&... args){
    async_ref<T,Imm,Sched> ret(in_place_construct, std::forward<Args>(args)...);
    return ret;
  }

  static async_ref<T,Imm,Sched> empty(){
    return async_ref<T,Imm,Sched>{Empty};
  }

  static async_ref<T,Imm,Sched> clone(async_ref_base<T>* old){
    async_ref<T,Imm,Sched> ret{old};
    return ret;
  }

 private:
  explicit async_ref<T,Imm,Sched>(empty_tag tag) : async_ref_base<T>(tag){}

  explicit async_ref(async_ref_base<T>* old) : async_ref_base<T>(old) {}

  template <class... Args>
  explicit async_ref(in_place_construct_t, Args&&... args) :
    async_ref_base<T>(in_place_construct, std::forward<Args>(args)...)
  {}
};


template <class T, class... Args>
T& setup_ref(Args&&... args){
  T* tmp = new T(std::forward<Args>(args)...);
  return *tmp;
}

template <class T, typename... Args>
async_ref<T,Modify,Modify> init(Args&&... args){
  return async_ref<T,Modify,Modify>(std::forward<Args>(args)...);    
}

template <class T>
auto make_ref(){
  return async_ref<T,None,Modify>();
}


template <class... Args>
struct print_type;

template <class T>
using async_ref_nm = async_ref<T,None,Modify>;

template <class T>
using async_ref_nr = async_ref<T,None,ReadOnly>;

template <class T>
using async_ref_rr = async_ref<T,ReadOnly,ReadOnly>;

template <class T>
using async_ref_rm = async_ref<T,ReadOnly,Modify>;

template <class T>
using async_ref_mm = async_ref<T,Modify,Modify>;

template <class T>
using async_ref_im = async_ref<T,Idempotent,Modify>;

template <class T>
using async_ref_ii = async_ref<T,Idempotent,Idempotent>;

template <class T>
using async_ref_ri = async_ref<T,ReadOnly,Idempotent>;

template <class T, class Index>
using async_collection = async_ref<collection<T,Index>,None,Modify>;

template <class T>
template <class Imm, class Sched>
async_ref_base<T>::async_ref_base(async_ref<T,Imm,Sched>&& parent) :
  async_ref_base(std::move(static_cast<async_ref_base&&>(parent)))
{
}

/**
template <class T>
template <class Imm, class Sched>
async_ref_base<T>::async_ref_base(async_ref<T,Imm,Sched>* parent) :
  async_ref_base(parent)
{
}
*/

#endif

