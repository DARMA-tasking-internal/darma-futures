#ifndef darma_async_ref_h_
#define darma_async_ref_h_

#include <utility>
#include <tuple>
#include "permissions.h"
#include "backend_fwd.h"

template <class T, class Imm, class Sched>
struct async_ref : public async_ref_base<T> {
  template <class... Args>
  async_ref(Args&&... args) : 
    async_ref_base<T>(std::forward<Args>(args)...)
  {}

  auto modify(){
    return async_ref<T,None,Sched>();
  }

  auto read(){
    return async_ref<T,None,Sched>();
  }

  async_ref(async_ref&& t) = default; 

  async_ref& operator=(async_ref&& t) = default;
  async_ref& operator=(async_ref& t) = default;

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

#endif

