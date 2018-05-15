#include "backend.h"
#include <iostream>

using Context=DarmaContext;

struct Sum { 
  auto operator()(Context* ctx, async_ref_mm<int> result,
                  async_ref_rr<int> n1, async_ref_rr<int> n2){
    *result = *n1 + *n2;
    return result;
  }
};

struct Print {
  auto operator()(Context* ctx, async_ref_mm<int> result, int n){
    std::cout << "Fib(" << n << "=" << *result << std::endl;
    return result;
  }
};

struct Fib {
  async_ref_nm<int> operator()(Context* ctx, async_ref_mm<int> res, int n){
    if (n == 1 || n == 0) return res;

    auto n1 = init<int>(); 
    auto n2 = init<int>();
    auto [n1_ret] = ctx->create_work<Fib>(n1,n-1);
    auto [n2_ret] = ctx->create_work<Fib>(n2,n-2);
    auto [final_result,_1,_2] = ctx->create_work<Sum>(res,n1_ret,n2_ret);

    return final_result;
  }
};

int main(){
  auto ctx = new DarmaContext(MPI_COMM_WORLD);
  int n = 5;
  auto [result] = ctx->create_work<Fib>(make_ref<int>(),n);
  auto [_] = ctx->create_work<Print>(result,n);
  return 0;
}

