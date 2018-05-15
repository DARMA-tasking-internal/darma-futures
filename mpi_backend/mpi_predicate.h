#ifndef mpi_predicate_h
#define mpi_predicate_h

#include "mpi_task.h"

template <class Predicate, class Task, class Context>
struct PredicateTask : public TaskBase<Context> {
  PredicateTask(Predicate&& pred, Task&& t) : 
    pred_(std::move(pred)), 
    task_(std::move(t))
  {}

  void run(Context* ctx) override {
    bool check = pred_.run(ctx);  
    if (check){
      task_.run(ctx);      
    }
  }

  auto& getConditionArgs(){
    return pred_.getArgs();
  }

  auto& getBodyArgs(){
    return task_.getArgs();
  }

  Predicate pred_;
  Task task_;
};

#endif

