#ifndef mpi_task_h
#define mpi_task_h

#include "frontend.h"

template <class Context>
struct TaskBase {
  TaskBase() : join_counter_(0), counters_(nullptr) {}

  void increment_join_counter(){ ++join_counter_; }

  int decrement_join_counter(){ --join_counter_; return join_counter_; }

  int join_counter() const {
    return join_counter_;
  }

  virtual void run(Context* ctx) = 0;

  bool hasCounters() const {
    return counters_;
  }

  void setCounters(PerformanceCounter* ctr){
    ctr = counters_;
  }

 private:
  PerformanceCounter* counters_;
  int join_counter_;
};

template <class Context, class FrontendTask>
struct Task : public TaskBase<Context> {
  Task(FrontendTask&& fe_task) : 
    fe_task_(std::move(fe_task)) {}

  void run(Context* ctx) override {
    fe_task_.run(ctx);
  }

  auto& getArgs(){
    return fe_task_.getArgs();
  }

  static constexpr int nArgs = FrontendTask::nArgs;

  FrontendTask fe_task_;
};

#endif

