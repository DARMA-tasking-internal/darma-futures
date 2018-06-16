#ifndef mpi_listener_h
#define mpi_listener_h

#include <memory>

struct Listener {

  Listener() : next_(nullptr), join_counter_(0){}

  virtual ~Listener(){}

  int increment_join_counter(){ 
    ++join_counter_; 
    return join_counter_;
  }

  int decrement_join_counter(){ 
    --join_counter_; 
    return join_counter_;
  }

  int join_counter() const {
    return join_counter_;
  }

  /**
   * @brief finalize
   * Perform any operations necessary to finalize the listener
   * when the join counter goes to zero
   * @return Whether the listener should be deleted after finalizing
   */
  virtual bool finalize(){
    return false;
  }

 private:
  Listener* next_;
  int join_counter_;
};

#endif

