#include "mpi_backend.h"

void
MpiBackend::inform_listener(int idx)
{
  task* listener = listeners_[idx];
  listeners_[idx] = nullptr;
  listener->decrement_join_counter();
}

void
MpiBackend::progress_dependencies()
{
  int nComplete;
  MPI_Testsome(requests_.size(), requests_.data(), &nComplete,
               indices_.data(), MPI_STATUSES_IGNORE);
  //start from the end for backfilling
  for (int i=nComplete-1; i >= 0; i--){
    int idxDone = indices_[i];
    inform_listener(idxDone);
    int lastIdx = requests_.size() - 1;
    std::swap(listeners_[idxDone], listeners_[lastIdx]); 
    std::swap(requests_[idxDone], requests_[lastIdx]);
    requests_.resize(lastIdx);
    listeners_.resize(lastIdx);
  }
  
}

void
MpiBackend::progress_tasks()
{
  while(!taskQueue_.empty()){
    task* t = taskQueue_.front();
    if (t->join_counter() == 0){
      taskQueue_.pop_front();
      t->run(static_cast<Context*>(this));
    } else {
      return;
    }
  }
}

void
MpiBackend::progress_engine()
{
  progress_dependencies();
  progress_tasks();
}

void
MpiBackend::clear_dependencies()
{
  MPI_Waitall(requests_.size(), requests_.data(), MPI_STATUSES_IGNORE);
  for (int i=0; i < requests_.size(); ++i){
    inform_listener(i);
  }
  requests_.clear();
  listeners_.clear();
}

void
MpiBackend::clear_tasks()
{
  while (!taskQueue_.empty()){
    progress_dependencies();
    progress_tasks();
  }
}

void
MpiBackend::make_rank_mapping(int total_size, std::vector<int>& mapping, const std::vector<int>& local){
  //do a prefix sum or something
}

