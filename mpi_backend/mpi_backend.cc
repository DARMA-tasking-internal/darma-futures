#include "mpi_backend.h"
#include <cstdlib>
#include <iostream>

MpiBackend::MpiBackend(MPI_Comm comm) :
  comm_(comm),
  collIdCtr_(0),
  numPendingProbes_(0)
{
  MPI_Comm_rank(comm, &rank_);
  MPI_Comm_size(comm, &size_);
}

void
MpiBackend::create_pending_recvs()
{
  for (int i=0; i < numPendingProbes_; ++i){
    MPI_Status stat;
    MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, comm_, &stat);
    auto& rankMap = pendingRecvs_[stat.MPI_SOURCE];
    auto iter = rankMap.find(stat.MPI_TAG);
    PendingRecvBase* recv = iter->second;
    rankMap.erase(iter);

    int size;
    void* data = allocate_temp_buffer(size);
    int reqId = allocate_request();
    MPI_Irecv(data, size, MPI_BYTE, stat.MPI_SOURCE, stat.MPI_TAG, comm_, &requests_[reqId]);
    listeners_[reqId] = recv;
    recv->configure(this, size, data);
  }
  numPendingProbes_ = 0;
}

int
MpiBackend::allocate_request()
{
  int ret = requests_.size();
  requests_.emplace_back();
  return ret;
}

void
MpiBackend::register_dependency(task* t, mpi_async_ref& in)
{
  for (int reqId : in.pendingRequests()){
    t->increment_join_counter();
    listeners_[reqId] = t;
  }
  in.clearRequests();
}

void
MpiBackend::inform_listener(int idx)
{
  Listener* listener = listeners_[idx];
  listeners_[idx] = nullptr;
  int cnt = listener->decrement_join_counter();
  if (cnt == 0){
    bool del = listener->finalize();
    if (del) delete listener;
  }
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

void*
MpiBackend::allocate_temp_buffer(int size)
{
  return new char[size];
}

void
MpiBackend::free_temp_buffer(void* buf, int size)
{
  char* cbuf = (char*) buf;
  delete [] cbuf;
}

void
MpiBackend::clear_dependencies()
{
  create_pending_recvs();
  MPI_Waitall(requests_.size(), requests_.data(), MPI_STATUSES_IGNORE);
  for (int i=0; i < requests_.size(); ++i){
    inform_listener(i);
  }
  requests_.clear();
  listeners_.clear();
}

void
MpiBackend::error(const std::string& error)
{
  std::cerr << error << std::endl;
  abort();
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
MpiBackend::make_rank_mapping(int nEntriesGlobal, std::vector<IndexInfo>& mapping, std::vector<int>& local)
{
  if (nEntriesGlobal % size_){
    error("do not yet support collections that do not evenly divide ranks");
  }
  //do a prefix sum or something in future versions
  int entriesPer = nEntriesGlobal;
  mapping.resize(nEntriesGlobal);
  for (int i=0; i < nEntriesGlobal; ++i){
    int rank = i / entriesPer;
    mapping[i].rank = rank;
    mapping[i].rankUniqueId = i % entriesPer;
    if (rank == rank_){
      local.push_back(i);
    }
  }
}

void
PendingRecvBase::clear()
{
  be_->free_temp_buffer(data_, size_);
}
