#include "mpi_backend.h"
#include "mpi_index_entry.h"
#include <cstdlib>
#include <iostream>
#include <cstdarg>
#include <sstream>
#include <CLI/CLI.hpp>

#if DARMA_ZOLTAN_LB
#include <z2LoadBalancer.hpp>
#endif

int MpiBackend::taskIdCtr_ = 1;
std::vector<RecvOpGeneratorBase<MpiBackend::Context>*> MpiBackend::generators_;

static inline std::string str_tolower(std::string&& s) {
  std::transform(s.begin(), s.end(), s.begin(),
    [](unsigned char c){ return std::tolower(c); } // correct
  );
  return s;
}

static void perfCtrReduceFxn(void* a, void* b, int* len, MPI_Datatype* type)
{
  //length better be 1
  MpiBackend::PerfCtrReduce* in = (MpiBackend::PerfCtrReduce*) a;
  MpiBackend::PerfCtrReduce* inout = (MpiBackend::PerfCtrReduce*) b;

  inout->max = std::max(inout->max, in->max);
  inout->min = std::min(inout->min, in->min);
  inout->total += in->total;
  inout->maxLocalTasks = std::max(inout->maxLocalTasks, in->maxLocalTasks);
}

MpiBackend::MpiBackend(MPI_Comm comm, int argc, char** argv) :
  comm_(comm),
  collIdCtr_(0),
  numPendingProbes_(0)
{
  auto& fe = frontend();
  int app_argc = fe.split_argv(argc, argv);
  //argv[app_argc] = "--", or app_argc = argc
  int be_argc = argc - app_argc;
  char** be_argv = argv + app_argc;
  //'--' is treated as 'app' name

  CLI::App app{"DARMA MPI Backend"};
  std::string lbType = "commSplit";
  app.add_option("--lb", lbType, "the load balancer type to use");
  try {
    app.parse(be_argc, be_argv);
  } catch (const CLI::ParseError &e) {
    int rc = app.exit(e);
    exit(rc);
  }

  std::map<std::string, lb_type_t> lbs{
    { "random", RandomLB },
    { "commsplit", CommSplitLB },
#if DARMA_ZOLTAN_LB
    { "zoltan", ZoltanLB },
#endif
    { "debug", DebugLB },
  };

  auto iter = lbs.find(str_tolower(std::move(lbType)));
  if (iter == lbs.end()){
    std::cerr << "Supported load balancers are:\n";
    for (auto& pair : lbs){
      std::cerr << pair.first << "\n";
    }
    error("Invalid load balancer %s specified - either type-o or not configured to support",
          lbType.c_str());
  }
  lbType_ = iter->second;

  MPI_Comm_rank(comm, &rank_);
  MPI_Comm_size(comm, &size_);

  requests_.reserve(1024);
  statuses_.reserve(1024);
  indices_.resize(1024);




  int ierr = MPI_Type_vector(1, 4, 0, MPI_UINT64_T, &perfCtrType_);
  if (ierr != MPI_SUCCESS){
    error("Unable to create performance counter reduce type");
  }
  MPI_Type_commit(&perfCtrType_);

  ierr = MPI_Op_create(perfCtrReduceFxn, 1, &perfCtrOp_);
  if (ierr != MPI_SUCCESS){
    error("Unable to create performance counter reduce op");
  }

}

struct TagMaker {
  unsigned int frontPadding : 1,
    collId : 8,
    dstId : 9,
    srcId : 9,
    taskId : 4,
    backPadding : 1;
};

template <int T>
struct print_size;

int
MpiBackend::makeUniqueTag(int collId, int dstId, int srcId, int taskId){
  //this is dirty - but don't hate
  TagMaker tag;
  tag.frontPadding = 0;
  tag.collId = collId;
  tag.dstId = dstId;
  tag.srcId = srcId;
  tag.taskId = taskId; //zero means no task
  tag.backPadding = 0;
  static_assert(sizeof(TagMaker) <= sizeof(int), "tag is small enough");
  return *reinterpret_cast<int*>(&tag);
}

std::set<int>
MpiBackend::takeTasks(uint64_t desiredDelta, const std::vector<pair64>& giver)
{
  std::set<int> toRet;
  uint64_t deltaCutoff = desiredDelta / 10;
  uint64_t maxGiveAway = desiredDelta + deltaCutoff;
  uint64_t totalGiven = 0;
  uint64_t remainingDelta = maxGiveAway;
  for (int i=giver.size() - 1; i >= 0; --i){
    uint64_t taskSize = giver[i].first;
    //std::cout << "Rank " << rank_
    //          << " considering give/take of size=" << taskSize
    //          << " for index " << giver[i].second
    //          << std::endl;
    if (taskSize < remainingDelta){
      toRet.insert(i); //give it away, give it away, give it away now
      totalGiven += taskSize;
      remainingDelta -= taskSize;
    } else {
      break; //can do no better
    }
  }
  return toRet;
}

uint64_t
MpiBackend::tradeTasks(uint64_t desiredDelta,
                       const std::vector<pair64>& bigger,
                       const std::vector<pair64>& smaller,
                       int& bigTaskIdx, int& smallTaskIdx)
{
  //I assume the smaller, bigger are sorted least to greatest coming in
  smallTaskIdx = 0;
  size_t smallTaskStop = smaller.size() - 1;
  bigTaskIdx = bigger.size() - 1;
  int bestBigIdx = bigTaskIdx, bestSmallIdx = smallTaskIdx;
  uint64_t bestDeltaDelta = std::numeric_limits<uint64_t>::max();
  uint64_t smallTaskSize = 0, bigTaskSize = 1; //just to get us started
  while (smallTaskIdx <= smallTaskStop && bigTaskIdx >= 0 && smallTaskSize < bigTaskSize){
    smallTaskSize = smaller[smallTaskIdx].first;
    bigTaskSize = bigger[bigTaskIdx].first;
    uint64_t delta = bigTaskSize - smallTaskSize;
    //if (rank_ == 0){
      //std::cout << "Considering " << bigTaskSize << "<->" << smallTaskSize
      //          << " for delta=" << delta << "<>" << desiredDelta << std::endl;
    //}

    if (desiredDelta > delta){
      uint64_t delta_delta = desiredDelta - delta;
      if (bestDeltaDelta < delta_delta){
        //this is only getting worse - return what we had before
        smallTaskIdx = bestSmallIdx;
        bigTaskIdx = bestBigIdx;
        return bestDeltaDelta;
      }
      //this is the best I can do - I hope it's good enough
      return delta_delta;
    } else {
      uint64_t delta_delta = delta - desiredDelta;
      if (bestDeltaDelta < delta_delta){
        //this is only going to get worse - return what we had before
        smallTaskIdx = bestSmallIdx;
        bigTaskIdx = bestBigIdx;
        return bestDeltaDelta;
      }

      bestDeltaDelta = delta_delta;
      bestSmallIdx = smallTaskIdx;
      bestBigIdx = bigTaskIdx;

      uint64_t smallTaskDelta = std::numeric_limits<uint64_t>::max();
      uint64_t bigTaskDelta = std::numeric_limits<uint64_t>::max();
      //increment whichever index causes the least change
      if (smallTaskIdx < smallTaskStop)
        smallTaskDelta = smaller[smallTaskIdx+1].first - smaller[smallTaskIdx].first;
      if (bigTaskIdx > 0)
        bigTaskDelta = bigger[bigTaskIdx].first - bigger[bigTaskIdx-1].first;

      //depending on which produces the smallest delta, change that index
      if (bigTaskDelta < smallTaskDelta) bigTaskIdx--;
      else smallTaskIdx++;
    }
  }
  //the closest thing we have is the biggest task on the small side, smallest task on the big side
  smallTaskIdx = smallTaskStop;
  bigTaskIdx = 0;
  return bestDeltaDelta;
}

std::vector<MpiBackend::pair64>
MpiBackend::balance(const std::vector<LocalIndex>& local)
{
  std::vector<pair64> localConfig(local.size());
  for (int i=0; i < local.size(); ++i){
    pair64& p = localConfig[i];
    const LocalIndex& lidx = local[i];
    p.first = lidx.counters.counter;
    p.second = lidx.index;
  }
  return balance(std::move(localConfig));
}

std::vector<MpiBackend::pair64>
MpiBackend::balance(std::vector<pair64>&& localConfig)
{
  switch(lbType_){
    case ZoltanLB:
      return zoltanBalance(std::move(localConfig));
    case CommSplitLB:
      return commSplitBalance(std::move(localConfig));
    case RandomLB:
      return randomBalance(std::move(localConfig));
    case DebugLB:
      return debugBalance(std::move(localConfig));
  }
}

void
MpiBackend::create_pending_recvs()
{
  for (int i=0; i < numPendingProbes_; ++i){
    MPI_Status stat;
    MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, comm_, &stat);

    //this might be an incoming task - or it might just be a message
    PendingRecvBase* recv;
    TagMaker* tagger = (TagMaker*) &stat.MPI_TAG;
    if (tagger->taskId != 0){
      //this delivered a task to me
      RecvOpGeneratorBase<Context>* gen = generators_[tagger->taskId];
      recv = gen->generate(frontendPtr(), tagger->dstId, tagger->collId);
    } else {
      auto& rankMap = pendingRecvs_[stat.MPI_SOURCE];
      auto iter = rankMap.find(stat.MPI_TAG);
      if (iter == rankMap.end()){
        error("unabled to find tag %d", stat.MPI_TAG);
      }
      auto& list = iter->second;
      recv = list.front();
      list.pop_front();
      if (list.empty()){
        rankMap.erase(iter);
      }
    }

    int size; MPI_Get_count(&stat, MPI_BYTE, &size);
    void* data = allocate_temp_buffer(size);
    int reqId = recv->id();
    MPI_Irecv(data, size, MPI_BYTE, stat.MPI_SOURCE, stat.MPI_TAG, comm_,
              &requests_[reqId]);
    recv->configure(this, size, data);
  }
  numPendingProbes_ = 0;
}

void
MpiBackend::add_pending_recv(PendingRecvBase* pending, int collId,
                             const IndexInfo& local, const IndexInfo& remote)
{
  int tag = makeUniqueTag(collId, local.rankUniqueId, remote.rankUniqueId);
  pending->increment_join_counter();
  pendingRecvs_[remote.rank][tag].push_back(pending);
  int reqId = allocate_request();
  pending->setId(reqId);
  listeners_[reqId] = pending;
  requests_[reqId] = MPI_REQUEST_NULL;
  ++numPendingProbes_;
}

int
MpiBackend::allocate_request()
{
  int ret = requests_.size();
  requests_.emplace_back();
  listeners_.emplace_back();
  statuses_.emplace_back();
  return ret;
}

void
MpiBackend::register_dependency(task* t, mpi_async_ref& in)
{
  for (int reqId : in.pendingRequests()){
    if (listeners_[reqId] == (void*)REQUEST_CLEAR){
      //oh, nothing to do
      listeners_[reqId] = nullptr;
    } else if (listeners_[reqId]){
      error("listener should be null or cleared");
    } else {
      listeners_[reqId] = t;
      t->increment_join_counter();
    }
  }
  in.clearRequests();
}

void
MpiBackend::inform_listener(int idx)
{
  Listener* listener = listeners_[idx];
  if (listener){
    int cnt = listener->decrement_join_counter();
    if (cnt == 0){
      bool del = listener->finalize();
      if (del) delete listener;
      listeners_[idx] = nullptr;
    }
  } else {
    listeners_[idx] = (Listener*) REQUEST_CLEAR;
  }
}

void
MpiBackend::progress_dependencies()
{
  create_pending_recvs();
  int nComplete;
  MPI_Testsome(requests_.size(), requests_.data(), &nComplete,
               indices_.data(), statuses_.data());
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
      uint64_t t_start = rdtsc();
      t->run(static_cast<Context*>(this));
      uint64_t t_stop = rdtsc();
      t->addCounter(t_stop-t_start);
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
MpiBackend::error(const char *fmt, ...)
{
  va_list args;
  va_start(args, fmt);
  vfprintf(stderr, fmt, args);
  va_end(args);
  fflush(stderr);
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
MpiBackend::send_data(mpi_async_ref& ref, int collId,
                      const IndexInfo& src, const IndexInfo& dst,
                      void* data, int size, int taskId)
{
  int tag = makeUniqueTag(collId, dst.rankUniqueId, src.rankUniqueId, taskId);
  int request = allocate_request();
  ref.addRequest(request);
  send_data(dst.rank, data, size, tag, &requests_[request]);

}

void
MpiBackend::send_data(int dest, void *data, int size, int tag, MPI_Request *req)
{
  MPI_Isend(data, size, MPI_BYTE, dest, tag, comm_, req);
}

void
MpiBackend::recv_data(int src, void *data, int size, int tag, MPI_Request *req)
{
  MPI_Irecv(data, size, MPI_BYTE, src, tag, comm_, req);
}

void
MpiBackend::make_global_mapping_from_local(int total_size, const std::vector<int>& local, std::vector<IndexInfo>& mapping)
{
  //okay, this is not the way I want to have to do this
  int myNumLocal = local.size();
  int maxNumLocal;
  MPI_Allreduce(&myNumLocal, &maxNumLocal, 1, MPI_INT, MPI_MAX, comm_);
  std::vector<int> allIndices(maxNumLocal*size_);
  std::vector<int> indices(maxNumLocal, -1);
  for (int i=0; i < local.size(); ++i){
    indices[i] = local[i];
  }

  MPI_Allgather(indices.data(), maxNumLocal, MPI_INT, allIndices.data(), maxNumLocal, MPI_INT, comm_);
  mapping.resize(total_size);

  std::vector<int> rankCounts(size_);
  for (int i=0; i < size_; ++i){
    int* currentBlock = &allIndices[i*maxNumLocal];
    for (int localIndex=0; localIndex < maxNumLocal && currentBlock[localIndex] != -1; ++localIndex){
      int globalIndex = currentBlock[localIndex];
      mapping[globalIndex].rank = i;
      mapping[globalIndex].rankUniqueId = rankCounts[i]++;
    }
  }

  /**
  std::stringstream sstr;
  sstr << "Rank=" << rank_ << " maps { ";
  for (int i=0; i < total_size; ++i){
    sstr << " " << i << ":" << mapping[i].rank;
  }
  sstr << "}\n";
  std::cout << sstr.str();
  */
}

void
MpiBackend::make_rank_mapping(int nEntriesGlobal, std::vector<IndexInfo>& mapping, std::vector<int>& local)
{
  if (nEntriesGlobal % size_){
    error("do not yet support collections that do not evenly divide ranks");
  }
  //do a prefix sum or something in future versions
  int entriesPer = nEntriesGlobal / size_;
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
MpiBackend::reset_phase(const std::vector<pair64>& config,
                        std::vector<LocalIndex>& local,
                        std::vector<IndexInfo>& indices)
{
  for (int i=0; i < local.size() && i < config.size(); ++i){
    LocalIndex& lidx = local[i];
    const pair64& pair = config[i];
    lidx.index = pair.second;
    lidx.counters.counter = 0;
  }

  int oldSize = local.size();
  int newSize = config.size();
  for (int i=oldSize; i < newSize; ++i){
    const pair64& pair = config[i];
    local.emplace_back(pair.second);
  }

  for (int i=newSize; i < oldSize; ++i){
    local.pop_back();
  }

  std::vector<int> localIndices;
  for (LocalIndex& lidx : local){
    localIndices.push_back(lidx.index);
  }

  /**
  std::cout << "Rank=" << rank_  << " now has {";
  for (auto& lidx  :local){
    std::cout << " " << lidx.index;
  }
  std::cout << "}" << std::endl;
  */
  make_global_mapping_from_local(indices.size(), localIndices, indices);
}

void
MpiBackend::rebalance(std::vector<migration>& objToSend,
               std::vector<migration>& objToRecv)
{
  int rebalance_info_tag = 444;
  int rebalance_data_tag = 445;

  static const int numInfoFields = 3;

  int numSends = objToSend.size();
  int numRecvs = objToRecv.size();
  std::vector<MPI_Request> sendDataReqs(numSends);
  std::vector<MPI_Request> sendInfoReqs(numSends);
  std::vector<MPI_Request> recvInfoReqs(numRecvs);
  std::vector<MPI_Request> recvDataReqs(numRecvs);
  std::vector<int> sendInfos(numSends*numInfoFields);
  std::vector<int> recvInfos(numRecvs*numInfoFields);

  for (int i=0; i < numSends; ++i){
    const migration& m = objToSend[i];
    int* info = &sendInfos[numInfoFields*i];
    info[0] = m.size;
    info[1] = m.index;
    info[2] = m.mpiParent;
    send_data(m.rank, info, sizeof(int)*numInfoFields, rebalance_info_tag, &sendInfoReqs[i]);
    send_data(m.rank, m.buf, m.size, rebalance_data_tag, &sendDataReqs[i]);
  }

  for (int i=0; i < numRecvs; ++i){
    migration& m = objToRecv[i];
    int* info = &recvInfos[numInfoFields*i];
    recv_data(m.rank, info, sizeof(int)*numInfoFields, rebalance_info_tag, &recvInfoReqs[i]);
  }

  MPI_Waitall(numSends, sendInfoReqs.data(), MPI_STATUSES_IGNORE);
  MPI_Waitall(numRecvs, recvInfoReqs.data(), MPI_STATUSES_IGNORE);

  for (int i=0; i < numRecvs; ++i){
    int* info = &recvInfos[numInfoFields*i];
    migration& m = objToRecv[i];
    m.size = info[0];
    m.index = info[1];
    m.mpiParent = info[2];
    m.buf = allocate_temp_buffer(m.size);
    recv_data(m.rank, m.buf, m.size, rebalance_data_tag, &recvDataReqs[i]);
  }

  MPI_Waitall(numRecvs, recvDataReqs.data(), MPI_STATUSES_IGNORE);
  MPI_Waitall(numSends, sendDataReqs.data(), MPI_STATUSES_IGNORE);
}

void
PendingRecvBase::clear()
{
}

void
PendingRecvBase::configure(MpiBackend* be, int size, void* data)
{
  be_ = static_cast<Frontend<MpiBackend>*>(be);
  size_ = size;
  data_ = data;
}


