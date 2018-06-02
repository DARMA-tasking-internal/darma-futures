#include "mpi_backend.h"
#include "mpi_index_entry.h"
#include <cstdlib>
#include <iostream>
#include <cstdarg>

int MpiBackend::taskIdCtr_ = 1;
std::vector<RecvOpGeneratorBase<MpiBackend::Context>*> MpiBackend::generators_;

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

MpiBackend::MpiBackend(MPI_Comm comm) :
  comm_(comm),
  collIdCtr_(0),
  numPendingProbes_(0)
{
  MPI_Comm_rank(comm, &rank_);
  MPI_Comm_size(comm, &size_);

  requests_.reserve(1024);
  statuses_.reserve(1024);
  indices_.resize(1024);

  MPI_Type_vector(1, 4, 0, MPI_UINT64_T, &perfCtrType_);
  MPI_Op_create(perfCtrReduceFxn, 1, &perfCtrOp_);
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
  uint64_t smallTaskSize = smaller[smallTaskIdx].first;
  uint64_t bigTaskSize = bigger[bigTaskIdx].first;
  int bestBigIdx = -1, bestSmallIdx = -1;
  uint64_t bestDeltaDelta = std::numeric_limits<uint64_t>::max();
  while (smallTaskIdx <= smallTaskStop && bigTaskIdx >= 0 && smallTaskSize < bigTaskSize){
    uint64_t delta = bigTaskSize - smallTaskSize;
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
  static const int maxNumTries = 5;
  static const double diffCutoff = 0.15;
  int tryNum = 0;
  double maxDiffFraction;

  std::vector<pair64> oldConfig = std::move(localConfig);

  while(1) {
    if (tryNum >= maxNumTries){
      return oldConfig;
    }

    uint64_t localWork = 0;
    uint64_t minWork = std::numeric_limits<uint64_t>::max();
    uint64_t maxWork = std::numeric_limits<uint64_t>::min();
    std::vector<uint64_t> outgoingSizes(localConfig.size());
    for (int i=0; i < oldConfig.size(); ++i){
      auto& pair = oldConfig[i];
      uint64_t weight = pair.first;
      localWork += weight;
      minWork = std::min(minWork, weight);
      maxWork = std::max(maxWork, weight);
      outgoingSizes[i] = weight;
    }

    PerfCtrReduce local;
    local.min = minWork;
    local.max = maxWork;
    local.total = localWork;
    local.maxLocalTasks = localConfig.size();
    PerfCtrReduce global;

    MPI_Allreduce(&local, &global, 1, perfCtrType_, perfCtrOp_, comm_);
    uint64_t perfBalance = global.total / size_;
    uint64_t maxDiff = global.max - global.min;
    maxDiffFraction = double(maxDiff) / double(perfBalance);
    if (maxDiffFraction < diffCutoff){
      return oldConfig;
    }

    bool allowGiveTake = tryNum >= 3;
    //sort the old config by task weight
    std::sort(oldConfig.begin(), oldConfig.end(), sortByWeight());
    std::vector<pair64> newConfig;
    runBalancer(std::move(oldConfig), newConfig,
        localWork, global.total, global.maxLocalTasks,
         allowGiveTake);

    ++tryNum;

    oldConfig = std::move(newConfig);
  }
  return oldConfig; //not really needed, but make compilers happy
}

void
MpiBackend::runBalancer(std::vector<pair64>&& localConfig,
                    std::vector<pair64>& newLocalConfig,
                    uint64_t localWork, uint64_t globalWork,
                    int maxNumLocalTasks, bool allowGiveTake)
{
  uint64_t perfBalance = globalWork / size_;

  MPI_Comm balanceComm;
  int color = 0;
  MPI_Comm_split(comm_, color, localWork, &balanceComm);
  int balanceRank;
  MPI_Comm_rank(balanceComm, &balanceRank);

  std::vector<pair64> incomingConfig;
  incomingConfig.resize(maxNumLocalTasks);

  int partner;
  /* if an odd number, round up */
  int halfSize = (size_ + 1)  / 2;
  if (perfBalance > localWork){
    //there less work here
    int rankDelta = halfSize - balanceRank;
    partner = halfSize + rankDelta;
  } else {
    int rankDelta = balanceRank - halfSize;
    partner = halfSize - rankDelta;
  }

  if (partner == balanceRank){
    MPI_Comm_free(&balanceComm);
    //oh, this is as good as it gets
    newLocalConfig = std::move(localConfig);
    return;
  }

  int tag = 451;
  MPI_Status stat;
  MPI_Sendrecv(localConfig.data(), localConfig.size()*2, MPI_UINT64_T, partner, tag,
               incomingConfig.data(), incomingConfig.size()*2, MPI_UINT64_T, partner, tag,
               balanceComm, &stat);

  int numIncoming;
  MPI_Get_count(&stat, MPI_UINT64_T, &numIncoming);
  //we maybe posted a recv larger than we need
  //factor of 2 for pair64
  incomingConfig.resize(numIncoming/2);


  uint64_t partnerTotalWork = 0;
  for (auto& pair : incomingConfig) partnerTotalWork += pair.first;

  int numLocalTasks = localConfig.size();
  int numPartnerTasks = numIncoming;

  if (localWork < partnerTotalWork){
    uint64_t desiredDelta = partnerTotalWork - localWork;
    bool exchangeFailed = true;
    //the exchange needs to get this close to be called a "success"
    uint64_t minCloseness = desiredDelta / 10;
    //less work here
    if (numLocalTasks >= numPartnerTasks){
      //this is awkward... I have more (or same) tasks but also less work
      //I guess try to exchange some tasks, but don't make num task mismatch worse
      int smallTaskIdx, bigTaskIdx;
      /*incoming is bigger tasks, local is smaller tasks */
      uint64_t closeness = tradeTasks(desiredDelta, incomingConfig, localConfig,
                                          bigTaskIdx, smallTaskIdx);

      if (closeness <= minCloseness){
        for (auto& smallTaskPair : newLocalConfig){
          if (smallTaskPair.second == smallTaskIdx){
            auto& bigTaskPair = incomingConfig[bigTaskIdx];
            //swap the small task to the big task
            smallTaskPair.second = bigTaskPair.second;
            smallTaskPair.first = bigTaskPair.first;
          }
        }
        exchangeFailed = false; //great!
      }
    }

    if (exchangeFailed && allowGiveTake){
      //I have less work and also fewer tasks, take some tasks
      std::set<int> toTake = takeTasks(desiredDelta, incomingConfig);
      for (int bigTaskIdx : toTake){
        auto& bigTaskPair = incomingConfig[bigTaskIdx];
        localConfig.push_back(bigTaskPair);
      }
    }
  } else if (localWork > partnerTotalWork){
    //more work here
    uint64_t desiredDelta = localWork - partnerTotalWork;
    bool exchangeFailed = true;
    uint64_t minCloseness = desiredDelta / 10;
    if (numPartnerTasks >= numLocalTasks){
      //this is awkward... I have fewer (or same) tasks but also more work
      //I guess try to exchange some tasks, but don't make num task mismatch worse
      int smallTaskIdx, bigTaskIdx;
      /*local is bigger task, incoming is smaller task */
      uint64_t closeness = tradeTasks(desiredDelta, localConfig, incomingConfig,
                                  bigTaskIdx, smallTaskIdx);
      if (closeness <= minCloseness){ //great!
        for (auto& bigTaskPair : localConfig){
          if (bigTaskPair.second == bigTaskIdx){
            auto& smallTaskPair = incomingConfig[smallTaskIdx];
            //swap the big task to the small task
            bigTaskPair.second = smallTaskPair.second;
            bigTaskPair.first = smallTaskPair.first;
          }
        }
        exchangeFailed = false;
      }
    }

    if (exchangeFailed && allowGiveTake){
      //I have more work and also more tasks - give some tasks away
      std::set<int> toGive = takeTasks(desiredDelta, localConfig);
      for (int bigTaskIdx : toGive){
        for (int i=0; i < localConfig.size(); ++i){
          auto& bigTaskPair = localConfig[i];
          if (bigTaskPair.second == bigTaskIdx){
            //this is the task I'm giving away
            int lastIdx = localConfig.size() - 1;
            localConfig[i] = localConfig[lastIdx];
            localConfig.resize(lastIdx);
            break;
          }
        }
      }
    }
  } else {
    //exactly equal - don't do anything
  }

  newLocalConfig = std::move(localConfig);

  //well, I really hope that worked well
  MPI_Comm_free(&balanceComm);
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
MpiBackend::debug(const char *fmt, ...)
{
  va_list args;
  va_start(args, fmt);
  vfprintf(stdout, fmt, args);
  va_end(args);
  fflush(stdout);
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
  int globalIndex = 0;
  for (int i=0; i < size_; ++i){
    int* currentBlock = &allIndices[i*maxNumLocal];
    for (int localIndex=0; localIndex < maxNumLocal && currentBlock[localIndex] != -1;
          ++localIndex, ++globalIndex){
      mapping[globalIndex].rank = i;
      mapping[globalIndex].rankUniqueId = localIndex;
      ++globalIndex;
      ++localIndex;
    }
  }
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

  for (int i=local.size(); i < config.size(); ++i){
    const pair64& pair = config[i];
    local.emplace_back(pair.second);
  }

  for (int i=config.size(); i < local.size(); ++i){
    local.pop_back();
  }

  std::vector<int> localIndices;
  for (LocalIndex& lidx : local){
    localIndices.push_back(lidx.index);
  }

  make_global_mapping_from_local(indices.size(), localIndices, indices);
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


