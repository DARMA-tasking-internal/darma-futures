#include "mpi_backend.h"
#include "mpi_index_entry.h"
#include <cstdlib>
#include <iostream>
#include <cstdarg>
#include <sstream>
#include <CLI/CLI.hpp>

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


  std::string lbType = "commSplit";
  std::vector<std::string> debugs;
  if (be_argc > 0){
    CLI::App app{"DARMA MPI Backend"};
    app.add_option("--lb", lbType, "the load balancer type to use");
    app.add_option("-d,--debug", debugs, "debug flags to activate");
    try {
      app.parse(be_argc, be_argv);
    } catch (const CLI::ParseError &e) {
      int rc = app.exit(e);
      exit(rc);
    }
  }

  for (auto& str : debugs){
    auto name = str_tolower(std::move(str));
    if (name == "lb"){
      DebugFlag<DebugFlags::LB>::active = true;
    } else if (name == "interop"){
      DebugFlag<DebugFlags::Interop>::active = true;
    } else if (name == "sendrecv"){
      DebugFlag<DebugFlags::SendRecv>::active = true;
    } else if (name == "task"){
      DebugFlag<DebugFlags::Task>::active = true;
    } else {
      error("Invalid debug flag %s given", name.c_str());
    }
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

template <int T>
struct print_size;

MpiBackend::~MpiBackend()
{
  clear_tasks();

  //flush any dangling communication
  while(progress_dependencies() || numPendingProbes_);

  for (int i=0; i < requests_.size(); ++i){
    if (requests_[i] != MPI_REQUEST_NULL){
      error("Request %d not complete in backend destructor", i);
    }
    if (listeners_[i] != nullptr){
      error("Listener %d not cleared in backend destructor", i);
    }
  }

  MPI_Type_free(&perfCtrType_);
  MPI_Op_free(&perfCtrOp_);
}

static const uint32_t collIdMask = 0xF << 16;
static const uint32_t dstIdMask = 0x3F << 10;
static const uint32_t srcIdMask = 0x3F << 4;
static const uint32_t taskIdMask = 0x7;

int
MpiBackend::makeUniqueTag(int collId, int dstId, int srcId, int taskId){
  //this is dirty - but don't hate
  int tag = 0;
  tag = tag | uint32_t(collId) << 16;
  tag = tag | uint32_t(dstId) << 10;
  tag = tag | uint32_t(srcId) << 4;
  tag = tag | uint32_t(taskId);
  int maxTag = 1<<21 - 1;
  if (tag > maxTag){
    error("Invalid tag %d from %d-%d-%d-%d", tag, collId, dstId, srcId, taskId);
  }
  return tag;
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
  int numIter = 0;
  while(numPendingProbes_ > 0){
    MPI_Status stat;
    MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, comm_, &stat);

    //this might be an incoming task - or it might just be a message
    PendingRecvBase* recv = nullptr;
    int tag = stat.MPI_TAG;
    int collId = (tag & collIdMask) >> 16;
    int dstId = (tag & dstIdMask) >> 10;
    int srcId = (tag & srcIdMask) >> 4;
    int taskId = (tag & taskIdMask);
    if (taskId != 0){
      //this delivered a task to me
      auto& gen = generators_[taskId];
      recv = gen->generate(frontendPtr(), dstId, collId);
    } else {
      auto& rankMap = pendingRecvs_[stat.MPI_SOURCE];
      auto iter = rankMap.find(stat.MPI_TAG);
      if (iter != rankMap.end()){
        auto& list = iter->second;
        recv = list.front();
        list.pop_front();
        if (list.empty()){
          rankMap.erase(iter);
        }
      }
    }

    int size; MPI_Get_count(&stat, MPI_BYTE, &size);
    void* data = allocate_temp_buffer(size);
    int reqId;
    if (recv){
      //probe we expected
      --numPendingProbes_;
      recv->configure(this, size, data);
      reqId = recv->id();
    } else {
      //this was not a pending pro
      reqId = allocate_request();
      recvsQueued_[stat.MPI_TAG].emplace_back(reqId, size,data);
    }
    MPI_Irecv(data, size, MPI_BYTE, stat.MPI_SOURCE, stat.MPI_TAG, comm_,
              &requests_[reqId]);
  }
  numPendingProbes_ = 0;
}

void
MpiBackend::add_pending_recv(PendingRecvBase* pending, int collId,
                             const IndexInfo& local, const IndexInfo& remote)
{
  int tag = makeUniqueTag(collId, local.rankUniqueId, remote.rankUniqueId);
  darmaDebug(SendRecv, "collection {} made tag={} for receiving elem={},{} from elem={},{}",
             collId, tag, local.rank, local.rankUniqueId, remote.rank, remote.rankUniqueId);
  auto iter = recvsQueued_.find(tag);
  if (iter == recvsQueued_.end()){
    pendingRecvs_[remote.rank][tag].push_back(pending);
    int reqId = allocate_request();
    pending->setId(reqId);
    listeners_[reqId] = pending;
    requests_[reqId] = MPI_REQUEST_NULL;
    ++numPendingProbes_;
  } else {
    auto& list = iter->second;
    PostedRecv& post = list.front();
    Listener* listener = listeners_[post.id];
    if (listener == (Listener*)REQUEST_CLEAR){
      //the recv was posted and already finished
      pending->configure(this, post.size, post.data);
      bool del = pending->finalize();
      if (del){
        delete pending;
      }
      listeners_[post.id] = nullptr;
      freeRequests_.push_back(post.id);
    } else {
      //the recv is posted, but not yet completed
      listeners_[post.id] = pending;
      pending->configure(this, post.size, post.data);
    }
    list.pop_front();
    if (list.empty())
      recvsQueued_.erase(iter);
  }
}

int
MpiBackend::allocate_request()
{
  if (freeRequests_.empty()){
    int ret = requests_.size();
    requests_.push_back(MPI_REQUEST_NULL);
    listeners_.push_back(nullptr);
    statuses_.emplace_back();
    return ret;
  } else {
    int ret = freeRequests_.back();
    freeRequests_.pop_back();
    if (listeners_[ret]){
      error("Allocating free request %d, but listener on index not cleared", ret);
    }
    return ret;
  }
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
      if (del){
        delete listener;
      }
      listeners_[idx] = nullptr;
    }
  } else {
    listeners_[idx] = (Listener*) REQUEST_CLEAR;
  }
}

bool
MpiBackend::progress_dependencies()
{
  create_pending_recvs();

  int nComplete;
  MPI_Testsome(requests_.size(), requests_.data(), &nComplete,
               indices_.data(), statuses_.data());


  if (nComplete == MPI_UNDEFINED){
    return false;
  }

  int freeSize = freeRequests_.size();

  //start from the end for backfilling
  for (int i=0; i < nComplete; ++i){
    int idxDone = indices_[i];
    inform_listener(idxDone);
    freeRequests_.push_back(idxDone);
  }

  int nonNull = 0;
  for (MPI_Request req : requests_){
    if (req != MPI_REQUEST_NULL) ++nonNull;
  }
  if ( (nComplete + freeSize + numPendingProbes_ + nonNull) != requests_.size()){
    error("Sum of individual request types (complete=%d,free=%d,pending=%d,active=%d), do not sum total=%d",
          nComplete, freeSize, numPendingProbes_, nonNull, requests_.size());
  }

  //if all requests are now free requests
  return freeRequests_.size() < requests_.size();
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
      delete t;
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
  void* ptr = new char[size];
  return ptr;
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
  uint64_t numTries = 0;
  while (!taskQueue_.empty()){
    progress_dependencies();
    progress_tasks();
    ++numTries;
    if (numTries > 1e3){
      std::cerr << "Have " << numPendingProbes_
        << " " << requests_.size()
        << " " << freeRequests_.size()
        << std::endl;
      abort();
    }
  }

  while(progress_dependencies());
}

int
MpiBackend::send_data(mpi_async_ref& ref, int collId,
                      const IndexInfo& src, const IndexInfo& dst,
                      void* data, int size, int taskId)
{
  int tag = makeUniqueTag(collId, dst.rankUniqueId, src.rankUniqueId, taskId);
  darmaDebug(SendRecv, "collection {} made tag={} for sending elem={},{} to elem={},{}",
             collId, tag, src.rank, src.rankUniqueId, dst.rank, dst.rankUniqueId);
  int request = allocate_request();
  ref.addRequest(request);
  send_data(dst.rank, data, size, tag, &requests_[request]);
  return request;
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
    //darmaDebug(LB, "Rank {} now has local index {}={}", rank_, i, local[i]);
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
      //darmaDebug(LB, "Rank {} thinks global index {} is now {},{}",
      //           rank_, globalIndex, mapping[globalIndex].rank, mapping[globalIndex].rankUniqueId);
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
  be_->free_temp_buffer(data_, size_);
}

void
PendingRecvBase::configure(MpiBackend* be, int size, void* data)
{
  be_ = static_cast<Frontend<MpiBackend>*>(be);
  size_ = size;
  data_ = data;
}


