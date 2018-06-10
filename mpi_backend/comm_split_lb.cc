#include "mpi_backend.h"

std::vector<MpiBackend::pair64>
MpiBackend::commSplitBalance(std::vector<pair64>&& localConfig)
{
  static const int maxNumTries = 5;
  static const double diffCutoff = 0.15;
  int tryNum = 0;
  double maxDiffFraction;

  std::vector<pair64> oldConfig = std::move(localConfig);

  uint64_t lastImbalance = 0;

  bool allowTrades = true;
  bool allowGiveTake = false;
  while(1) {
    if (tryNum >= maxNumTries){
      return oldConfig;
    }

    uint64_t localWork = 0;
    for (int i=0; i < oldConfig.size(); ++i){
      auto& pair = oldConfig[i];
      uint64_t weight = pair.first;
      localWork += weight;
    }

    darmaDebug("Rank={} has total {} from {} tasks", rank_, localWork, oldConfig.size());

    PerfCtrReduce local;
    local.min = localWork;
    local.max = localWork;
    local.total = localWork;
    local.maxLocalTasks = oldConfig.size();
    PerfCtrReduce global;

    MPI_Allreduce(&local, &global, 1, perfCtrType_, perfCtrOp_, comm_);

    uint64_t perfBalance = global.total / size_;

    if (rank_ == 0){
      darmaDebug("Try {} has global={} with maxTasks={} with minWork={} and maxWork={} and balanced={}",
            tryNum, global.total, global.maxLocalTasks, global.min, global.max, perfBalance);
    }

    uint64_t newImbalance = std::max(perfBalance - global.min, global.max - perfBalance);
    if (newImbalance == lastImbalance){
      if (allowGiveTake){
        //we fell into a floyd hole - stop trying
        return oldConfig;
      }
      //start allowing give/take
      allowGiveTake = true;
    }
    lastImbalance = newImbalance;


    uint64_t maxDiff = global.max - global.min;
    maxDiffFraction = double(maxDiff) / double(perfBalance);
    if (maxDiffFraction < diffCutoff){
      return oldConfig;
    }

    allowGiveTake = allowGiveTake || tryNum >= 2;
    //sort the old config by task weight
    std::sort(oldConfig.begin(), oldConfig.end(), sortByWeight());
    std::vector<pair64> newConfig;
    runCommSplitBalancer(std::move(oldConfig), newConfig,
        localWork, global.total, global.maxLocalTasks,
         allowTrades, allowGiveTake);

    ++tryNum;

    oldConfig = std::move(newConfig);
  }
  return oldConfig; //not really needed, but make compilers happy
}

void
MpiBackend::runCommSplitBalancer(std::vector<pair64>&& localConfig,
                    std::vector<pair64>& newLocalConfig,
                    uint64_t localWork, uint64_t globalWork,
                    int maxNumLocalTasks,
                    bool allowTrades,
                    bool allowGiveTake)
{
  uint64_t perfBalance = globalWork / size_;

  MPI_Comm balanceComm;
  int color = 0;
  int key = localWork/1000;
  MPI_Comm_split(comm_, color, key, &balanceComm);
  int balanceRank;
  MPI_Comm_rank(balanceComm, &balanceRank);

  std::vector<pair64> incomingConfig;
  incomingConfig.resize(maxNumLocalTasks);

  int partner;
  /* if an odd number, round up */

  if (perfBalance > localWork){
    //there is less work here
    if (size_%2){
      //odd number of ranks
      int halfSize = size_  / 2;
      int rankDelta = halfSize - balanceRank;
      partner = halfSize + rankDelta;
    } else {
      //even number of ranks
      int halfSize = size_ / 2;
      int rankDelta = halfSize - balanceRank;
      partner = halfSize + (rankDelta-1);
    }
  } else {
    if (size_%2){
      //odd number of ranks
      int halfSize = size_  / 2;
      int rankDelta = balanceRank - halfSize;
      partner = halfSize - rankDelta;
    } else {
      //even number of ranks
      int halfSize = size_ / 2;
      int rankDelta =  balanceRank - halfSize + 1;
      partner = halfSize - rankDelta;
    }

  }


  darmaDebug("Rank={} has localWork={} compared to balanced={} with key={} became rank={} with partner={}",
        rank_, localWork, perfBalance, key, balanceRank, partner);

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
  numIncoming /= 2;
  incomingConfig.resize(numIncoming);

  uint64_t partnerTotalWork = 0;
  for (auto& pair : incomingConfig) partnerTotalWork += pair.first;

  int numLocalTasks = localConfig.size();
  int numPartnerTasks = numIncoming;

  if (localWork < partnerTotalWork){
    // a bit tricky - the change in task sizes should be 1/2 the difference
    uint64_t desiredDelta = (partnerTotalWork - localWork) / 2;
    bool exchangeFailed = true;
    //the exchange needs to get this close to be called a "success"
    uint64_t minCloseness = desiredDelta / 10;
    uint64_t minExchangeCloseness = 2*desiredDelta/3;
    //less work here
    if (numLocalTasks >= numPartnerTasks && allowTrades){
      //this is awkward... I have more (or same) tasks but also less work
      //I guess try to exchange some tasks, but don't make num task mismatch worse
      int smallTaskIdx, bigTaskIdx;
      /*incoming is bigger tasks, local is smaller tasks */
      uint64_t closeness = tradeTasks(desiredDelta, incomingConfig, localConfig,
                                          bigTaskIdx, smallTaskIdx);

      if (closeness < minExchangeCloseness){ //only trade if it actually make solution better
        auto& bigTaskPair = incomingConfig[bigTaskIdx];
        auto& smallTaskPair = localConfig[smallTaskIdx];
        darmaDebug("Rank={}:{} would like to trade small={},{},{} for big={},{},{} for closeness={} to delta={}",
              balanceRank, rank_,
              smallTaskIdx, smallTaskPair.first, smallTaskPair.second,
              bigTaskIdx, bigTaskPair.first, bigTaskPair.second,
              closeness, desiredDelta);
        smallTaskPair.second = bigTaskPair.second;
        smallTaskPair.first = bigTaskPair.first;
      }
      exchangeFailed = closeness > minCloseness; //great!
    }

    if (exchangeFailed && allowGiveTake){
      //I have less work and also fewer tasks, take some tasks
      std::set<int> toTake = takeTasks(desiredDelta, incomingConfig);
      uint64_t totalDelta = 0;
      for (int bigTaskIdx : toTake){
        auto& bigTaskPair = incomingConfig[bigTaskIdx];
        totalDelta += bigTaskPair.first;
        darmaDebug("Rank={}:{} would like to take {},{},{} for total={} delta={}",
              balanceRank, rank_, bigTaskIdx, bigTaskPair.first, bigTaskPair.second,
              totalDelta, desiredDelta);
        localConfig.push_back(bigTaskPair);
      }
    }
  } else if (localWork > partnerTotalWork){
    //more work here
    uint64_t desiredDelta = (localWork - partnerTotalWork) / 2;
    bool exchangeFailed = true;
    uint64_t minCloseness = desiredDelta / 10;
    uint64_t minExchangeCloseness = 2*desiredDelta/3;
    if (numPartnerTasks >= numLocalTasks && allowTrades){
      //this is awkward... I have fewer (or same) tasks but also more work
      //I guess try to exchange some tasks, but don't make num task mismatch worse
      int smallTaskIdx, bigTaskIdx;
      /*local is bigger task, incoming is smaller task */
      uint64_t closeness = tradeTasks(desiredDelta, localConfig, incomingConfig,
                                  bigTaskIdx, smallTaskIdx);

      if (closeness < minExchangeCloseness){ //only trade if it actually make solution better
        auto& bigTaskPair = localConfig[bigTaskIdx];
        auto& smallTaskPair = incomingConfig[smallTaskIdx];
        darmaDebug("Rank={} would like to trade big={},{},{} for small={},{},{} for closeness={} to delta={}",
              balanceRank, rank_,
              bigTaskIdx, bigTaskPair.first, bigTaskPair.second,
              smallTaskIdx, smallTaskPair.first, smallTaskPair.second,
              closeness, desiredDelta);
        bigTaskPair.second = smallTaskPair.second;
        bigTaskPair.first = smallTaskPair.first;
      }
      exchangeFailed = closeness > minCloseness;
    }

    if (exchangeFailed && allowGiveTake){
      //I have more work and also more tasks - give some tasks away
      std::set<int> toGive = takeTasks(desiredDelta, localConfig);
      uint64_t totalDelta = 0;
      for (int bigTaskIdx : toGive){
        auto& bigTaskPair = localConfig[bigTaskIdx];
        totalDelta += bigTaskPair.first;
        darmaDebug("Rank={}:{} would like to give {},{},{} for total={} delta={}",
              balanceRank, rank_, bigTaskIdx, bigTaskPair.first, bigTaskPair.second,
              totalDelta, desiredDelta);
        //this is the task I'm giving away
        int lastIdx = localConfig.size() - 1;
        localConfig[bigTaskIdx] = std::move(localConfig[lastIdx]);
        localConfig.pop_back();
      }
    }
  } else {
    //exactly equal - don't do anything
  }

  newLocalConfig = std::move(localConfig);

  //well, I really hope that worked well
  MPI_Comm_free(&balanceComm);
}
