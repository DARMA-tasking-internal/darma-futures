#include "mpi_backend.h"
#include <algorithm>
#include <sstream>

std::vector<MpiBackend::pair64>
MpiBackend::commSplitBalance(std::vector<pair64>&& localConfig)
{
  static const int maxNumTries = 5;
  static const double diffCutoff = 0.15;
  int tryNum = 0;
  double maxDiffFraction;

  double t_start = get_time();
  MPI_Barrier(comm_); //bad, but for timers
  double t_stop = get_time();
  double t_ms = (t_stop - t_start)*1e3;
  if (rank_ == 0){
    std::cout << "Load balance synchronization delay took " << t_ms << "ms" << std::endl;
  }

  t_start = get_time();

  std::vector<pair64> oldConfig = std::move(localConfig);

  uint64_t lastImbalance = std::numeric_limits<uint64_t>::max();

  bool allowTrades = true;
  bool allowGiveTake = false;
  while(1) {
    if (tryNum >= maxNumTries){
      break;
    }

    uint64_t localWork = 0;
    uint64_t maxTask = 0;
    uint64_t minTask = std::numeric_limits<uint64_t>::max();
    for (int i=0; i < oldConfig.size(); ++i){
      auto& pair = oldConfig[i];
      uint64_t weight = pair.first;
      maxTask = std::max(weight, maxTask);
      minTask = std::min(weight, minTask);
      localWork += weight;
    }

    darmaDebug(LB, "Rank={} has total {} from {} tasks", rank_, localWork, oldConfig.size());

    PerfCtrReduce local;
    local.min = localWork;
    local.max = localWork;
    local.total = localWork;
    local.minTask = minTask;
    local.maxTask = maxTask;
    local.maxLocalTasks = oldConfig.size();
    PerfCtrReduce global;

    MPI_Allreduce(&local, &global, 1, perfCtrType_, perfCtrOp_, comm_);

    uint64_t perfBalance = global.total / size_;

    if (rank_ == 0){
      double max = double(global.max) / double(global.min);
      double avg = double(global.total) / size_ / double(global.min);
      double maxTask = double(global.maxTask) / double(global.min);
      double minTask = double(global.minTask) / double(global.min);
      std::cout << "Try " << tryNum << " balancing range 1.0--"  
                << avg << "--" << max << " (" << maxTask << "," << minTask << ")" 
                << "        "
                << global.min << "--" << (global.total/size_) 
                << "--" << global.max << " (" << global.maxTask << "," << minTask << ")"
                << std::endl;
      darmaDebug(LB, "Try {} has global={} with maxTasks={} with minWork={} and maxWork={} and balanced={}",
            tryNum, global.total, global.maxLocalTasks, global.min, global.max, perfBalance);
    }

    uint64_t newImbalance = global.max - perfBalance;
    double improvement = double(lastImbalance) / double(newImbalance);
    if (improvement < 1.05){
      break;
    }
    lastImbalance = newImbalance;

    double imbalanceRatio = double(global.max) / double(perfBalance);
    if (imbalanceRatio < 1.1){
      break;
    }

    uint64_t maxDiff = global.max - global.min;
    maxDiffFraction = double(maxDiff) / double(perfBalance);
    if (maxDiffFraction < diffCutoff){
      return oldConfig;
    }

    allowGiveTake = allowGiveTake || tryNum >= 4;
    //sort the old config by task weight
    for (auto& pair : oldConfig){
      if (pair.first == 0) error("Rank %d has zero weight before sort", rank_);
    }
    std::sort(oldConfig.begin(), oldConfig.end(), sortByWeight());
    for (auto& pair : oldConfig){
      if (pair.first == 0) error("Rank %d has zero weight after sort", rank_);
    }
    std::vector<pair64> newConfig;

    std::stringstream sstr;
    sstr << "Rank " << rank_ << " try " << tryNum << " [";
    for (auto& pair : oldConfig){
      sstr << " " << pair.second;
    }
    sstr << " ] -> [";
    
    runCommSplitBalancer(std::move(oldConfig), newConfig,
        localWork, global.total, global.maxLocalTasks,
         allowTrades, allowGiveTake);

    for (auto& pair : newConfig){
      sstr << " " << pair.second;
    }
    sstr << " ]";
    //std::cout << sstr.str() << std::endl;

    ++tryNum;

    oldConfig = std::move(newConfig);
  }

  if (rank_ == 0){
    double t_stop = get_time();
    double t_ms = (t_stop - t_start)*1e3;
    std::cout << "Load balance compute took " << t_ms << "ms" << std::endl;
  }
  return oldConfig; //not really needed, but make compilers happy
}

int
MpiBackend::getTradingPartner(int rank) const
{
  int halfSize = size_  / 2;
  int partner;
  if (rank < halfSize){
    //there is less work here
    if (size_%2){
      //odd number of ranks
      int rankDelta = halfSize - rank;
      partner = halfSize + rankDelta;
    } else {
      //even number of ranks
      int rankDelta = halfSize - rank;
      partner = halfSize + (rankDelta-1);
    }
  } else {
    if (size_%2){
      //odd number of ranks
      int rankDelta = rank - halfSize;
      partner = halfSize - rankDelta;
    } else {
      //even number of ranks
      int rankDelta =  rank - halfSize + 1;
      partner = halfSize - rankDelta;
    }
  }
  return partner;
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

  allowGiveTake = false;

  MPI_Comm balanceComm;
  int color = 0;
  int key = localWork/1000;
  MPI_Comm_split(comm_, color, key, &balanceComm);
  int balanceRank;
  MPI_Comm_rank(balanceComm, &balanceRank);

  std::vector<pair64> incomingConfig;
  incomingConfig.resize(maxNumLocalTasks);

  int partner = getTradingPartner(balanceRank);
  /* if an odd number, round up */

  MPI_Group splitGrp; MPI_Comm_group(balanceComm, &splitGrp);
  MPI_Group worldGrp; MPI_Comm_group(MPI_COMM_WORLD, &worldGrp);

  int partnerWorldRank;
  MPI_Group_translate_ranks(splitGrp, 1, &partner, worldGrp, &partnerWorldRank);


  if (partner == balanceRank){
    MPI_Comm_free(&balanceComm);
    //oh, this is as good as it gets
    newLocalConfig = std::move(localConfig);
    return;
  }

  int tag = 451;
  MPI_Status stat;
  MPI_Sendrecv(localConfig.data(), localConfig.size()*2, MPI_UINT64_T, partner, tag,
               incomingConfig.data(), maxNumLocalTasks*2, MPI_UINT64_T, partner, tag,
               balanceComm, &stat);

  for (auto& pair : localConfig){
    if (pair.first == 0) error("Rank %d has zero weight after sendrecv", rank_);
  }

  int numIncoming;
  MPI_Get_count(&stat, MPI_UINT64_T, &numIncoming);
  //we maybe posted a recv larger than we need
  //factor of 2 for pair64
  numIncoming /= 2;
  incomingConfig.resize(numIncoming);

  darmaDebug(LB, "Rank {}={} has localWork={} balanced={} partner Rank {}={} min={} max={} sent={} recvd={} maxTasks={}",
        rank_, balanceRank, localWork, perfBalance, partnerWorldRank, partner,
        localConfig.front().first, localConfig.back().first,
        localConfig.size(), incomingConfig.size(), maxNumLocalTasks);

  uint64_t partnerTotalWork = 0;
  for (auto& pair : incomingConfig) partnerTotalWork += pair.first;

  int numLocalTasks = localConfig.size();
  int numPartnerTasks = numIncoming;

  int maxTrades = maxNumLocalTasks;
  bool exchangeFailed = true;
  uint64_t closeness, minCloseness, totalDelta;

  bool nothingToGain =    (localWork > perfBalance && partnerTotalWork > perfBalance)
                       || (localWork <= perfBalance && partnerTotalWork <= perfBalance);

  if (nothingToGain){
    newLocalConfig = std::move(localConfig);
    //well, I really hope that worked well
    MPI_Comm_free(&balanceComm);
    return;
  }


  int maxTake, maxGive;
  if (localWork < partnerTotalWork){
    // a bit tricky - the change in task sizes should be 1/2 the difference
    //uint64_t desiredDelta = (perfBalance - localWork);
    uint64_t desiredDelta = (partnerTotalWork - localWork) / 2;
    //the exchange needs to get this close to be called a "success"
    minCloseness = desiredDelta / 10;
    //less work here
    if (numLocalTasks >= numPartnerTasks && allowTrades){
      //this is awkward... I have more (or same) tasks but also less work
      //I guess try to exchange some tasks, but don't make num task mismatch worse
      std::vector<int> smallTaskIdx;
      std::vector<int> bigTaskIdx;
      //int smallTaskIdx, bigTaskIdx;
      /*incoming is bigger tasks, local is smaller tasks */
      uint64_t totalDelta = tradeTasks(desiredDelta, incomingConfig, localConfig,
                             bigTaskIdx, smallTaskIdx, maxTrades);

      closeness = totalDelta > desiredDelta ? totalDelta - desiredDelta : desiredDelta - totalDelta;
      int numTrades = smallTaskIdx.size();
      uint64_t testDelta = 0;
      for (int i=0; i < numTrades; ++i){
        auto& bigTaskPair = incomingConfig[bigTaskIdx[i]];
        auto& smallTaskPair = localConfig[smallTaskIdx[i]];
        testDelta += (bigTaskPair.first - smallTaskPair.first);
        darmaDebug(LB, "Rank {}={} would like to trade small={},{},{} for big={},{},{} for brings up delta={} to desired={}",
              rank_, balanceRank, 
              smallTaskIdx[i], smallTaskPair.first, smallTaskPair.second,
              bigTaskIdx[i], bigTaskPair.first, bigTaskPair.second,
              testDelta, desiredDelta);
        std::swap(bigTaskPair.second, smallTaskPair.second);
        std::swap(bigTaskPair.first, smallTaskPair.first);
      }
      darmaDebug(LB, "Rank {}={} has total delta={} for desired={} with closeness={} and minCloseness={}",
                 rank_, balanceRank, totalDelta, desiredDelta, closeness, minCloseness);
      exchangeFailed = closeness > minCloseness; 
    }
  } else if (localWork > partnerTotalWork){
    //more work here
    uint64_t desiredDelta = (localWork - partnerTotalWork) / 2;
    //uint64_t desiredDelta = (perfBalance - partnerTotalWork);

    minCloseness = desiredDelta / 10;
    if (numPartnerTasks >= numLocalTasks && allowTrades){
      //this is awkward... I have fewer (or same) tasks but also more work
      //I guess try to exchange some tasks, but don't make num task mismatch worse
      /*local is bigger task, incoming is smaller task */
      std::vector<int> smallTaskIdx;
      std::vector<int> bigTaskIdx;
      uint64_t totalDelta = tradeTasks(desiredDelta, localConfig, incomingConfig,
                                       bigTaskIdx, smallTaskIdx, maxTrades);

      closeness = totalDelta > desiredDelta ? totalDelta - desiredDelta : desiredDelta - totalDelta;
      int numTrades = smallTaskIdx.size();
      uint64_t testDelta = 0;
      for (int i=0; i < numTrades; ++i){
        auto& bigTaskPair = localConfig[bigTaskIdx[i]];
        auto& smallTaskPair = incomingConfig[smallTaskIdx[i]];
        testDelta += (bigTaskPair.first - smallTaskPair.first);
        darmaDebug(LB, "Rank {}={} would like to trade big={},{},{} for small={},{},{} brings to delta={} for desired={}",
              rank_, balanceRank, 
              bigTaskIdx[i], bigTaskPair.first, bigTaskPair.second,
              smallTaskIdx[i], smallTaskPair.first, smallTaskPair.second,
              testDelta, desiredDelta);
        std::swap(bigTaskPair.second, smallTaskPair.second);
        std::swap(bigTaskPair.first, smallTaskPair.first);
      }
      darmaDebug(LB, "Rank {}={} has total delta={} for desired={} with closeness={} and minCloseness={}",
                 rank_, balanceRank, totalDelta, desiredDelta, closeness, minCloseness);
      exchangeFailed = closeness > minCloseness;
    }
  } else {
    //exactly equal - don't do anything
  }
  
  //recompute where we got
  partnerTotalWork = 0;
  for (auto& pair : incomingConfig) partnerTotalWork += pair.first;
  std::sort(incomingConfig.begin(), incomingConfig.end(), sortByWeight());

  localWork = 0;
  for (auto& pair : localConfig) localWork += pair.first;
  std::sort(localConfig.begin(), localConfig.end(), sortByWeight());

  if (exchangeFailed && allowGiveTake){
    if (localWork < partnerTotalWork){
      maxTake = maxGive = (numPartnerTasks - numLocalTasks) / 2 + 2;
      uint64_t desiredDelta = (partnerTotalWork - localWork) / 2;
      //uint64_t desiredDelta = (perfBalance - localWork);
      //I have less work and also fewer tasks, take some tasks
      std::set<int> toTake = takeTasks(desiredDelta, incomingConfig, maxTake);
      uint64_t totalDelta = 0;
      darmaDebug(LB, "Rank {}={} took {} tasks from partner", 
                 rank_, balanceRank, toTake.size());
      for (int bigTaskIdx : toTake){
        auto& bigTaskPair = incomingConfig[bigTaskIdx];
        totalDelta += bigTaskPair.first;
        darmaDebug(LB, "Rank {}={} would like to take {},{},{} for total={} delta={}",
              rank_, balanceRank, bigTaskIdx, bigTaskPair.first, bigTaskPair.second,
              totalDelta, desiredDelta);
        localConfig.push_back(bigTaskPair);
      }
    } else if (localWork > partnerTotalWork){
      maxTake = maxGive = (numLocalTasks - numPartnerTasks) / 2 + 2;
      uint64_t desiredDelta = (localWork - partnerTotalWork) / 2;
      //uint64_t desiredDelta = (perfBalance - partnerTotalWork);
      //I have more work and also more tasks - give some tasks away
      std::set<int> toGive = takeTasks(desiredDelta, localConfig, maxGive);
      uint64_t totalDelta = 0;
      darmaDebug(LB, "Rank {}={} giving {} tasks to partner", 
                 rank_, balanceRank, toGive.size());
      std::list<int> sortedGive;
      for (int bigTaskIdx : toGive){
        sortedGive.push_front(bigTaskIdx);
      }
      for (int bigTaskIdx : sortedGive){
        auto& bigTaskPair = localConfig[bigTaskIdx];
        totalDelta += bigTaskPair.first;
        darmaDebug(LB, "Rank {}={} would like to give {},{},{} for total={} delta={}",
              rank_, balanceRank, bigTaskIdx, bigTaskPair.first, bigTaskPair.second,
              totalDelta, desiredDelta);
        //this is the task I'm giving away
        int lastIdx = localConfig.size() - 1;
        localConfig[bigTaskIdx] = std::move(localConfig[lastIdx]);
        localConfig.pop_back();
      }
    } else {
      //huh - not sure how exchange failed but partners are exactly equal
    }
  } else {
    darmaDebug(LB, "Rank {}={} passed closeness test {} < {}",
               rank_, balanceRank, closeness, minCloseness);
  }

  newLocalConfig = std::move(localConfig);

  //well, I really hope that worked well
  MPI_Comm_free(&balanceComm);
}

std::set<int>
MpiBackend::takeTasks(uint64_t desiredDelta, const std::vector<pair64>& giver, int maxTake)
{
  std::set<int> toRet;
  uint64_t deltaCutoff = desiredDelta / 10;
  uint64_t maxGiveAway = desiredDelta + deltaCutoff;
  uint64_t totalGiven = 0;
  uint64_t remainingDelta = maxGiveAway;
  int numTaken = 0;
  for (int i=giver.size() - 1; i >= 0 && numTaken < maxTake; --i){
    uint64_t taskSize = giver[i].first;
    //std::cout << "Rank " << rank_ << "="
    //         << " considering give/take of size=" << taskSize
    //         << " for index " << giver[i].second
    //         << " with maxGive=" << maxGiveAway
    //         << " remainining=" << remainingDelta
    //         << " taskSize=" << taskSize
    //         << std::endl;
    if (taskSize < remainingDelta){
      toRet.insert(i); //give it away, give it away, give it away now
      totalGiven += taskSize;
      remainingDelta -= taskSize;
      ++numTaken;
    }
  }
  return toRet;
}

uint64_t
MpiBackend::tradeTasks(uint64_t desiredDelta,
                       uint64_t maxOverage,
                       const std::vector<pair64>& bigger,
                       const std::vector<pair64>& smaller,
                       int bigTaskIdx,
                       int smallTaskIdx)
{
  if (bigTaskIdx >= bigger.size() || smallTaskIdx >= smaller.size()){
    error("Rank %d out of bounds on tradeTasks", rank_);
  }
  //I assume the smaller, bigger are sorted least to greatest coming in
  uint64_t smallTaskSize = smaller[smallTaskIdx].first;
  uint64_t bigTaskSize = bigger[bigTaskIdx].first;
  uint64_t delta = bigTaskSize - smallTaskSize;
  //std::cout << "Rank " << rank_ << "= considering trade "
  //  << bigTaskIdx << "," << bigTaskSize << " for " 
  //  << smallTaskIdx << "," << smallTaskSize << std::endl;
  if (desiredDelta > delta){
    //this is moving us in the right direction - add it
    return delta;
  } else {
    uint64_t delta_delta = delta - desiredDelta;
    //this is too far - but maybe close enough that it's okay
    if (delta_delta < maxOverage){
      return delta; 
    } 
    return 0;
  }
}

uint64_t
MpiBackend::tradeTasks(uint64_t desiredDelta,
                       const std::vector<pair64>& bigger,
                       const std::vector<pair64>& smaller,
                       std::vector<int>& bigTaskIdxs,
                       std::vector<int>& smallTaskIdxs,
                       int maxTrades)
{
  uint64_t totalDelta = 0;
  int numTrades = 0;
  int smallTaskIdx = 0;
  int bigTaskIdx = bigger.size() - 1;
  uint64_t maxOverage = desiredDelta / 8;
  int smallTaskStop = smaller.size() - 1;
  while (numTrades < maxTrades && totalDelta < desiredDelta
         && smallTaskIdx <= smallTaskStop && bigTaskIdx >= 0)
  {
    uint64_t nextDelta = tradeTasks(desiredDelta-totalDelta, maxOverage, 
      bigger, smaller, bigTaskIdx, smallTaskIdx);

    if (nextDelta > 0){
      bigTaskIdxs.push_back(bigTaskIdx);
      smallTaskIdxs.push_back(smallTaskIdx);
      totalDelta += nextDelta;
      ++smallTaskIdx;
      --bigTaskIdx;
      ++numTrades;
    } else {
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
  return totalDelta;
}

