#include "mpi_backend.h"

std::vector<MpiBackend::pair64>
MpiBackend::debugBalance(std::vector<pair64>&& localConfig)
{
  int partner = getTradingPartner(rank_);
  if (partner == rank_) return localConfig;


  int toSend = localConfig.empty() ? -1 : localConfig.back().second;
  int toRecv;
  int tag = 278;
  MPI_Sendrecv(&toSend, 1, MPI_INT, partner, tag,
               &toRecv, 1, MPI_INT, partner, tag,
               comm_, MPI_STATUS_IGNORE);

  if (localConfig.empty()){
    if (toRecv != -1){
      localConfig.emplace_back(0,toRecv);
    }
  } else {
    if (toRecv != -1){
      auto& pair = localConfig.back();
      pair.first = 0;
      pair.second = toRecv;
    }
  }
  darmaDebug("Rank {} trading tasks {},{} with rank {}",
             rank_, toSend, toRecv, partner);
  return localConfig;
}
