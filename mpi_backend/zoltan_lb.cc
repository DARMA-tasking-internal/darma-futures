#include "mpi_backend.h"
#if DARMA_ZOLTAN_LB
#include <z2LoadBalancer.hpp>
#endif

std::vector<MpiBackend::pair64>
MpiBackend::zoltanBalance(std::vector<pair64>&& localConfig)
{
  z2lb::localId_t nLocal = localConfig.size();
  std::vector<z2lb::globalId_t> globalIds(nLocal);
  std::vector<z2lb::scalar_t> weights(nLocal);
  std::vector<const z2lb::scalar_t*> zWeights(1);
  std::vector<int> weightStrides(1,1);
  for (int i=0; i < nLocal; ++i){
    auto& pair = localConfig[i];
    weights[i] = pair.first;
    globalIds[i] = pair.second;
  }
  zWeights[0] = weights.data();

  std::vector<z2lb::globalId_t> newGlobalIds;
  z2lb::z2LoadBalancer(nLocal, globalIds.data(), zWeights,
                 weightStrides, newGlobalIds);

  std::vector<pair64> newConfig(newGlobalIds.size());
  std::stringstream sstr;
  sstr << "{";
  for (int i=0; i < newGlobalIds.size(); ++i){
    uint64_t id = newGlobalIds[i];
    auto& pair = newConfig[i];
    pair.first = 0; //ignore weight
    pair.second = id;
    sstr << " "  << id;
  }
  sstr << " }";
  darmaDebug("Rank {} now has IDs {} after Zoltan LB", rank_, sstr.str());
  return newConfig;
}
