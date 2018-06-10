#include "mpi_backend.h"

std::vector<MpiBackend::pair64>
MpiBackend::zoltanBalance(std::vector<pair64>&& localConfig)
{
  error("Zoltan load balancer not yet implemented");
  return localConfig;
}
