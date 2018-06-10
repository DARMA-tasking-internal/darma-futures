#include "mpi_backend.h"

std::vector<MpiBackend::pair64>
MpiBackend::debugBalance(std::vector<pair64>&& localConfig)
{
  error("Debug load balancer not yet implemented");
  return localConfig;
}
