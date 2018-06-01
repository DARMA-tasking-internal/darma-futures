#include "mpi_backend.h"

std::vector<MpiBackend::pair64>
MpiBackend::randomBalance(std::vector<pair64>&& localConfig)
{
  error("Random load balancer not yet implemented");
  return localConfig;
}
