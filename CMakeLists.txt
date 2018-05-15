cmake_minimum_required (VERSION 2.8)

# Silence an error about the version option to project
# this will need to be changed if we ever bump the version to 3.0 and put the
# version stuff in the project command itself
if (NOT CMAKE_VERSION VERSION_LESS 3.0)
  cmake_policy(SET CMP0048 OLD)
endif()

project (DarmaConcept)
# The version number.
set (DarmaConcept_VERSION_MAJOR 0)
set (DarmaConcept_VERSION_MINOR 1)

#include (CheckCXXCompilerFlag)
#include (CheckCXXSourceCompiles)

add_subdirectory(frontend)
add_subdirectory(mpi_backend)
add_subdirectory(examples)


