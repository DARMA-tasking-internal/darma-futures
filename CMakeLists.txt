cmake_minimum_required (VERSION 3.9)

# Silence an error about the version option to project
# this will need to be changed if we ever bump the version to 3.0 and put the
# version stuff in the project command itself
#if (NOT CMAKE_VERSION VERSION_LESS 3.0)
#  cmake_policy(SET CMP0048 OLD)
#endif()
#cmake_policy(SET CMP0025 NEW)
#cmake_policy(SET CMP0028 OLD)

project (DarmaConcept)
# The version number.
set (DarmaConcept_VERSION_MAJOR 0)
set (DarmaConcept_VERSION_MINOR 1)


#include (CheckCXXCompilerFlag)
#include (CheckCXXSourceCompiles)

enable_testing()

add_subdirectory(frontend)
add_subdirectory(mpi_backend)
add_subdirectory(examples)
add_subdirectory(tests)


