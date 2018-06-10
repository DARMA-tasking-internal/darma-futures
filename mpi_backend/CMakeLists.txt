
find_package(MPI REQUIRED)

add_subdirectory(parser)

option(DARMA_PRINT_DEBUG "Whether to active debug printing" 0)
option(DARMA_ZOLTAN_LB "Whether to use Zoltan LB library" 0)

set(SOURCES
 mpi_backend.cc 
 gather.cc
 broadcast.cc
 zoltan_lb.cc
 random_lb.cc
 debug_lb.cc
 comm_split_lb.cc
)

add_library(darma ${SOURCES})
add_library(darma::darma ALIAS darma)


target_include_directories(darma PUBLIC
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/mpi_backend>
  $<INSTALL_INTERFACE:include/darma_backend>
)
target_include_directories(darma PUBLIC
  $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/mpi_backend>
  $<INSTALL_INTERFACE:include/darma_backend>
)


find_package(DarmaSerialization REQUIRED)

target_link_libraries(darma PUBLIC darma_serialization::darma_serialization)
target_link_libraries(darma PUBLIC darma::darma_frontend)
target_link_libraries(darma PUBLIC MPI::MPI_CXX)
target_link_libraries(darma PUBLIC CLI11::CLI11)

if (DARMA_PRINT_DEBUG)
find_package(fmt PATHS ${fmt_DIR} ${fmt_DIR}/lib/cmake)
if(NOT fmt_FOUND)
message(FATAL_ERROR
"debug printing requires the fmt library. Please specify valid installation with "
"-Dfmt_DIR= or install by downloading from https://github.com/fmtlib/fmt"
)
endif()
target_link_libraries(darma PUBLIC fmt::fmt)
set(DARMA_DEBUG_PRINT_OPT 1)
else()
set(DARMA_DEBUG_PRINT_OPT 0)
endif()

if (DARMA_ZOLTAN_LB)
find_package(zoltanLB PATHS ${zoltanLB_DIR} ${zoltanLB_DIR}/cmake)
if(NOT zoltanLB_FOUND)
message(FATAL_ERROR
"Zoltan load balancing requires the zoltanLB library. Please specify valid installation with "
"-DzoltanLB_DIR="
)
endif()
target_link_libraries(darma PUBLIC zoltanLB::zoltanLB)
set(DARMA_ZOLTAN_LB_OPT 1)
else()
set(DARMA_ZOLTAN_LB_OPT 0)
endif()

target_compile_options(darma PUBLIC $<$<CONFIG:DEBUG>:-O0 -ggdb>)

install(TARGETS darma EXPORT darma
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
        INCLUDES DESTINATION include)

install(EXPORT darma
        FILE darmaTargets.cmake
        NAMESPACE darma::
        DESTINATION cmake)

export(TARGETS darma
        NAMESPACE darma::
        FILE darmaTargets.cmake)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/darma_config.h 
        DESTINATION include/darma_backend)

install(DIRECTORY . DESTINATION include/darma_backend
        FILES_MATCHING
        PATTERN "*.h")

include(CMakePackageConfigHelpers)
configure_package_config_file(cmake/darmaConfig.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/darmaConfig.cmake
        INSTALL_DESTINATION cmake)
write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/darmaConfigVersion.cmake
        VERSION "${DarmaConcept_VERSION_MAJOR}.${DarmaConcept_VERSION_MINOR}"
        COMPATIBILITY SameMajorVersion)

install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/darmaConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/darmaConfigVersion.cmake
  DESTINATION cmake
)

configure_file(
 ${CMAKE_CURRENT_SOURCE_DIR}/darma_config.h.in
 ${CMAKE_CURRENT_BINARY_DIR}/darma_config.h @ONLY
)

