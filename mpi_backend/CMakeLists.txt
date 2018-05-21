
find_package(MPI REQUIRED)

add_library(darma mpi_backend.cc gather.cc)
add_library(darma::darma ALIAS darma)
target_include_directories(darma PUBLIC
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/mpi_backend>
  $<INSTALL_INTERFACE:include>
)
find_package(DarmaSerialization)
target_link_libraries(darma PUBLIC darma_serialization::darma_serialization)
target_link_libraries(darma PUBLIC darma_frontend)

target_link_libraries(darma PUBLIC MPI::MPI_CXX)

target_compile_options(darma PUBLIC $<$<CONFIG:DEBUG>:-O0 -ggdb>)

install(TARGETS darma EXPORT darma DESTINATION lib)
install(EXPORT darma DESTINATION cmake)

configure_file(
  ${CMAKE_SOURCE_DIR}/cmake/darma_frontendConfig.cmake.in
  "${PROJECT_BINARY_DIR}/darmaConfig.cmake" @ONLY
)

install(
  FILES
  "${PROJECT_BINARY_DIR}/darmaConfig.cmake" 
  DESTINATION cmake 
)

