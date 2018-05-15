
add_library(darma mpi_backend.cc)
target_include_directories(darma PUBLIC
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/mpi_backend>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(darma darma_frontend)

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

