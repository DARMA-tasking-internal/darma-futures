
add_library(darma mpi_backend.cc)
target_include_directories(darma PUBLIC
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/mpi_backend>
  $<INSTALL_INTERFACE:include>
)
find_package(DarmaSerialization)
target_link_libraries(darma darma_serialization::darma_serialization)
target_link_libraries(darma darma_frontend)

set(HEADERS
backend_fwd.h
darma.h
mpi_async_ref.h
mpi_backend.h
mpi_collection.h
mpi_index_entry.h
mpi_listener.h
mpi_pending_recv.h
mpi_phase.h
mpi_predicate.h
mpi_send_recv.h
mpi_task.h
)

install(TARGETS darma EXPORT darma DESTINATION lib)
install(EXPORT darma DESTINATION cmake)
install(FILES ${HEADERS} DESTINATION include)

configure_file(
  ${CMAKE_SOURCE_DIR}/cmake/darmaConfig.cmake.in
  "${PROJECT_BINARY_DIR}/darmaConfig.cmake" @ONLY
)

install(
  FILES
  "${PROJECT_BINARY_DIR}/darmaConfig.cmake" 
  DESTINATION cmake 
)

