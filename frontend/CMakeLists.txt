
add_library(darma_frontend INTERFACE)
target_include_directories(darma_frontend INTERFACE
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/frontend>
  $<INSTALL_INTERFACE:include>
)

set(headers
  async_ref.h
  frontend.h
  frontend_task.h
  frontend_task_mpl.h
  frontend_tuple_mpl.h
  mpl.h
  permissions.h
  sequencer.h
)


install(TARGETS darma_frontend EXPORT darma_frontend)
install(EXPORT darma_frontend DESTINATION cmake)
install(FILES ${headers} DESTINATION include)

configure_file(
  ${CMAKE_SOURCE_DIR}/cmake/darma_frontendConfig.cmake.in
  "${PROJECT_BINARY_DIR}/darma_frontendConfig.cmake" @ONLY
)

install(
  FILES
  "${PROJECT_BINARY_DIR}/darma_frontendConfig.cmake" 
  DESTINATION cmake 
)

