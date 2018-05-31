
add_library(darma_frontend INTERFACE)
add_Library(darma::darma_frontend ALIAS darma_frontend)

target_include_directories(darma_frontend INTERFACE
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/frontend>
  $<INSTALL_INTERFACE:include/darma>
)

target_compile_features(darma_frontend INTERFACE cxx_std_14)

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


install(TARGETS darma_frontend EXPORT darma_frontend
        INCLUDES DESTINATION include)

install(EXPORT darma_frontend
        FILE darma_frontendTargets.cmake
        NAMESPACE darma::
        DESTINATION cmake)

export(TARGETS darma_frontend
       NAMESPACE darma::
       FILE darma_frontendTargets.cmake)

install(DIRECTORY . DESTINATION include/darma FILES_MATCHING PATTERN "*.h")

include(CMakePackageConfigHelpers)
configure_package_config_file(cmake/darma_frontendConfig.cmake.in
                              ${CMAKE_CURRENT_BINARY_DIR}/darma_frontendConfig.cmake 
                              INSTALL_DESTINATION cmake)
write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/darma_frontendConfigVersion.cmake
                                 VERSION "${DarmaConcept_VERSION_MAJOR}.${DarmaConcept_VERSION_MINOR}"
                                 COMPATIBILITY SameMajorVersion)

install(
  FILES 
        ${CMAKE_CURRENT_BINARY_DIR}/darma_frontendConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/darma_frontendConfigVersion.cmake
  DESTINATION cmake 
)

