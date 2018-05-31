
find_package(MPI REQUIRED)

add_library(darma mpi_backend.cc gather.cc broadcast.cc)
add_library(darma::darma ALIAS darma)

target_include_directories(darma PUBLIC
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/mpi_backend>
  $<INSTALL_INTERFACE:include/darma_backend>
)

find_package(DarmaSerialization REQUIRED)

target_link_libraries(darma PUBLIC darma_serialization::darma_serialization)
target_link_libraries(darma PUBLIC darma::darma_frontend)

target_link_libraries(darma PUBLIC MPI::MPI_CXX)

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

