find_package(GTest)

if (GTEST_FOUND)

  include(GoogleTest)

  add_executable(darma_mpi_backend_tests
                 mpi_test_main.cc
                 mpi_gather_test.cc
                 mpi_broadcast_test.cc
                 )
  target_link_libraries(darma_mpi_backend_tests GTest::GTest)
  target_link_libraries(darma_mpi_backend_tests darma::darma)

  gtest_add_tests(TARGET darma_mpi_backend_tests)
endif()