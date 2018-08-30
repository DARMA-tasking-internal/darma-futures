DARMA futures library
====================================

The DARMA futures library provides type-safe concurrency for implementing distributed memory tasking and load balancing.
DARMA provides a frontend header-only library. This distribution also includes a default MPI backend implementing
the runtime.

Building only requires an MPI C++ compiler

````
cmake ${srcdir} \
  -DCMAKE_CXX_COMPILER=${mpi_compiler} \
  -DCMAKE_INSTALL_PREFIX=${install}
````

Aftering running `make install`, both the header library and default MPI runtime will be installed.
Applications using DARMA will simply need to load the installed darma CMake package.


## License

[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

## Contact information

Questions? Contact Jeremiah Wilke (jjwilke@sandia.gov)

