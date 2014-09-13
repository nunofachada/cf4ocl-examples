### Summary

This repository contains examples and experiments using the [cf4ocl][]
library for OpenCL.

### Examples

| Example      | Specific dependencies | Description                                                 |
| ------------ | --------------------- | ----------------------------------------------------------- |
| matmult      | [OpenMP][]            | Comparison of matrix multiplication with OpenCL and OpenMP  |
| bankconf     | None                  | Example of GPU bank conflicts                               |


### Global dependencies

* [GLib][]
* [cf4ocl][]
* [OpenCL][] ([ocl-impl][any implementation])

### Build tools

* [CMake][]
* A C99 compiler

### License

These examples are licensed under [GPLv3][].

[GLib]: https://developer.gnome.org/glib/ "GLib"
[OpenCL]: http://www.khronos.org/opencl/ "OpenCL"
[ocl-impl]: https://github.com/FakenMC/cf4ocl/wiki/OpenCL-implementations
[GPLv3]: http://www.gnu.org/licenses/gpl.html "GPLv3"
[CMake]: http://www.cmake.org/
[cf4ocl]: https://github.com/FakenMC/cf4ocl "cf4ocl"
[OpenMP]: http://openmp.org/

