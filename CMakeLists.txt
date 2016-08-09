cmake_minimum_required(VERSION 2.8.3)
project(cf4ocl2-examples)

# Set modules directory
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules")

# Set version
set(${PROJECT_NAME}_VERSION_STRING "2.1.0")

# Set log domain
add_definitions(-DG_LOG_DOMAIN=\"${PROJECT_NAME}\")

# Activate profiling?
option(PROFILING "Use OpenCL profiling?" ON)
if (${PROFILING})
	add_definitions(-DWITH_PROFILING)
endif()

# Compiler options for GCC/Clang
# -Wno-comment because of comment within comment in OpenCL headers
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wextra -Wall -Wno-comment -std=c99")

# Find required libraries
find_package(PkgConfig REQUIRED)
pkg_check_modules(CF4OCL2 REQUIRED cf4ocl2>=2.0.0)
pkg_check_modules(GLIB REQUIRED glib-2.0>=2.32.1)
find_package(OpenMP QUIET)
find_package(OpenCL REQUIRED)

# Library include directories
include_directories(${OpenCL_INCLUDE_DIRS} ${CF4OCL2_INCLUDE_DIRS}
	${CMAKE_SOURCE_DIR}/src ${CMAKE_BINARY_DIR})

# Add src folder
add_subdirectory(src)
