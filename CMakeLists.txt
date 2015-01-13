cmake_minimum_required(VERSION 2.8)
project(cf4ocl2-examples)

# Set modules directory
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules")

# Set version
set(${PROJECT_NAME}_VERSION_STRING "2.0.0")

# Set log domain
add_definitions(-DG_LOG_DOMAIN=\"${PROJECT_NAME}\")

# Compiler options for GCC/Clang
# -Wno-comment because of comment within comment in OpenCL headers
if((${CMAKE_C_COMPILER_ID} STREQUAL "Clang")
        OR (${CMAKE_C_COMPILER_ID} STREQUAL "GNU"))
        # Clang and GCC flags
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wextra -Wall -Wno-comment -std=c99")
endif()

# Find required libraries
find_package(PkgConfig REQUIRED)
find_package(OpenCL REQUIRED)
pkg_check_modules(CF4OCL2 REQUIRED cf4ocl2>=2.0.0)
pkg_check_modules(GLIB REQUIRED glib-2.0>=2.0.0)
find_package(OpenMP QUIET)

# Library include directories
include_directories(${CF4OCL2_INCLUDE_DIRS} ${OPENCL_INCLUDE_DIRS}
	${CMAKE_SOURCE_DIR}/src ${CMAKE_BINARY_DIR})

# Add src folder
add_subdirectory(src)
