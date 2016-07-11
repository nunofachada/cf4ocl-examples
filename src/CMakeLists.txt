# Setup the configuration header
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/examples_common.in.h
	${CMAKE_BINARY_DIR}/examples_common.h @ONLY)

# Add common examples library
add_library(examples_common examples_common.c)
target_link_libraries(examples_common ${CF4OCL2_LIBRARIES} m ${GLIB_LIBRARIES})

# Process examples
add_subdirectory(bankconf)
add_subdirectory(ca_mt)
add_subdirectory(matmult)
add_subdirectory(prng)
