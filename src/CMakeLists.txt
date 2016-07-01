# List of examples
SET(EXAMPLES matmult bank_conflicts ca_mt)

# Add common examples library
add_library(examples_common examples_common.c)
target_link_libraries(examples_common ${CF4OCL2_LIBRARIES} m ${GLIB_LIBRARIES})

# Setup the configuration header
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/examples_common.in.h
	${CMAKE_BINARY_DIR}/examples_common.h @ONLY)

# Download header required for ca_mt
if (NOT EXISTS "stb_image_write.h")
	file(DOWNLOAD
		"https://raw.githubusercontent.com/nothings/stb/master/stb_image_write.h"
		"${CMAKE_BINARY_DIR}/stb_image_write.h")
endif()

# Is required header present?
if (NOT EXISTS "${CMAKE_BINARY_DIR}/stb_image_write.h")
	message(FATAL_ERROR "Unable to download stb_image_write.h")
endif()

# Add a target for each example
foreach(EXAMPLE ${EXAMPLES})
	add_executable(${EXAMPLE} ${EXAMPLE}.c)
	target_link_libraries(${EXAMPLE} examples_common)
	# Copy the OpenCL kernel to the same location as the example executable
	add_custom_command(TARGET ${EXAMPLE} POST_BUILD
		COMMAND ${CMAKE_COMMAND} -E copy_if_different
		${CMAKE_CURRENT_SOURCE_DIR}/${EXAMPLE}.cl
		$<TARGET_FILE_DIR:${EXAMPLE}>
	)
endforeach(EXAMPLE)

# Set specific matmult properties
if (OPENMP_FOUND)
	set_target_properties(matmult PROPERTIES
		COMPILE_FLAGS "${OpenMP_C_FLAGS} -DUSE_OPENMP"
		LINK_FLAGS ${OpenMP_C_FLAGS})
endif()
