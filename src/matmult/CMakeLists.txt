# Current example
set(EXAMPLE matmult)

# Add a target for current example
add_executable(${EXAMPLE} ${EXAMPLE}.c)
target_link_libraries(${EXAMPLE} examples_common)

# Copy the OpenCL kernel to the same location as the example executable
add_custom_command(TARGET ${EXAMPLE} POST_BUILD
	COMMAND ${CMAKE_COMMAND} -E copy_if_different
	${CMAKE_CURRENT_SOURCE_DIR}/${EXAMPLE}.cl
	$<TARGET_FILE_DIR:${EXAMPLE}>
)

# Set specific matmult properties
if (OPENMP_FOUND)
	set_target_properties(matmult PROPERTIES
		COMPILE_FLAGS "${OpenMP_C_FLAGS} -DUSE_OPENMP"
		LINK_FLAGS ${OpenMP_C_FLAGS})
endif()
