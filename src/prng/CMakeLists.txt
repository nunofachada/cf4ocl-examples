# Current example
set(EXAMPLE rng_ccl)

# Add a target for current example
add_executable(${EXAMPLE} ${EXAMPLE}.c)
target_link_libraries(${EXAMPLE} ${CF4OCL2_LIBRARIES})

# Copy the OpenCL kernels to the same location as the example executable
foreach(KERNEL init rng)
	# Copy the OpenCL kernel to the same location as the example executable
	add_custom_command(TARGET ${EXAMPLE} POST_BUILD
		COMMAND ${CMAKE_COMMAND} -E copy_if_different
		${CMAKE_CURRENT_SOURCE_DIR}/${KERNEL}.cl
		$<TARGET_FILE_DIR:rng_ccl>
	)
endforeach()

# Set pthread link flag
set_target_properties(${EXAMPLE} PROPERTIES
	LINK_FLAGS "-pthread")
