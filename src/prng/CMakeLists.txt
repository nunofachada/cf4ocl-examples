# Add a target for rng_ccl
add_executable(rng_ccl rng_ccl.c)
target_link_libraries(rng_ccl ${CF4OCL2_LIBRARIES})

# Add a target for rng_ocl
add_executable(rng_ocl rng_ocl.c)
target_link_libraries(rng_ocl ${OpenCL_LIBRARIES})

# Copy the OpenCL kernels to the same location as the executables
foreach(KERNEL init rng)
	add_custom_command(TARGET rng_ccl POST_BUILD
		COMMAND ${CMAKE_COMMAND} -E copy_if_different
		${CMAKE_CURRENT_SOURCE_DIR}/${KERNEL}.cl
		$<TARGET_FILE_DIR:rng_ccl>
	)
endforeach()

# Set compile and link flags
set_target_properties(rng_ccl PROPERTIES
	COMPILE_FLAGS "-Wno-unused-result"
	LINK_FLAGS "-pthread")
set_target_properties(rng_ocl PROPERTIES
	COMPILE_FLAGS "-Wno-deprecated-declarations -Wno-unused-result"
	LINK_FLAGS "-pthread")
