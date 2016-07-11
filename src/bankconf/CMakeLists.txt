# Current example
set(EXAMPLE bank_conflicts)

# Add a target for current example
add_executable(${EXAMPLE} ${EXAMPLE}.c)
target_link_libraries(${EXAMPLE} examples_common)

# Copy the OpenCL kernel to the same location as the example executable
add_custom_command(TARGET ${EXAMPLE} POST_BUILD
	COMMAND ${CMAKE_COMMAND} -E copy_if_different
	${CMAKE_CURRENT_SOURCE_DIR}/${EXAMPLE}.cl
	$<TARGET_FILE_DIR:${EXAMPLE}>
)
