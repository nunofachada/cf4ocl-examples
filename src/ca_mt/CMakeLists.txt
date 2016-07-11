# Current example
set(EXAMPLE ca_mt)

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

# Add a target for current example
add_executable(${EXAMPLE} ${EXAMPLE}.c)
target_link_libraries(${EXAMPLE} examples_common)

# Copy the OpenCL kernel to the same location as the example executable
add_custom_command(TARGET ${EXAMPLE} POST_BUILD
	COMMAND ${CMAKE_COMMAND} -E copy_if_different
	${CMAKE_CURRENT_SOURCE_DIR}/${EXAMPLE}.cl
	$<TARGET_FILE_DIR:${EXAMPLE}>
)
