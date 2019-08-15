option(WEIRDLIB_ENABLE_APP_EXTENSION_FIXER "Enable building extension fixer application" ON)
include_directories("${PROJECT_SOURCE_DIR}/include")

# File extension fixer
if(WEIRDLIB_ENABLE_APP_EXTENSION_FIXER)
	if (NOT WERIDLIB_ENABLE_BUILD_FILEOPS)
		message(SEND_ERROR "Requested building extension fixer but fileops module will not be built")
	endif()
	file(GLOB WLIB_EXTFIXER_SOURCES "src/extension_fix/*.c*" "src/extension_fix/*.h*")
	add_executable(extension-fix ${WLIB_EXTFIXER_SOURCES})
	set_target_properties(extension-fix PROPERTIES
		CXX_STANDARD 17
		CXX_STANDARD_REQUIRED ON
	)

	target_link_libraries(extension-fix PRIVATE weirdlib)
endif(WEIRDLIB_ENABLE_APP_EXTENSION_FIXER)
