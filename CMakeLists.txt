cmake_minimum_required(VERSION 3.9.6)
project(weirdlib C CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# TODO: Remove unnecessary files from build sources when module disabled

cmake_policy(SET CMP0077 NEW)

##|-----------------------|##
##| COMPILATION SELECTION |##
##|-----------------------|##

add_compile_definitions(WEIRDLIB_MTMODE_NONE=0 WEIRDLIB_MTMODE_TBB=1 WEIRDLIB_MTMODE_OMP=2)

option(WEIRDLIB_BUILD_DOCS "Enable building documentation" ON)
option(WEIRDLIB_ENABLE_TESTING "Enable building test target" ON)
option(WEIRDLIB_ENABLE_APPLICATIONS "Enable building example applications" OFF)
option(WEIRDLIB_ENABLE_BUILD_STRINGOPS "Enable building of string module" ON)
option(WEIRDLIB_ENABLE_BUILD_CRYPTO "Enable building of crypto module" ON)
option(WERIDLIB_ENABLE_BUILD_IMAGEOPS "Enable building image operations module" ON)
option(WERIDLIB_ENABLE_BUILD_FILEOPS "Enable building file operations module" ON)
option(WERIDLIB_ENABLE_BUILD_PARSERS "Enable building parsers module" ON)
option(WERIDLIB_ENABLE_BUILD_VECMATH "Enable building vector math module" ON)

option(WEIRDLIB_INSTRUMENT_COVERAGE "Enable generating coverage instrumentation" OFF)
option(WEIRDLIB_ENABLE_MULTITHREADING "Enable use of external threading library" ON)
option(WEIRDLIB_FORCE_MAXIMUM_ARCHITECTURE_EXTENSIONS "Enable all instruction set extensions regardless of target system (MMX to all AVX-512 subsets)" OFF)
option(WEIRDLIB_BUILD_NATIVE "Enable building version specific to host system (with -march=native and -mtune=native) on gnu-compatible compilers" ON)
option(WEIRDLIB_ENABLE_PREFETCH "Enable use of prefetch instructions in optimized code.
This can lead to potential performance improvements, or massive slowdowns in some cases (Intel Ivy Bridge[43 cycles], AMD Jaguar[~100 cycles])" ON)

set(WEIRDLIB_SOURCES "")
set(WEIRDLIB_INCLUDE_SOURCES "")

# Module selection
include("${CMAKE_CURRENT_LIST_DIR}/cmake_extra/modules.cmake")

# Compilation selection helpers
set(WEIRDLIB_LINKER_LIBRARIES "")
set(WEIRDLIB_ADDITIONAL_CXX_FLAGS "")


include("${CMAKE_CURRENT_LIST_DIR}/cmake_extra/compiler_settings.cmake")


##|---------------------|##
##| MAIN TARGET LIBRARY |##
##|---------------------|##

message(STATUS "Configuring main library")

# Library sources and headers
list(APPEND WEIRDLIB_SOURCES "src/common.cpp" "src/common.hpp" "src/weirdlib.cpp")

# Primary build target
add_library(weirdlib ${WEIRDLIB_SOURCES} ${WEIRDLIB_INCLUDE_SOURCES})
set_target_properties(weirdlib PROPERTIES
	CXX_STANDARD 17
	CXX_STANDARD_REQUIRED ON
)

# STB
add_library(weirdlib_stb_impl "${CMAKE_CURRENT_LIST_DIR}/external/stb_impl.cpp")
target_compile_options(weirdlib_stb_impl PRIVATE
	$<$<CXX_COMPILER_ID:Clang>: -O3 -g0 -w >
	$<$<CXX_COMPILER_ID:GNU>: -O3 -g0 -w >
	$<$<CXX_COMPILER_ID:Intel>: -O3 -g0 -w >
	$<$<CXX_COMPILER_ID:MSVC>: /O2 /W0 >
)

target_link_libraries(weirdlib weirdlib_stb_impl)

##|-----------------------------|##
##| DOCUMENTATION CONFIGURATION |##
##|-----------------------------|##

if(WEIRDLIB_BUILD_DOCS)
	message(STATUS "Configuring documentation")

	find_package(Doxygen REQUIRED)
	set(DOXYGEN_IN ${CMAKE_CURRENT_LIST_DIR}/docs/Doxyfile.in)
	set(DOXYGEN_OUT ${CMAKE_CURRENT_LIST_DIR}/docs/Doxyfile)

	configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)
	message("Doxygen build started")

	add_custom_target(wlib_doc COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
		WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/docs
		COMMENT "Generating documentation..."
		VERBATIM
	)
endif(WEIRDLIB_BUILD_DOCS)


# Unit tests
include("${CMAKE_CURRENT_LIST_DIR}/tests/unit_tests.cmake")

# Applications
if(WEIRDLIB_ENABLE_APPLICATIONS)
	add_subdirectory(apps)
endif(WEIRDLIB_ENABLE_APPLICATIONS)
