cmake_minimum_required(VERSION 3.9.6)
project(weirdlib C CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)




##|-----------------------|##
##| COMPILATION SELECTION |##
##|-----------------------|##

add_compile_definitions(WEIRDLIB_MTMODE_NONE=0 WEIRDLIB_MTMODE_TBB=1 WEIRDLIB_MTMODE_OMP=2)

option(WEIRDLIB_ENABLE_MULTITHREADING "Enable use of external threading library" ON)
option(WEIRDLIB_INSTRUMENT_COVERAGE "Enable generating coverage instrumentation" OFF)
option(WEIRDLIB_BUILD_DOCS "Enable building documentation" ON)
option(WEIRDLIB_ENABLE_TESTING "Enable building test target" ON)
option(WEIRDLIB_ENABLE_BENCHMARKS "Enable building benchmark target" ON)
option(WEIRDLIB_ENABLE_APPLICATIONS "Enable building example applications" ON)
option(WEIRDLIB_ENABLE_BUILD_STRINGOPS "Enable building of string module" ON)
option(WEIRDLIB_ENABLE_BUILD_CRYPTO "Enable building of crypto module" ON)
option(WERIDLIB_ENABLE_BUILD_IMAGEOPS "Enable building image operations module" ON)
option(WERIDLIB_ENABLE_BUILD_FILEOPS "Enable building file operations module" ON)
option(WERIDLIB_ENABLE_BUILD_ANXIETY "Enable building stress test module" ON)
option(WEIRDLIB_FORCE_MAXIMUM_ARCHITECTURE_EXTENSIONS "Enable all instruction set extensions regardless of target system (MMX to all AVX-512 subsets)" OFF)
option(WEIRDLIB_ENABLE_PREFETCH "Enable use of prefetch instructions in optimized code.
This can lead to potential performance improvements, or massive slowdowns in some cases (Intel Ivy Bridge[43 cycles], AMD Jaguar[~100 cycles])" ON)

# Cryptography module
if(WEIRDLIB_ENABLE_BUILD_CRYPTO)
	add_compile_definitions(WEIRDLIB_ENABLE_CRYPTOGRAPHY)
endif(WEIRDLIB_ENABLE_BUILD_CRYPTO)

# String operations module
if(WEIRDLIB_ENABLE_BUILD_STRINGOPS)
	add_compile_definitions(WEIRDLIB_ENABLE_STRING_OPERATIONS)
endif(WEIRDLIB_ENABLE_BUILD_STRINGOPS)

# Image operations module
if(WERIDLIB_ENABLE_BUILD_IMAGEOPS)
	add_compile_definitions(WEIRDLIB_ENABLE_IMAGE_OPERATIONS)
endif(WERIDLIB_ENABLE_BUILD_IMAGEOPS)

# Stress test module
if(WERIDLIB_ENABLE_BUILD_ANXIETY)
	add_compile_definitions(WEIRDLIB_ENABLE_ANXIETY)
endif(WERIDLIB_ENABLE_BUILD_ANXIETY)

# Conditionally enable use of prefetch instructions
if(WEIRDLIB_ENABLE_PREFETCH)
	add_compile_definitions(WLIB_ENABLE_PREFETCH)
endif(WEIRDLIB_ENABLE_PREFETCH)

# Compilation selection helpers
set(WEIRDLIB_LINKER_LIBRARIES "")
set(WEIRDLIB_ADDITIONAL_CXX_FLAGS "")




##|----------------|##
##| MULTITHREADING |##
##|----------------|##

find_package(Threads REQUIRED)
set(WEIRDLIB_MULTITHREADING_LIBRARY "tbb" CACHE STRING "Which external threading library to use: OpenMP, TBB")
set_property(CACHE WEIRDLIB_MULTITHREADING_LIBRARY PROPERTY STRINGS OpenMP TBB)

if(WEIRDLIB_ENABLE_MULTITHREADING)
	find_package(OpenMP QUIET)

	string(TOLOWER ${WEIRDLIB_MULTITHREADING_LIBRARY} WEIRDLIB_MULTITHREADING_LIBRARY)
	if(WEIRDLIB_MULTITHREADING_LIBRARY STREQUAL "openmp" OR WEIRDLIB_MULTITHREADING_LIBRARY STREQUAL "omp")
		string(APPEND WEIRDLIB_ADDITIONAL_CXX_FLAGS ${OpenMP_CXX_FLAGS})
	elseif(WEIRDLIB_MULTITHREADING_LIBRARY STREQUAL "tbb")
		string(APPEND WEIRDLIB_LINKER_LIBRARIES "tbb")
	endif()

	# Enable additional config definitions
	if(WEIRDLIB_MULTITHREADING_LIBRARY STREQUAL "openmp" OR WEIRDLIB_MULTITHREADING_LIBRARY STREQUAL "omp")
		add_compile_definitions(WEIRDLIB_MULTITHREADING_MODE=2)
	elseif(WEIRDLIB_MULTITHREADING_LIBRARY STREQUAL "tbb")
		add_compile_definitions(WEIRDLIB_MULTITHREADING_MODE=1)
	endif()

endif(WEIRDLIB_ENABLE_MULTITHREADING)




##|----------------------------|##
##| COMPILER-DEPENDENT OPTIONS |##
##|----------------------------|##

# Enable additional flags depending on host compiler and target
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Intel")
	message(STATUS "Compiler ${CMAKE_CXX_COMPILER_ID} is GCC-compatible")

	# Force using Intel assembly syntax
	string(APPEND CMAKE_C_FLAGS " -masm=intel ")
	string(APPEND CMAKE_CXX_FLAGS " -masm=intel ")

	# Set correct optimization and symbol flags for profile builds
	string(APPEND CMAKE_C_FLAGS_RELWITHDEBINFO " -g3 -Og ")
	string(APPEND CMAKE_CXX_FLAGS_RELWITHDEBINFO " -g3 -Og ")

	# Enable warnings
	string(APPEND CMAKE_C_FLAGS_DEBUG " -g3 -Wall -Wextra -Wpedantic -Wno-unknown-pragmas ")
	string(APPEND CMAKE_CXX_FLAGS_DEBUG " -g3 -Wall -Wextra -Wpedantic -Wno-unknown-pragmas ")

	# Clang-specific settings
	if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
		# Enable warnings in Clang and disable multiple unwanted ones
		string(APPEND CMAKE_C_FLAGS_DEBUG " -Weverything -Wno-sign-conversion -Wno-implicit-int-conversion -Wno-padded -Wno-c++98-compat -Wno-c++98-c++11-compat -Wno-c++98-c++11-c++14-compat -Wno-weak-vtables -Wno-global-constructors -Wno-switch-enum -Wno-unknown-pragmas -Wno-covered-switch-default -Wno-ctad-maybe-unsupported -Wno-c++98-c++11-compat-binary-literal -Wno-exit-time-destructors ")
		string(APPEND CMAKE_CXX_FLAGS_DEBUG " -Weverything -Wno-sign-conversion -Wno-implicit-int-conversion -Wno-padded -Wno-c++98-compat -Wno-c++98-c++11-compat -Wno-c++98-c++11-c++14-compat -Wno-weak-vtables -Wno-global-constructors -Wno-switch-enum -Wno-unknown-pragmas -Wno-covered-switch-default -Wno-ctad-maybe-unsupported -Wno-c++98-c++11-compat-binary-literal -Wno-exit-time-destructors ")

		# Enable standalone debug information
		string(APPEND CMAKE_C_FLAGS_DEBUG " -fstandalone-debug ")
		string(APPEND CMAKE_CXX_FLAGS_DEBUG " -fstandalone-debug ")
		string(APPEND CMAKE_C_FLAGS_RELWITHDEBINFO " -fstandalone-debug ")
		string(APPEND CMAKE_CXX_FLAGS_RELWITHDEBINFO " -fstandalone-debug ")

		# Enable embedding coverage instrumentation with Clang
		if(WEIRDLIB_INSTRUMENT_COVERAGE)
			string(APPEND CMAKE_C_FLAGS " -fprofile-instr-generate -fcoverage-mapping ")
			string(APPEND CMAKE_CXX_FLAGS " -fprofile-instr-generate -fcoverage-mapping ")
		endif(WEIRDLIB_INSTRUMENT_COVERAGE)
	endif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")

	# Enable all instruction set extensions
	if(WEIRDLIB_FORCE_MAXIMUM_ARCHITECTURE_EXTENSIONS)
		string(APPEND CMAKE_C_FLAGS " -mmmx -msse -msse2 -msse3 -mssse3 -msse4.1 -msse4.2 -mavx -mavx2 -mavx512f -mavx512pf -mavx512er -mavx512cd -mavx512vl -mavx512bw -mavx512dq -mavx512ifma -mavx512vbmi -mavx512bitalg -mavx512vpopcntdq -mpclmul -mfma -mbmi -mbmi2 -mlzcnt -mtbm -mpopcnt -maes -msha ")
		string(APPEND CMAKE_CXX_FLAGS " -mmmx -msse -msse2 -msse3 -mssse3 -msse4.1 -msse4.2 -mavx -mavx2 -mavx512f -mavx512pf -mavx512er -mavx512cd -mavx512vl -mavx512bw -mavx512dq -mavx512ifma -mavx512vbmi -mavx512bitalg -mavx512vpopcntdq -mpclmul -mfma -mbmi -mbmi2 -mlzcnt -mtbm -mpopcnt -maes -msha ")
	endif(WEIRDLIB_FORCE_MAXIMUM_ARCHITECTURE_EXTENSIONS)

endif(CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Intel")
message(STATUS "Flags: ${CMAKE_CXX_FLAGS}")
string(APPEND CMAKE_CXX_FLAGS " ${WEIRDLIB_ADDITIONAL_CXX_FLAGS} ")




##|---------------------|##
##| MAIN TARGET LIBRARY |##
##|---------------------|##

message(STATUS "Configuring main library")

# Library sources and headers
file(GLOB_RECURSE WEIRDLIB_SOURCES "include/*.hpp" "src/*.c*" "src/*.h*")

# Primary build target
add_library(weirdlib ${WEIRDLIB_SOURCES})
set_target_properties(weirdlib PROPERTIES
	CXX_STANDARD 17
	CXX_STANDARD_REQUIRED ON
)




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




##|--------------------------|##
##| BENCHMARKS CONFIGURATION |##
##|--------------------------|##

message(STATUS "Configuring benchmarks")

set(WEIRDLIB_BENCHMARK_DIR "${PROJECT_SOURCE_DIR}/benchmark")
add_executable(wlib_bench ${WEIRDLIB_BENCHMARK_DIR}/wlibbench.cpp ${WEIRDLIB_BENCHMARK_DIR}/bench_load.cpp ${WEIRDLIB_BENCHMARK_DIR}/bench_cases.cpp)
set_target_properties(wlib_bench PROPERTIES
	CXX_STANDARD 17
	CXX_STANDARD_REQUIRED ON
)
target_link_libraries(wlib_bench weirdlib Threads::Threads ${WEIRDLIB_LINKER_LIBRARIES})




##|--------------------------|##
##| TEST SUITE CONFIGURATION |##
##|--------------------------|##

if (WEIRDLIB_ENABLE_TESTING)
	message(STATUS "Configuring tests")

	enable_testing()
	find_package(GTest REQUIRED)
	include(GoogleTest)
	file(GLOB WEIRDLIB_TEST_SOURCES "tests/*.c*")

	add_executable(weirdlib_tests ${WEIRDLIB_TEST_SOURCES})
	target_link_libraries(weirdlib_tests Threads::Threads weirdlib ${WEIRDLIB_LINKER_LIBRARIES} GTest::GTest)
	set_target_properties(weirdlib_tests PROPERTIES
		CXX_STANDARD 17
		CXX_STANDARD_REQUIRED ON
	)

	target_compile_definitions(weirdlib_tests PUBLIC WLIBTEST_TESTING_DIRECTORY=\"${CMAKE_CURRENT_LIST_DIR}/tests/\")

	gtest_add_tests(TARGET weirdlib_tests SOURCES ${WEIRDLIB_TEST_SOURCES} WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/tests/")

	# Coverage information
	if(WEIRDLIB_INSTRUMENT_COVERAGE)
		set(TEST_EXECUTABLE_TARGET_NAME weirdlib_tests)

		add_custom_target(wlib-coverage
			COMMAND LLVM_PROFILE_FILE=${TEST_EXECUTABLE_TARGET_NAME}.profraw $<TARGET_FILE:${TEST_EXECUTABLE_TARGET_NAME}>
			COMMAND llvm-profdata merge -sparse ${TEST_EXECUTABLE_TARGET_NAME}.profraw -o ${TEST_EXECUTABLE_TARGET_NAME}.profdata
			COMMAND llvm-cov report $<TARGET_FILE:${TEST_EXECUTABLE_TARGET_NAME}> -instr-profile=${TEST_EXECUTABLE_TARGET_NAME}.profdata
				-ignore-filename-regex="external|traits" -use-color=true ${WEIRDLIB_SOURCES}
			DEPENDS ${TEST_EXECUTABLE_TARGET_NAME}
		)

		add_custom_target(wlib-coverage-html
			COMMAND llvm-cov show $<TARGET_FILE:${TEST_EXECUTABLE_TARGET_NAME}> -instr-profile=${TEST_EXECUTABLE_TARGET_NAME}.profdata
				-show-line-counts-or-regions -ignore-filename-regex="external|traits"
				-output-dir=${CMAKE_CURRENT_LIST_DIR}/coverage -format="html"
			DEPENDS wlib-coverage
		)
	endif(WEIRDLIB_INSTRUMENT_COVERAGE)
endif (WEIRDLIB_ENABLE_TESTING)


if(WEIRDLIB_ENABLE_APPLICATIONS)
	add_subdirectory(apps)
endif(WEIRDLIB_ENABLE_APPLICATIONS)
