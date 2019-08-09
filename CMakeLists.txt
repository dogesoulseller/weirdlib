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
option(WEIRDLIB_ENABLE_BENCHMARKS "(Stub) Enable building benchmark target" ON)
option(WEIRDLIB_ENABLE_BUILD_STRINGOPS "Enable building of string module" ON)
option(WEIRDLIB_ENABLE_BUILD_CRYPTO "Enable building of crypto module" ON)
option(WERIDLIB_ENABLE_BUILD_IMAGEOPS "Enable building image operations module" ON)
option(WERIDLIB_ENABLE_BUILD_ANXIETY "Enable building stress test module" ON)
option(WEIRDLIB_FORCE_MAXIMUM_ARCHITECTURE_EXTENSIONS "Enable all instruction set extensions regardless of target system (MMX to all AVX-512 subsets)" OFF)
option(WEIRDLIB_ENABLE_PREFETCH "Enable use of prefetch instructions in optimized code.
This can lead to potential performance improvements, or massive slowdowns in some cases (Intel Ivy Bridge[43 cycles], AMD Jaguar[~100 cycles])" ON)

# Compilation selection helpers
set(WEIRDLIB_LINKER_LIBRARIES "")
set(WEIRDLIB_ADDITIONAL_CXX_FLAGS "")

##|----------------|##
##| MULTITHREADING |##
##|----------------|##

# Multithreading options
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
		string(APPEND CMAKE_C_FLAGS_DEBUG " -Weverything -Wno-padded -Wno-c++98-compat -Wno-c++98-c++11-compat -Wno-c++98-c++11-c++14-compat -Wno-weak-vtables -Wno-global-constructors -Wno-switch-enum -Wno-unknown-pragmas -Wno-covered-switch-default -Wno-ctad-maybe-unsupported -Wno-c++98-c++11-compat-binary-literal -Wno-exit-time-destructors ")
		string(APPEND CMAKE_CXX_FLAGS_DEBUG " -Weverything -Wno-padded -Wno-c++98-compat -Wno-c++98-c++11-compat -Wno-c++98-c++11-c++14-compat -Wno-weak-vtables -Wno-global-constructors -Wno-switch-enum -Wno-unknown-pragmas -Wno-covered-switch-default -Wno-ctad-maybe-unsupported -Wno-c++98-c++11-compat-binary-literal -Wno-exit-time-destructors ")

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
		string(APPEND CMAKE_C_FLAGS " -mmmx -msse -msse2 -msse3 -mssse3 -msse4.1 -msse4.2 -mavx -mavx2 -mavx512f -mavx512pf -mavx512er -mavx512cd -mavx512vl -mavx512bw -mavx512dq -mavx512ifma -mavx512vbmi -mavx512bitalg -mavx512vpopcntdq -mavx5124fmaps -mavx512vnni -mavx5124vnniw -mpclmul -mfma -mbmi -mbmi2 -mlzcnt -mtbm -mpopcnt -maes -msha ")
	endif(WEIRDLIB_FORCE_MAXIMUM_ARCHITECTURE_EXTENSIONS)

endif(CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Intel")

string(APPEND CMAKE_CXX_FLAGS "${WEIRDLIB_ADDITIONAL_CXX_FLAGS}")

##|-------------------------|##
##| BUILD TARGET SUBFOLDERS |##
##|-------------------------|##

# Library
add_subdirectory("./src")

# API Documentation
if(WEIRDLIB_BUILD_DOCS)
	add_subdirectory("./docs")
endif(WEIRDLIB_BUILD_DOCS)

# Library benchmarks
if(WEIRDLIB_ENABLE_BENCHMARKS)
	add_subdirectory("./benchmark")
endif(WEIRDLIB_ENABLE_BENCHMARKS)

# Google Test testing
if(WEIRDLIB_ENABLE_TESTING)
	add_subdirectory("./tests")
endif(WEIRDLIB_ENABLE_TESTING)
