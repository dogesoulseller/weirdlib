cmake_minimum_required(VERSION 3.9.6)
project(weirdlib C CXX)

add_compile_definitions(WEIRDLIB_MTMODE_NONE=0 WEIRDLIB_MTMODE_TBB=1 WEIRDLIB_MTMODE_OMP=2)

option(WEIRDLIB_INSTRUMENT_COVERAGE "Enable generating coverage instrumentation" OFF)
option(WEIRDLIB_BUILD_DOCS "Enable building documentation" ON)
option(WEIRDLIB_ENABLE_TESTING "Enable building test target" ON)
option(WEIRDLIB_ENABLE_BENCHMARKS "(Stub) Enable building benchmark target" ON)
option(WEIRDLIB_ENABLE_BUILD_STRINGOPS "Enable building of string module" ON)
option(WEIRDLIB_ENABLE_BUILD_CRYPTO "Enable building of crypto module" ON)
option(WERIDLIB_ENABLE_BUILD_IMAGEOPS "Enable building image operations module" ON)
option(WEIRDLIB_ENABLE_PREFETCH "Enable use of prefetch instructions in optimized code.
This can lead to potential performance improvements, or massive slowdowns in some cases (Intel Ivy Bridge[43 cycles], AMD Jaguar[~100 cycles])" ON)

# Compilation selection helpers
set(WEIRDLIB_LINKER_LIBRARIES "")
set(WEIRDLIB_ADDITIONAL_CXX_FLAGS "")

# Multithreading options
option(WEIRDLIB_ENABLE_MULTITHREADING "Enable use of external threading library" ON)
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
endif(WEIRDLIB_ENABLE_MULTITHREADING)

set(CMAKE_CXX_FLAGS "${WEIRDLIB_ADDITIONAL_CXX_FLAGS} ${CMAKE_CXX_FLAGS}")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
file(GLOB_RECURSE WEIRDLIB_SOURCES "include/*.hpp" "src/*.c*" "src/*.h*")

# Force use of Intel syntax assembly
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Intel")
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -masm=intel -Wno-unknown-pragmas -march=native")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -masm=intel -Wno-unknown-pragmas -march=native")
endif(CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Intel")

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Intel")
	set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -g3 -Wall -Wextra -Wpedantic -Wno-unknown-pragmas")
	set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g3 -Wall -Wextra -Wpedantic -Wno-unknown-pragmas")
endif(CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Intel")

# Enable standalone debug info generation
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
	set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -fstandalone-debug")
	set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fstandalone-debug")

	set(CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_C_FLAGS_RELWITHDEBINFO} -fstandalone-debug")
	set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -fstandalone-debug")
endif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Intel")
	set(CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_C_FLAGS_RELWITHDEBINFO} -g3 -O2 -Wno-unknown-pragmas")
	set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -g3 -O2 -Wno-unknown-pragmas")
endif(CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Intel")

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -mtune=native -mllvm -polly -fstandalone-debug -Wno-c++98-compat-pedantic -Wno-switch-enum -Wno-covered-switch-default -Wno-exit-time-destructors -Wno-missing-variable-declarations -Wno-global-constructors -Wno-c++98-c++11-c++14-compat -Wno-c++98-compat -Wno-pedantic -Wno-c++98-c++11-compat-binary-literal")
# set(CMAKE_CXX_FLAGS "-march=native -mtune=native -mavx512f -mavx512bw -mavx512dq -mavx512vl -fstandalone-debug")

# Clang instrumentation
if(WEIRDLIB_INSTRUMENT_COVERAGE AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fprofile-instr-generate -fcoverage-mapping")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-instr-generate -fcoverage-mapping")
endif(WEIRDLIB_INSTRUMENT_COVERAGE AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")

# Primary build target
add_library(weirdlib ${WEIRDLIB_SOURCES})
set_target_properties(weirdlib PROPERTIES
	CXX_STANDARD 17
	CXX_STANDARD_REQUIRED ON
)

# Enable additional config definitions
if(WEIRDLIB_ENABLE_MULTITHREADING)
	if(WEIRDLIB_MULTITHREADING_LIBRARY STREQUAL "openmp" OR WEIRDLIB_MULTITHREADING_LIBRARY STREQUAL "omp")
		add_compile_definitions(WEIRDLIB_MULTITHREADING_MODE=2)
	elseif(WEIRDLIB_MULTITHREADING_LIBRARY STREQUAL "tbb")
		add_compile_definitions(WEIRDLIB_MULTITHREADING_MODE=1)
	endif()

endif(WEIRDLIB_ENABLE_MULTITHREADING)

# Conditionally enable building cryptography module
if(WEIRDLIB_ENABLE_BUILD_CRYPTO)
	target_compile_definitions(weirdlib PUBLIC WEIRDLIB_ENABLE_CRYPTOGRAPHY)
endif(WEIRDLIB_ENABLE_BUILD_CRYPTO)

# Conditionally enable building string operations module
if(WEIRDLIB_ENABLE_BUILD_STRINGOPS)
	target_compile_definitions(weirdlib PUBLIC WEIRDLIB_ENABLE_STRING_OPERATIONS)
endif(WEIRDLIB_ENABLE_BUILD_STRINGOPS)

# Conditionally enable building image operations module
if(WERIDLIB_ENABLE_BUILD_IMAGEOPS)
	target_compile_definitions(weirdlib PUBLIC WEIRDLIB_ENABLE_IMAGE_OPERATIONS)
endif(WERIDLIB_ENABLE_BUILD_IMAGEOPS)

# Conditionally enable use of prefetch instructions
if(WEIRDLIB_ENABLE_PREFETCH)
	target_compile_definitions(weirdlib PUBLIC WLIB_ENABLE_PREFETCH)
endif(WEIRDLIB_ENABLE_PREFETCH)


# API Documentation
if(WEIRDLIB_BUILD_DOCS)
	find_package(Doxygen)
	if(DOXYGEN_FOUND)
		set(DOXYGEN_IN ${CMAKE_CURRENT_LIST_DIR}/docs/Doxyfile.in)
		set(DOXYGEN_OUT ${CMAKE_CURRENT_LIST_DIR}/docs/Doxyfile)

		configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)
		message("Doxygen build started")

		add_custom_target(wlib_doc COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
			WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/docs
			COMMENT "Generating documentation..."
			VERBATIM
		)
	endif(DOXYGEN_FOUND)

endif(WEIRDLIB_BUILD_DOCS)

# Library benchmarks
if(WEIRDLIB_ENABLE_BENCHMARKS)
	set(WEIRDLIB_BENCHMARK_DIR "${CMAKE_CURRENT_LIST_DIR}/benchmark")
	add_executable(wlib_bench ${WEIRDLIB_BENCHMARK_DIR}/wlibbench.cpp ${WEIRDLIB_BENCHMARK_DIR}/bench_load.cpp ${WEIRDLIB_BENCHMARK_DIR}/bench_cases.cpp)
	set_target_properties(wlib_bench PROPERTIES
		CXX_STANDARD 17
		CXX_STANDARD_REQUIRED ON
	)
	target_link_libraries(wlib_bench weirdlib Threads::Threads ${WEIRDLIB_LINKER_LIBRARIES})

endif(WEIRDLIB_ENABLE_BENCHMARKS)

# Google Test testing
if(WEIRDLIB_ENABLE_TESTING)
	enable_testing()

	find_package(GTest REQUIRED)
	include(GoogleTest)
	file(GLOB WEIRDLIB_TEST_SOURCES "tests/*.cpp")

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

endif(WEIRDLIB_ENABLE_TESTING)
