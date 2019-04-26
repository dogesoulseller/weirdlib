cmake_minimum_required(VERSION 3.9.6)
project(weirdlib CXX)

option(WEIRDLIB_ENABLE_TESTING "Enable building test target" ON)
option(WEIRDLIB_ENABLE_BENCHMARKS "(Stub) Enable building benchmark target" OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
file(GLOB WEIRDLIB_SOURCES "include/*.hpp" "src/*.c*" "src/*.h*")

# Force use of Intel syntax assembly
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Intel")
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -masm=intel")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -masm=intel")
endif()

add_library(weirdlib ${WEIRDLIB_SOURCES})
set_target_properties(weirdlib PROPERTIES
	CXX_STANDARD 17
	CXX_STANDARD_REQUIRED ON
)
# set(CMAKE_CXX_FLAGS "-march=native -mtune=native -mllvm -polly -fstandalone-debug -Weverything -Wno-missing-variable-declarations -Wno-global-constructors -Wno-c++98-c++11-c++14-compat -Wno-c++98-compat -Wno-pedantic -Wno-c++98-c++11-compat-binary-literal")
# set(CMAKE_CXX_FLAGS "-march=native -mtune=native -mavx512f -mavx512bw -mavx512dq -mavx512vl -fstandalone-debug")

if(WEIRDLIB_ENABLE_BENCHMARKS)
	set(WEIRDLIB_BENCHMARK_DIR "${CMAKE_CURRENT_LIST_DIR}/benchmark")
	add_executable(wlib_bench_strlen ${WEIRDLIB_BENCHMARK_DIR}/bench_strlen.cpp)
	set_target_properties(wlib_bench_strlen PROPERTIES
		CXX_STANDARD 17
		CXX_STANDARD_REQUIRED ON
	)
	target_link_libraries(wlib_bench_strlen weirdlib)

endif(WEIRDLIB_ENABLE_BENCHMARKS)

if(WEIRDLIB_ENABLE_TESTING)
	enable_testing()

	find_package(GTest REQUIRED)
	include(GoogleTest)
	file(GLOB WEIRDLIB_TEST_SOURCES "tests/*.cpp")

	add_executable(weirdlib_tests ${WEIRDLIB_TEST_SOURCES})
	target_link_libraries(weirdlib_tests Threads::Threads weirdlib GTest::GTest)
	set_target_properties(weirdlib_tests PROPERTIES
		CXX_STANDARD 17
		CXX_STANDARD_REQUIRED ON
	)

	gtest_add_tests(TARGET weirdlib_tests SOURCES ${WEIRDLIB_TEST_SOURCES} WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/tests/")
endif(WEIRDLIB_ENABLE_TESTING)
