cmake_minimum_required(VERSION 3.9.6)
project(weirdlib)

option(WEIRDLIB_ENABLE_TESTING "Enable building test target" ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
file(GLOB WEIRDLIB_SOURCES "include/*.hpp" "src/*.c*" "src/*.h*")

add_library(weirdlib ${WEIRDLIB_SOURCES})
set_target_properties(weirdlib PROPERTIES
	CXX_STANDARD 17
	CXX_STANDARD_REQUIRED ON
)

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

	# set(CMAKE_CXX_FLAGS "-march=native -mtune=native -mavx512f -mavx512bw -mavx512dq -mavx512vl -fstandalone-debug")
	# set(CMAKE_CXX_FLAGS "-march=native -mtune=native -fstandalone-debug -mavx2 -Weverything")

	gtest_add_tests(TARGET weirdlib_tests SOURCES ${WEIRDLIB_TEST_SOURCES} WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/tests/")
endif(WEIRDLIB_ENABLE_TESTING)
