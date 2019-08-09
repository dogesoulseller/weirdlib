message(STATUS "Configuring benchmarks")

set(WEIRDLIB_BENCHMARK_DIR "${PROJECT_SOURCE_DIR}/benchmark")
add_executable(wlib_bench ${WEIRDLIB_BENCHMARK_DIR}/wlibbench.cpp ${WEIRDLIB_BENCHMARK_DIR}/bench_load.cpp ${WEIRDLIB_BENCHMARK_DIR}/bench_cases.cpp)
set_target_properties(wlib_bench PROPERTIES
	CXX_STANDARD 17
	CXX_STANDARD_REQUIRED ON
)
target_link_libraries(wlib_bench weirdlib Threads::Threads ${WEIRDLIB_LINKER_LIBRARIES})
