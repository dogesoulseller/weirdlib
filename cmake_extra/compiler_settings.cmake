# Conditionally enable use of prefetch instructions
if(WEIRDLIB_ENABLE_PREFETCH)
	add_compile_definitions(WLIB_ENABLE_PREFETCH)
endif(WEIRDLIB_ENABLE_PREFETCH)


##|----------------|##
##| MULTITHREADING |##
##|----------------|##

set(WEIRDLIB_MULTITHREADING_LIBRARY "tbb" CACHE STRING "Which external threading library to use: OpenMP, TBB")
set_property(CACHE WEIRDLIB_MULTITHREADING_LIBRARY PROPERTY STRINGS OpenMP TBB)

if(WEIRDLIB_ENABLE_MULTITHREADING)
	find_package(Threads REQUIRED)

	if(WEIRDLIB_MULTITHREADING_LIBRARY STREQUAL "openmp")
		find_package(OpenMP QUIET)
	endif(WEIRDLIB_MULTITHREADING_LIBRARY STREQUAL "openmp")


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

	# Non-ICC
	if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Intel")
		# Set correct optimization and symbol flags for profile builds
		string(APPEND CMAKE_C_FLAGS_RELWITHDEBINFO " -g3 -Og ")
		string(APPEND CMAKE_CXX_FLAGS_RELWITHDEBINFO " -g3 -Og ")

		# Enable warnings
		string(APPEND CMAKE_C_FLAGS_DEBUG " -g3 -Wall -Wextra -Wpedantic -Wno-unknown-pragmas -Wnull-dereference -Wparentheses -Wjump-misses-init -Wdouble-promotion -Wshadow -Wformat=2 -Wcast-qual")
		string(APPEND CMAKE_CXX_FLAGS_DEBUG " -g3 -Wall -Wextra -Wpedantic -Wno-unknown-pragmas -Wnull-dereference -Wparentheses -Wold-style-cast -Wdouble-promotion -Wshadow -Wformat=2 -Wcast-qual")
		string(APPEND CMAKE_C_FLAGS_RELWITHDEBINFO " -g3 -Wall -Wextra -Wpedantic -Wno-unknown-pragmas -Wnull-dereference -Wparentheses -Wjump-misses-init -Wdouble-promotion -Wshadow -Wformat=2 -Wcast-qual")
		string(APPEND CMAKE_CXX_FLAGS_RELWITHDEBINFO " -g3 -Wall -Wextra -Wpedantic -Wno-unknown-pragmas -Wnull-dereference -Wparentheses -Wold-style-cast -Wdouble-promotion -Wshadow -Wformat=2 -Wcast-qual")
	else() # ICC
		# Set correct optimization and symbol flags for profile builds
		string(APPEND CMAKE_C_FLAGS_RELWITHDEBINFO " -g3 ")
		string(APPEND CMAKE_CXX_FLAGS_RELWITHDEBINFO " -g3 ")

		# Enable warnings
		string(APPEND CMAKE_C_FLAGS_DEBUG " -g3 -Wall -Wextra -Wno-unknown-pragmas -Wparentheses -Wjump-misses-init -Wshadow -Wformat=2 -Wcast-qual")
		string(APPEND CMAKE_CXX_FLAGS_DEBUG " -g3 -Wall -Wextra -Wno-unknown-pragmas -Wparentheses -Wshadow -Wformat=2 -Wcast-qual")
		string(APPEND CMAKE_C_FLAGS_RELWITHDEBINFO " -g3 -Wall -Wextra -Wno-unknown-pragmas -Wparentheses -Wjump-misses-init -Wshadow -Wformat=2 -Wcast-qual")
		string(APPEND CMAKE_CXX_FLAGS_RELWITHDEBINFO " -g3 -Wall -Wextra -Wno-unknown-pragmas -Wparentheses -Wshadow -Wformat=2 -Wcast-qual")

	endif()

	# GCC-specific
	if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
		string(APPEND CMAKE_C_FLAGS_DEBUG " -Wduplicated-cond -Wduplicated-branches -Wlogical-op -Wrestrict ")
		string(APPEND CMAKE_CXX_FLAGS_DEBUG " -Wduplicated-cond -Wduplicated-branches -Wlogical-op -Wrestrict -Wuseless-cast ")
		string(APPEND CMAKE_C_FLAGS_RELWITHDEBINFO " -Wduplicated-cond -Wduplicated-branches -Wlogical-op -Wrestrict ")
		string(APPEND CMAKE_CXX_FLAGS_RELWITHDEBINFO " -Wduplicated-cond -Wduplicated-branches -Wlogical-op -Wrestrict -Wuseless-cast ")
	endif()

	# Clang-specific settings
	if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
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

	# Enable native build
	if(WEIRDLIB_BUILD_NATIVE)
		string(APPEND CMAKE_C_FLAGS " -march=native -mtune=native ")
		string(APPEND CMAKE_CXX_FLAGS " -march=native -mtune=native ")
	endif(WEIRDLIB_BUILD_NATIVE)

endif(CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Intel")

string(APPEND CMAKE_CXX_FLAGS " ${WEIRDLIB_ADDITIONAL_CXX_FLAGS} ")