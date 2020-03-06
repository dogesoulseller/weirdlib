##|--------------------------|##
##| TEST SUITE CONFIGURATION |##
##|--------------------------|##

if(WEIRDLIB_ENABLE_TESTING)
	message(STATUS "Configuring tests")

	enable_testing()
	find_package(GTest REQUIRED)
	include(GoogleTest)

	set(WEIRDLIB_TEST_SOURCES "tests/bitop_tests.cpp" "tests/container_tests.cpp"
		"tests/math_tests.cpp" "tests/simdop_tests.cpp" "tests/test_main.cpp"
	)

	# Cryptography module
	if(WEIRDLIB_ENABLE_BUILD_CRYPTO)
		list(APPEND WEIRDLIB_TEST_SOURCES "tests/crypto_tests.cpp")
	endif(WEIRDLIB_ENABLE_BUILD_CRYPTO)

	# String operations module
	if(WEIRDLIB_ENABLE_BUILD_STRINGOPS)
		list(APPEND WEIRDLIB_TEST_SOURCES "tests/stringop_tests.cpp")
	endif(WEIRDLIB_ENABLE_BUILD_STRINGOPS)

	# Image operations module
	if(WERIDLIB_ENABLE_BUILD_IMAGEOPS)
		list(APPEND WEIRDLIB_TEST_SOURCES "tests/imgloader_tests.cpp" "tests/imgop_tests.cpp")
	endif(WERIDLIB_ENABLE_BUILD_IMAGEOPS)

	# File operations module
	if(WERIDLIB_ENABLE_BUILD_FILEOPS)
		list(APPEND WEIRDLIB_TEST_SOURCES "tests/fileop_tests.cpp")
	endif(WERIDLIB_ENABLE_BUILD_FILEOPS)

	# File parsers module
	if(WERIDLIB_ENABLE_BUILD_PARSERS)
		list(APPEND WEIRDLIB_TEST_SOURCES "tests/parser_tests.cpp")
	endif(WERIDLIB_ENABLE_BUILD_PARSERS)

	# Vector math module
	if(WERIDLIB_ENABLE_BUILD_VECMATH)
		list(APPEND WEIRDLIB_TEST_SOURCES "tests/vecmath_tests.cpp")
	endif(WERIDLIB_ENABLE_BUILD_VECMATH)

	add_executable(weirdlib_tests ${WEIRDLIB_TEST_SOURCES})
	target_link_libraries(weirdlib_tests Threads::Threads weirdlib ${WEIRDLIB_LINKER_LIBRARIES} GTest::GTest)
	set_target_properties(weirdlib_tests PROPERTIES
		CXX_STANDARD 17
		CXX_STANDARD_REQUIRED ON
	)

	target_compile_definitions(weirdlib_tests PUBLIC WLIBTEST_TESTING_DIRECTORY=\"${CMAKE_CURRENT_LIST_DIR}/\")

	gtest_add_tests(TARGET weirdlib_tests SOURCES ${WEIRDLIB_TEST_SOURCES} WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/")

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