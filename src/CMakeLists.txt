message(STATUS "Configuring main library")

# Conditionally enable building cryptography module
if(WEIRDLIB_ENABLE_BUILD_CRYPTO)
	add_compile_definitions(WEIRDLIB_ENABLE_CRYPTOGRAPHY)
endif(WEIRDLIB_ENABLE_BUILD_CRYPTO)

# Conditionally enable building string operations module
if(WEIRDLIB_ENABLE_BUILD_STRINGOPS)
	add_compile_definitions(WEIRDLIB_ENABLE_STRING_OPERATIONS)
endif(WEIRDLIB_ENABLE_BUILD_STRINGOPS)

# Conditionally enable building image operations module
if(WERIDLIB_ENABLE_BUILD_IMAGEOPS)
	add_compile_definitions(WEIRDLIB_ENABLE_IMAGE_OPERATIONS)
endif(WERIDLIB_ENABLE_BUILD_IMAGEOPS)

# Conditionally enable building stress test module
if(WERIDLIB_ENABLE_BUILD_ANXIETY)
	add_compile_definitions(WEIRDLIB_ENABLE_ANXIETY)
endif(WERIDLIB_ENABLE_BUILD_ANXIETY)

# Conditionally enable use of prefetch instructions
if(WEIRDLIB_ENABLE_PREFETCH)
	add_compile_definitions(WLIB_ENABLE_PREFETCH)
endif(WEIRDLIB_ENABLE_PREFETCH)

# Library sources and headers
file(GLOB_RECURSE WEIRDLIB_SOURCES "../include/*.hpp" "./*.c*" "./*.h*")

# Primary build target
add_library(weirdlib ${WEIRDLIB_SOURCES})
set_target_properties(weirdlib PROPERTIES
	CXX_STANDARD 17
	CXX_STANDARD_REQUIRED ON
)
