# Cryptography module
if(WEIRDLIB_ENABLE_BUILD_CRYPTO)
	file(GLOB_RECURSE WEIRDLIB_CRYPTO_SOURCES "src/crypto/*.*")
	list(APPEND WEIRDLIB_SOURCES ${WEIRDLIB_CRYPTO_SOURCES})
	list(APPEND WEIRDLIB_INCLUDE_SOURCES "include/weirdlib_crypto.hpp")
	add_compile_definitions(WEIRDLIB_ENABLE_CRYPTOGRAPHY)
endif(WEIRDLIB_ENABLE_BUILD_CRYPTO)

# String operations module
if(WEIRDLIB_ENABLE_BUILD_STRINGOPS)
	file(GLOB_RECURSE WEIRDLIB_STRINGOPS_SOURCES "src/stringops/*.*")
	list(APPEND WEIRDLIB_SOURCES ${WEIRDLIB_STRINGOPS_SOURCES})
	list(APPEND WEIRDLIB_INCLUDE_SOURCES "include/weirdlib_string.hpp")
	add_compile_definitions(WEIRDLIB_ENABLE_STRING_OPERATIONS)
endif(WEIRDLIB_ENABLE_BUILD_STRINGOPS)

# Image operations module
if(WERIDLIB_ENABLE_BUILD_IMAGEOPS)
	file(GLOB_RECURSE WEIRDLIB_IMAGEOP_SOURCES "src/img_process/*.*" "src/img_loaders/*.*")
	list(APPEND WEIRDLIB_SOURCES ${WEIRDLIB_IMAGEOP_SOURCES})
	list(APPEND WEIRDLIB_INCLUDE_SOURCES "include/weirdlib_image.hpp")
	add_compile_definitions(WEIRDLIB_ENABLE_IMAGE_OPERATIONS)
endif(WERIDLIB_ENABLE_BUILD_IMAGEOPS)

# File operations module
if(WERIDLIB_ENABLE_BUILD_FILEOPS)
	file(GLOB_RECURSE WEIRDLIB_FILEOPS_SOURCES "src/file_ops/*.*")
	list(APPEND WEIRDLIB_SOURCES ${WEIRDLIB_FILEOPS_SOURCES})
	list(APPEND WEIRDLIB_INCLUDE_SOURCES "include/weirdlib_fileops.hpp")
	add_compile_definitions(WEIRDLIB_ENABLE_FILE_OPERATIONS)
endif(WERIDLIB_ENABLE_BUILD_FILEOPS)

# File parsers module
if(WERIDLIB_ENABLE_BUILD_PARSERS)
	file(GLOB_RECURSE WEIRDLIB_FILEPARSE_SOURCES "src/parsers/*.*")
	list(APPEND WEIRDLIB_SOURCES ${WEIRDLIB_FILEPARSE_SOURCES})
	list(APPEND WEIRDLIB_INCLUDE_SOURCES "include/weirdlib_parsers.hpp")
	add_compile_definitions(WEIRDLIB_ENABLE_FILE_PARSERS)
endif(WERIDLIB_ENABLE_BUILD_PARSERS)

# Vector math module
if(WERIDLIB_ENABLE_BUILD_VECMATH)
	file(GLOB_RECURSE WEIRDLIB_VECMATH_SOURCES "src/vecmath/*.*")
	list(APPEND WEIRDLIB_SOURCES ${WEIRDLIB_VECMATH_SOURCES})
	list(APPEND WEIRDLIB_INCLUDE_SOURCES "include/weirdlib_vecmath.hpp")
	add_compile_definitions(WEIRDLIB_ENABLE_VECTOR_MATH)
endif(WERIDLIB_ENABLE_BUILD_VECMATH)

# Remaining files
list(APPEND WEIRDLIB_INCLUDE_SOURCES
	"include/weirdlib_bitops.hpp" "include/weirdlib_containers.hpp" "include/weirdlib_math.hpp"
	"include/weirdlib_simdhelper.hpp" "include/weirdlib_traits.hpp"
)