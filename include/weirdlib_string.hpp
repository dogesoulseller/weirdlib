#pragma once
#include <string>
#include <cstddef>
#include <utility>

namespace wlib
{
	size_t strlen(const char* s) noexcept;

	bool strcmp(const char* str0, const char* str1);
	bool strcmp(const char* str0, const char* str1, const size_t len);

	// strncmp is aliased to strcmp overloaded for a max len parameter
	bool strncmp(const char* str0, const char* str1, const size_t len);
} // wlib
