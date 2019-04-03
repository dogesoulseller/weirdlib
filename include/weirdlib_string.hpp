#pragma once
#include <string>
#include <cstddef>
#include <utility>

namespace wlib
{
	size_t strlen(const char* s) noexcept;

	bool strcmp(const std::string& str0, const std::string& str1);
	bool strcmp(const std::string& str0, const std::string& str1, const size_t len);

	// strncmp is aliased to strcmp overloaded for a max len parameter
	template<typename SType>
	bool strncmp(SType&& str0, SType&& str1, const size_t len) {
		return wlib::strcmp(std::forward<SType>(str0), std::forward<SType>(str1), len);
	}
} // wlib
