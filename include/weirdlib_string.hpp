#pragma once
#include <string>
#include <cstddef>
#include <utility>

namespace wlib
{
namespace str
{
	/// Get length of string
	/// @param s pointer to start of string
	/// @return size of string
	size_t strlen(const char* s) noexcept;

	/// Compare two strings
	/// @param str0 first string
	/// @param str1 second string
	/// @return true if strings are identical, false otherwise
	bool strcmp(const char* str0, const char* str1);

	/// Compare two strings up to **len** characters
	/// @param str0 first string
	/// @param str1 second string
	/// @param maximum length of comparison
	/// @return true if strings are identical, false otherwise
	bool strcmp(const char* str0, const char* str1, const size_t len);

	/// Alias for {@link strcmp(const char*, const char*, const size_t) strcmp}
	bool strncmp(const char* str0, const char* str1, const size_t len);

} // namespace str
} // wlib
