#pragma once
#include <string>
#include <cstddef>

namespace wlib
{

/// Operations on strings that mimic the C standard library <br>
/// In some cases, they might be faster than the standard library's implementation <br>
/// In other cases, they might be slower or equal
namespace str
{
	/// Get length of string
	/// @param s pointer to start of string
	/// @return size of string
	size_t strlen(const char* s);

	/// Compare two strings
	/// @param str0 first string
	/// @param str1 second string
	/// @return true if strings are identical, false otherwise
	bool strcmp(const char* str0, const char* str1);

	/// Compare two strings up to **len** characters
	/// @param str0 first string
	/// @param str1 second string
	/// @param len maximum length of comparison
	/// @return true if strings are identical, false otherwise
	bool strcmp(const char* str0, const char* str1, size_t len);

	/// Alias for {@link strcmp(const char*, const char*, size_t) strcmp}
	bool strncmp(const char* str0, const char* str1, size_t len);

	/// Find substring needle in str
	/// @param str string to search
	/// @param needle substring to search for
	/// @param strLen length of string to search (if set to 0, determined in-function)
	/// @param needleLen length of substring to search for (if set to 0, determined in-function)
	/// @return pointer to first occurence of substring in string
	const char* strstr(const char* str, const char* needle, size_t strLen = 0, size_t needleLen = 0);

	/// Find substring needle in str
	/// @param str string to search
	/// @param needle substring to search for
	/// @param strLen length of string to search (if set to 0, determined in-function)
	/// @param needleLen length of substring to search for (if set to 0, determined in-function)
	/// @return pointer to first occurence of substring in string
	char* strstr(char* str, const char* needle, size_t strLen = 0, size_t needleLen = 0);

} // namespace str
} // wlib
