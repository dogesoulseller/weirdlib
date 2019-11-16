#pragma once
#include <string>
#include <cstddef>
#include <tuple>

namespace wlib
{

/// Operations on strings that mimic the C standard library <br>
/// In some cases, they might be faster than the standard library's implementation <br>
/// In other cases, they might be slower or equal
namespace str
{
namespace detail
{
	bool parseStringToBool(const std::string& str);
	std::tuple<int64_t, bool> parseStringToInteger(const std::string& str);
	std::tuple<uint64_t, bool> parseStringToUinteger(const std::string& str);
	std::tuple<float, bool> parseStringToFloat(const std::string& str);
	std::tuple<double, bool> parseStringToDouble(const std::string& str);
} // namespace detail

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

	/// Find substring needle in str
	/// @param str string to search
	/// @param needle substring to search for
	/// @param strLen length of string to search (if set to 0, determined in-function)
	/// @param needleLen length of substring to search for (if set to 0, determined in-function)
	/// @return pointer to first occurence of substring in string
	const char* strstr(const std::string& str, const std::string& needle, size_t strLen = 0, size_t needleLen = 0);

	/// Parse string as boolean <br>
	/// Case-insensitive, detects y, yes, t, true, on, enable, enabled; all others are false
	inline bool ParseBool(const std::string& str) {
		return detail::parseStringToBool(str);
	}

	/// Parse string as OutValT using base 10 <br>
	/// This version is specialized for unsigned integers
	/// @param str input
	/// @param out output value
	/// @return true if no issues, false otherwise
	template<typename OutValT>
	std::enable_if_t<std::is_integral_v<OutValT> && std::is_unsigned_v<OutValT>, bool>
	ParseString(const std::string& str, OutValT& out) {
		const auto& [output, success] = detail::parseStringToUinteger(str);
		if (!success) {
			return false;
		} else {
			out = static_cast<OutValT>(output);
			return true;
		}
	}

	/// Parse string as OutValT using base 10 <br>
	/// This version is specialized for unsigned integers
	/// @param str input
	/// @param out output value
	/// @return true if no issues, false otherwise
	template<typename OutValT>
	std::enable_if_t<std::is_integral_v<OutValT> && std::is_signed_v<OutValT>, bool>
	ParseString(const std::string& str, OutValT& out) {
		const auto& [output, success] = detail::parseStringToInteger(str);
		if (!success) {
			return false;
		} else {
			out = static_cast<OutValT>(output);
			return true;
		}
	}

	/// Parse string as OutValT <br>
	/// This version is specialized for floats
	/// Temporarily uses standard lib functions internally
	/// @param str input
	/// @param out output value
	/// @return true if no issues, false otherwise
	template<typename OutValT>
	std::enable_if_t<std::is_same_v<OutValT, float>, bool>
	ParseString(const std::string& str, OutValT& out) {
		const auto& [output, success] = detail::parseStringToFloat(str);
		if (!success) {
			return false;
		} else {
			out = output;
			return true;
		}
	}

	/// Parse string as OutValT <br>
	/// This version is specialized for doubles <br>
	/// Temporarily uses standard lib functions internally
	/// @param str input
	/// @param out output value
	/// @return true if no issues, false otherwise
	template<typename OutValT>
	std::enable_if_t<!std::is_same_v<OutValT, float> && std::is_floating_point_v<OutValT>, bool>
	ParseString(const std::string& str, OutValT& out) {
		const auto& [output, success] = detail::parseStringToDouble(str);
		if (!success) {
			return false;
		} else {
			out = static_cast<OutValT>(output);
			return true;
		}
	}

	template<typename OutValT>
	std::enable_if_t<!std::is_floating_point_v<OutValT> && !std::is_integral_v<OutValT>, bool>
	ParseString(const std::string& /*str*/, OutValT& /*out*/) = delete;

} // namespace str
} // wlib
