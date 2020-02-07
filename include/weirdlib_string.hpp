#pragma once
#include <cstddef>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>
#include <utility>
#include <sstream>
#include <iomanip>

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

	/// Split string into multiple strings at every **delimiter**
	/// @param delimiter char to use for splitting
	/// @return vector of strings resulting from split
	std::vector<std::string> SplitAt(const std::string& str, char delimiter);

	/// Split string into multiple strings at every element of **delimiters**
	/// @param delimiters chars to use for splitting
	/// @return vector of strings resulting from split
	std::vector<std::string> SplitAt(const std::string& str, const std::string& delimiters);

	/// Split string into two strings at first **delimiter**
	/// @param str string to split
	/// @param delimiter char to use for splitting
	/// @return pair of strings resulting from split
	std::pair<std::string, std::string> SplitOnce(const std::string& str, char delimiter);

	/// Split string into two strings at first element of **delimiters**
	/// @param str string to split
	/// @param delimiters chars to use for splitting
	/// @return pair of strings resulting from split
	std::pair<std::string, std::string> SplitOnce(const std::string& str, const std::string& delimiters);

	/// Split string by newline (LF, not CRLF)
	/// @return vector of resulting lines
	std::vector<std::string> ToLines(const std::string& str);

	/// Join strings with newlines
	/// @param lines lines of text
	/// @return string containing lines joined with Line Feed characters
	template<typename IterT>
	std::string FromLines(IterT start, IterT end) {
		std::stringstream out;
		out << std::noskipws;

		while (start != end) {
			out << *start << '\n';
			++start;
		}

		auto outstr = out.str();
		outstr.pop_back();

		return outstr;
	}

	/// Get length of string
	/// @param s pointer to start of string
	/// @return size of string
	size_t strlen(const char* s);

	/// Compare two strings
	/// @return true if strings are identical, false otherwise
	bool strcmp(const char* str0, const char* str1);

	/// Compare two strings up to **len** characters
	/// @param len maximum length of comparison
	/// @return true if strings are identical, false otherwise
	bool strcmp(const char* str0, const char* str1, size_t len);

	/// Alias for {@link strcmp(const char*, const char*, size_t) strcmp}
	bool strncmp(const char* str0, const char* str1, size_t len);

	/// Find substring needle in str
	/// @param str string to search in
	/// @param needle substring to search for
	/// @param strLen length of string to search (if set to 0, determined in-function)
	/// @param needleLen length of substring to search for (if set to 0, determined in-function)
	/// @return pointer to first occurence of substring in string
	const char* strstr(const char* str, const char* needle, size_t strLen = 0, size_t needleLen = 0);

	/// Find substring needle in str
	/// @param str string to search in
	/// @param needle substring to search for
	/// @param strLen length of string to search (if set to 0, determined in-function)
	/// @param needleLen length of substring to search for (if set to 0, determined in-function)
	/// @return pointer to first occurence of substring in string
	char* strstr(char* str, const char* needle, size_t strLen = 0, size_t needleLen = 0);

	/// Find substring needle in str
	/// @param str string to search in
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
	/// This version is specialized for floats <br>
	/// Temporarily uses standard lib functions internally
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
