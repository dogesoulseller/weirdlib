#include "../../include/weirdlib_string.hpp"
#include <string>
#include <algorithm>

namespace wlib::str
{
	void RemoveLeadingWhitespace(std::string& str) {
		bool encounteredCharacter = false;
		str.erase(std::remove_if(str.begin(), str.end(), [&encounteredCharacter](char c) {
			if (encounteredCharacter) {
				return false;
			}

			if (std::isspace(c) && !encounteredCharacter) {
				return true;
			}

			if (!std::isspace(c)) {
				encounteredCharacter = true;
			}

			return false;
		}), str.end());
	}

	void RemoveAllOccurences(std::string& str, const std::string& needle) {
		auto needleStartPos = wlib::str::strstr(str, needle);
		while(needleStartPos != nullptr) {
			str.erase(str.begin()+(reinterpret_cast<size_t>(needleStartPos)-str.size()),
				str.begin()+(reinterpret_cast<size_t>(needleStartPos)-str.size())+needle.size());

			needleStartPos = wlib::str::strstr(str, needle);
		}
	}

	bool StartsWith(const std::string& str, const std::string& pattern) {
		return std::equal(pattern.cbegin(), pattern.cend(), str.cbegin());
	}

	bool EndsWith(const std::string& str, const std::string& pattern) {
		return std::equal(pattern.cbegin(), pattern.cend(), str.cend()-pattern.size());
	}

	bool StartsWith(const std::string_view& str, const std::string_view& pattern) {
		return std::equal(pattern.cbegin(), pattern.cend(), str.cbegin());
	}

	bool EndsWith(const std::string_view& str, const std::string_view& pattern) {
		return std::equal(pattern.cbegin(), pattern.cend(), str.cend()-pattern.size());
	}

} // namespace wlib::str
