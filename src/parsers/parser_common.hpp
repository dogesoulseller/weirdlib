#pragma once
#ifdef WEIRDLIB_ENABLE_FILE_PARSERS
#if !defined(WEIRDLIB_ENABLE_STRING_OPERATIONS)
	#error "File parsers module requires the string operations module"
#endif

#include "../../include/weirdlib_string.hpp"

#include <cstring>
#include <algorithm>
#include <cctype>
#include <string>

namespace wlib::parse::common
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

	inline void RemoveAllOccurences(std::string& str, const std::string& needle) {
		auto needleStartPos = wlib::str::strstr(str, needle);
		while(needleStartPos != nullptr) {
			str.erase(str.begin()+(reinterpret_cast<size_t>(needleStartPos)-str.size()),
				str.begin()+(reinterpret_cast<size_t>(needleStartPos)-str.size())+needle.size());

			needleStartPos = wlib::str::strstr(str, needle);
		}
	}

} // namespace wlib::parse::common
#endif
