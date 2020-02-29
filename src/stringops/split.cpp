#ifdef WEIRDLIB_ENABLE_STRING_OPERATIONS
#include "../../include/weirdlib_string.hpp"
#include "../common.hpp"
#include <algorithm>
#include <cstring>
#include <string_view>
#include <sstream>
#include <iomanip>

namespace wlib::str
{
	inline static bool isInvalidStrSlice(const std::string& str, char delimiter) noexcept {
		return str.empty() || (str.length() == 1 && str[0] == delimiter);
	}

	inline static bool isInvalidStrSlice(const std::string& str, const std::string& delimiter) noexcept {
		return str.empty() || (str.length() == 1 && wlib::util::EqualToOneOf(str[0], delimiter.cbegin(), delimiter.cend()));
	}

	std::vector<std::string> SplitAt(const std::string& str, char delimiter) {
		std::vector<size_t> breakPoints;
		breakPoints.reserve(str.size());

		// Collect break points (i.e. delimiter indices)
		for (size_t i = 0; i < str.size(); i++) {
			if (str[i] == delimiter) {
				breakPoints.push_back(i);
			}
		}

		if (breakPoints.empty()) {
			return std::vector<std::string>();
		}

		// Number of substrings is usually equal to delimiters +1
		std::vector<std::string> output;
		output.reserve(breakPoints.size()+1);

		size_t startOfNext = 0;
		for (const auto point : breakPoints) {
			std::string out = str.substr(startOfNext, point-startOfNext);

			startOfNext = point+1;

			if (isInvalidStrSlice(out, delimiter)) {
				continue;
			}

			output.push_back(out);
		}

		if (std::string final = str.substr(startOfNext); !isInvalidStrSlice(final, delimiter)) {
			output.emplace_back(final);
		}

		return output;
	}

	std::pair<std::string, std::string> SplitOnce(const std::string& str, char delimiter) {
		size_t pos;
		for (pos = 0; pos < str.size(); pos++) {
			if (str[pos] == delimiter) {
				break;
			}
		}

		if (pos == str.size()) {
			return std::make_pair(str, "");
		}

		return std::make_pair(str.substr(0, pos), str.substr(pos+1));
	}

	std::vector<std::string> SplitAt(const std::string& str, const std::string& delimiter) {
		std::vector<size_t> breakPoints;
		breakPoints.reserve(str.size());

		// Collect break points (i.e. delimiter indices)
		for (size_t i = 0; i < str.size(); i++) {
			if (wlib::util::EqualToOneOf(str[i], delimiter.cbegin(), delimiter.cend())) {
				breakPoints.push_back(i);
			}
		}

		if (breakPoints.empty()) {
			return std::vector<std::string>();
		}

		// Number of substrings is usually equal to delimiters +1
		std::vector<std::string> output;
		output.reserve(breakPoints.size()+1);

		size_t startOfNext = 0;
		for (const auto point : breakPoints) {
			std::string out = str.substr(startOfNext, point-startOfNext);

			startOfNext = point+1;

			if (isInvalidStrSlice(out, delimiter)) {
				continue;
			}

			output.push_back(out);
		}

		if (std::string final = str.substr(startOfNext); !isInvalidStrSlice(final, delimiter)) {
			output.emplace_back(final);
		}

		return output;
	}

	std::pair<std::string, std::string> SplitOnce(const std::string& str, const std::string& delimiter) {
		size_t pos;
		for (pos = 0; pos < str.size(); pos++) {
			if (wlib::util::EqualToOneOf(str[pos], delimiter.cbegin(), delimiter.cend())) {
				break;
			}
		}

		if (pos == str.size()) {
			return std::make_pair(str, "");
		}

		return std::make_pair(str.substr(0, pos), str.substr(pos+1));
	}

	std::vector<std::string> ToLines(const std::string& str) {
		return SplitAt(str, '\n');
	}

} // namespace wlib::str

#endif