#include "../../include/weirdlib_string.hpp"
#include "../common.hpp"
#include <vector>
#include <algorithm>
#include <cstring>
#include <numeric>
#include <cmath>
#include <charconv>

inline static bool isDigit(const char c) {
	switch (c)
	{
	  case '0':
	  case '1':
	  case '2':
	  case '3':
	  case '4':
	  case '5':
	  case '6':
	  case '7':
	  case '8':
	  case '9':
		return true;
	  default:
		return false;
	}
}

inline static uint64_t digitToNum(const char c) {
	switch (c)
	{
	  case '0':
		return 0;
	  case '1':
		return 1;
	  case '2':
		return 2;
	  case '3':
		return 3;
	  case '4':
		return 4;
	  case '5':
		return 5;
	  case '6':
		return 6;
	  case '7':
		return 7;
	  case '8':
		return 8;
	  case '9':
		return 9;
	}

	return 0;
}

namespace wlib::str::detail
{
	char _toLowerWorkaround(const char c) {
		return static_cast<char>(std::tolower(c));
	}

	bool parseStringToBool(const std::string& str) {
		std::string strLowercase = str;
		strLowercase.resize(str.size());
		std::transform(str.cbegin(), str.cend(), strLowercase.begin(), _toLowerWorkaround);
		strLowercase.erase(std::remove_if(strLowercase.begin(), strLowercase.end(), [](const char c){
			for (const auto w: whitespace) {
				if (w == c)
					return true;
			}
			return false;
		}), strLowercase.end());

		bool result = strLowercase == "true" || strLowercase == "yes" || strLowercase == "on"
		|| strLowercase == "enable" || strLowercase == "enabled"
		|| strLowercase == "t" || strLowercase == "y";

		return result;
	}

	std::tuple<int64_t, bool> parseStringToInteger(const std::string& str) {
		int64_t out = 0;
		bool encounteredFirst = false;
		std::vector<char> vec;
		vec.reserve(str.size());

		for (size_t i = 0; i < str.size(); i++) {
			if (isDigit(str[i])) {
				encounteredFirst = true;
				vec.push_back(str[i]);
			} else if (str[i] == '-') {
				if (encounteredFirst) {
					break;
				} else {
					encounteredFirst = true;
					vec.push_back(str[i]);
				}
			} else if (str[i] == '+') {
				if (encounteredFirst) {
					break;
				} else {
					encounteredFirst = true;
					vec.push_back(str[i]);
				}
			} else {
				break;
			}
		}

		if (vec.empty()) {
			return std::make_tuple(out, false);
		}

		bool isNegative = vec[0] == '-';
		bool hasSymbol = vec[0] == '-' || vec[0] == '+';

		if (hasSymbol) {
			size_t currentMul = std::pow(10, vec.size()-2);
			for (size_t i = 1; i < vec.size(); i++) {
				out += digitToNum(vec[i]) * currentMul;
				currentMul/=10;
			}
		} else {
			size_t currentMul = std::pow(10, vec.size()-1);
			for (size_t i = 0; i < vec.size(); i++) {
				out += digitToNum(vec[i]) * currentMul;
				currentMul/=10;
			}
		}

		return isNegative ? std::make_tuple(-out, true) : std::make_tuple(out, true);
	}

	std::tuple<uint64_t, bool> parseStringToUinteger(const std::string& str) {
		uint64_t out = 0;
		std::vector<char> vec;
		vec.reserve(str.size());

		for (size_t i = 0; i < str.size(); i++) {
			if (isDigit(str[i])) {
				vec.push_back(str[i]);
			} else {
				break;
			}
		}

		if (vec.empty()) {
			return std::make_tuple(out, false);
		}

		size_t currentMul = std::pow(10, vec.size()-1);
		for (size_t i = 0; i < vec.size(); i++) {
			out += digitToNum(vec[i]) * currentMul;
			currentMul/=10;
		}

		return std::make_tuple(out, true);
	}

	// Temporary
	std::tuple<float, bool> parseStringToFloat(const std::string& str) {
		auto res = std::strtof(str.c_str(), nullptr);
		return std::make_tuple(res, true);
	}

	// Temporary
	std::tuple<double, bool> parseStringToDouble(const std::string& str) {
		auto res = std::strtod(str.c_str(), nullptr);
		return std::make_tuple(res, true);
	}

} // namespace wlib::str::detail
