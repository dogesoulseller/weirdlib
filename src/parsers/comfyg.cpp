#include "../../include/weirdlib_parsers.hpp"
#include "../../include/weirdlib_traits.hpp"
#include "../../include/weirdlib_string.hpp"
#include "../common.hpp"

#include <cstring>
#include <algorithm>
#include <fstream>
#include <vector>
#include <string>
#include <string_view>

using namespace std::string_literals;

namespace wlib::parse
{
	static bool isComment(const std::string& str) noexcept {
		for (const auto c: str) {
			// Check if char is #
			if (c == '#') {
				return true;
			}

			if (isspace(c) == 0) {
				return false;
			}
		}

		return true;
	}

	Comfyg::Comfyg(const std::string& path) {
		std::vector<uint8_t> fileBytes;
		std::ifstream file(path, std::ios::binary | std::ios::ate);
		if (!file.good()) {
			throw std::runtime_error("Comfyg loader: Failed to load file "s + path);
		}

		auto fileSize = file.tellg();
		fileBytes.resize(fileSize);
		file.seekg(0);
		file.read(reinterpret_cast<char*>(fileBytes.data()), fileSize);

		ParseFormat(fileBytes.data(), fileBytes.size());
	}

	Comfyg::Comfyg(const uint8_t* ptr, size_t len) {
		ParseFormat(ptr, len);
	}

	void Comfyg::ParseFormat(const uint8_t* ptr, size_t len) {
		std::vector<std::string> lines;

		// Split into lines
		size_t offset = 0;
		bool isEOL = false;
		const uint8_t* oldNewlinePos = ptr;

		auto newlinePos = std::find(ptr+offset, ptr+len, '\n');
		if (newlinePos == ptr+len) {
			isEOL = true;
		}

		lines.emplace_back(oldNewlinePos, newlinePos);
		oldNewlinePos = newlinePos;
		offset = newlinePos - ptr + 1;

		while(!isEOL) {
			newlinePos = std::find(ptr+offset, ptr+len, '\n');
			if (newlinePos == ptr+len) {
				isEOL = true;
			}

			lines.emplace_back(oldNewlinePos+1, newlinePos);
			oldNewlinePos = newlinePos;
			offset = newlinePos - ptr + 1;
		}

		// Sort lines
		size_t currentLine = 0;
		for (const auto& l: lines) {
			if (isComment(l)) {
				sortedLines.emplace_back(l, ComfygType::Comment);
				currentLine++;
				continue;
			}

			// Tokenize into Name/Type/Value
			auto startKeyPos = l.find_first_not_of(std::string(whitespace.data()));
			auto openBracketPos = l.find('<');
			auto closeBracketPos = l.find('>');
			auto eqSignPos = l.find('=');

			// Detect line with error
			if (openBracketPos == std::string::npos || closeBracketPos == std::string::npos
			|| eqSignPos == std::string::npos || startKeyPos == std::string::npos) {
				auto errMsg = "Failed to tokenize value at line "s + std::to_string(currentLine);
				errors.emplace_back(std::move(errMsg), ParseErrorType::TokenizeFailed);
				currentLine++;
				continue;
			}

			std::string keyStr(l.data()+startKeyPos, l.data()+openBracketPos);
			std::string typeStr(l.data()+openBracketPos+1, l.data()+closeBracketPos);
			std::string valueStr(l.data()+eqSignPos+1, l.data()+l.size());

			// Remove whitespace in key
			keyStr.erase(std::remove_if(keyStr.begin(), keyStr.end(), isspace), keyStr.end());

			// Process value according to type
			typeStr.erase(std::remove_if(typeStr.begin(), typeStr.end(), isspace), typeStr.end());
			std::transform(typeStr.begin(), typeStr.end(), typeStr.begin(), tolower);

			if (typeStr == "integer" || typeStr == "int") {
				valueStr.erase(std::remove_if(valueStr.begin(), valueStr.end(), [](const char c){
					return isspace(c) || c == '_';
				}), valueStr.end());

				int64_t v;
				if (auto success = str::ParseString(valueStr, v); !success) {
					auto errMsg = "Failed to parse integer value at line "s + std::to_string(currentLine);
					errors.emplace_back(std::move(errMsg), ParseErrorType::InvalidValue);
					currentLine++;
					continue;
				}

				values.insert(keyStr, v);
			} else if (typeStr == "float" || typeStr == "decimal") {
				valueStr.erase(std::remove_if(valueStr.begin(), valueStr.end(), isspace), valueStr.end());

				double v;
				if (auto success = str::ParseString(valueStr, v); !success) {
					auto errMsg = "Failed to parse float value at line "s + std::to_string(currentLine);
					errors.emplace_back(std::move(errMsg), ParseErrorType::InvalidValue);
					currentLine++;
					continue;
				}

				values.insert(keyStr, v);
			} else if (typeStr == "string" || typeStr == "str") {
				auto stringStart = valueStr.find_first_of('"');
				auto stringEnd = valueStr.find_last_of('"');

				if (stringStart == std::string::npos || stringEnd == std::string::npos) {
					auto errMsg = "Invalid string at line "s + std::to_string(currentLine);
					errors.emplace_back(std::move(errMsg), ParseErrorType::InvalidValue);
					currentLine++;
					continue;
				}

				std::string stringContents(valueStr.data() + stringStart + 1, valueStr.data() + stringEnd);
				std::string stringOutput;
				stringOutput.reserve(stringContents.size());

				for (size_t i = 0; i < stringContents.size(); i++) {
					if (stringContents[i] == '\\') {
						switch (stringContents[i+1])
						{
						case 'a':
							stringOutput.push_back('\a');
							break;
						case 'b':
							stringOutput.push_back('\b');
							break;
						case 't':
							stringOutput.push_back('\t');
							break;
						case 'n':
							stringOutput.push_back('\n');
							break;
						case 'v':
							stringOutput.push_back('\v');
							break;
						case 'f':
							stringOutput.push_back('\f');
							break;
						case 'r':
							stringOutput.push_back('\r');
							break;
						case '0':
							stringOutput.push_back('\0');
							break;
						case '\\':
							stringOutput.push_back('\\');
							break;
						case '\"':
							stringOutput.push_back('\"');
							break;
						default:
							break;
						}

						i++;
					} else {
						stringOutput.push_back(stringContents[i]);
					}
				}

				stringOutput.shrink_to_fit();
				values.insert(keyStr, stringOutput);
			} else if (typeStr == "boolean" || typeStr == "bool") {
				valueStr.erase(std::remove_if(valueStr.begin(), valueStr.end(), isspace), valueStr.end());

				bool v = str::ParseBool(valueStr);
				values.insert(keyStr, v);
			} else {
				auto errMsg = "Unsupported type "s + typeStr + " at line "s + std::to_string(currentLine);
				errors.emplace_back(std::move(errMsg), ParseErrorType::InvalidType);
				currentLine++;
				continue;
			}

			currentLine++;
		}


	}

} // namespace wlib::parse
