#include "../../include/weirdlib_crypto.hpp"

constexpr const char* Base64DefaultCharset = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

// Implementation based on https://en.wikibooks.org/wiki/Algorithm_Implementation/Miscellaneous/Base64

// Encoder
namespace wlib::crypto
{
	Base64Encoder::Base64Encoder() noexcept : encodeLookup(Base64DefaultCharset), padCharacter{'='} {}

	Base64Encoder::Base64Encoder(const std::string& charset, char padding) : encodeLookup(charset), padCharacter{padding} {
		if (charset.size() < 64) {
			throw std::runtime_error("Not enough chars in charset to encode Base64");
		}
	}

	void Base64Encoder::reset() noexcept {
		buffer.clear();
		buffer.shrink_to_fit();
	}

	void Base64Encoder::update(const std::string& str) {
		update(str.cbegin(), str.cend());
	}

	std::string Base64Encoder::finalize() const noexcept {
		if (buffer.empty()) {
			return "";
		}

		std::string encodedString;
		encodedString.reserve(((buffer.size()/3) + (buffer.size() % 3 > 0)) * 4);
		uint32_t temp;

		auto cursor = buffer.begin();

		for (size_t idx = 0; idx < buffer.size()/3; idx++) {
			temp  = (*cursor++) << 16;
			temp += (*cursor++) << 8;
			temp += (*cursor++);

			char tmp[4] {
				encodeLookup[(temp & 0x00FC0000) >> 18],
				encodeLookup[(temp & 0x0003F000) >> 12],
				encodeLookup[(temp & 0x00000FC0) >> 6],
				encodeLookup[(temp & 0x0000003F)]
			};

			encodedString.append(tmp, tmp+4);
		}

		if (buffer.size() % 3 == 1) {
			temp  = (*cursor++) << 16;

			char tmp[4] {
				encodeLookup[(temp & 0x00FC0000) >> 18],
				encodeLookup[(temp & 0x0003F000) >> 12],
				padCharacter, padCharacter
			};

			encodedString.append(tmp, tmp+4);
		} else if (buffer.size() % 3 == 2) {
			temp  = (*cursor++) << 16;
			temp += (*cursor++) << 8;

			char tmp[4] {
				encodeLookup[(temp & 0x00FC0000) >> 18],
				encodeLookup[(temp & 0x0003F000) >> 12],
				encodeLookup[(temp & 0x00000FC0) >> 6],
				padCharacter
			};

			encodedString.append(tmp, tmp+4);
		}

		return encodedString;
	}
} // namespace wlib::crypto

// Decoder
namespace wlib::crypto
{
	Base64Decoder::Base64Decoder(bool useURLAlphabet, char padChar) noexcept :
		cursorVals{useURLAlphabet ? static_cast<uint8_t>(0x2D) : static_cast<uint8_t>(0x3E),
		useURLAlphabet ? static_cast<uint8_t>(0x5F) : static_cast<uint8_t>(0x3F)},
		padCharacter{padChar} {
	}

	std::vector<uint8_t> Base64Decoder::decode(const std::string& input) const {
		if (input.length() % 4) {
			throw std::runtime_error("Invalid Base64 string");
		}

		size_t padding = 0;

		if (input.length()) {
			if (input[input.length()-1] == padCharacter) {
				padding++;
			}

			if (input[input.length()-2] == padCharacter) {
				padding++;
			}
		}

		std::vector<uint8_t> decodedBytes;
		decodedBytes.reserve(((input.length()/4)*3) - padding);

		uint32_t temp = 0;
		auto cursor = input.cbegin();

		while (cursor < input.cend()) {
			for (size_t qPos = 0; qPos < 4; qPos++) {
				temp <<= 6;
				if (*cursor >= 0x41 && *cursor <= 0x5A) {
					temp |= *cursor - 0x41;
				} else if (*cursor >= 0x61 && *cursor <= 0x7A) {
					temp |= *cursor - 0x47;
				} else if (*cursor >= 0x30 && *cursor <= 0x39) {
					temp |= *cursor + 0x04;
				} else if (*cursor == 0x2B) {
					temp |= cursorVals[0];
				} else if (*cursor == 0x2F) {
					temp |= cursorVals[1];
				} else if (*cursor == padCharacter) {
					switch(input.end() - cursor) {
					case 1:
						decodedBytes.push_back((temp >> 16) & 0x000000FF);
						decodedBytes.push_back((temp >> 8) & 0x000000FF);
						return decodedBytes;
					case 2:
						decodedBytes.push_back((temp >> 10) & 0x000000FF);
						return decodedBytes;
					default:
						throw std::runtime_error("Invalid padding");
					}
				} else {
					throw std::runtime_error("Invalid character");
				}

				cursor++;
			}

			decodedBytes.push_back((temp >> 16) & 0x000000FF);
			decodedBytes.push_back((temp >> 8) & 0x000000FF);
			decodedBytes.push_back(temp & 0x000000FF);
		}

		return decodedBytes;
	}
} // namespace wlib::crypto
