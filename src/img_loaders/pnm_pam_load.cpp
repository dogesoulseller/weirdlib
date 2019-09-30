#include "image_format_loaders.hpp"
#include "img_loader_exceptions.hpp"

#include <memory>
#include <utility>
#include <stdexcept>
#include <cstring>
#include <sstream>
#include <string>
#include <algorithm>
#include <regex>
#include <charconv>
#include "../../include/weirdlib_string.hpp"

static constexpr std::array<char, 10> digits = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'};
static constexpr std::array<char, 4> whitespace = {' ', '\t', '\n', '\r'};

inline static std::vector<uint8_t> GetASCIIPBMPixels(const char* pixin, size_t width, size_t height, size_t maxFileSize) noexcept {
	static const std::regex whitespaceRemoveRegex("\\s+", std::regex_constants::ECMAScript | std::regex_constants::optimize);

	auto pix_nowspace = std::make_unique<char[]>(maxFileSize);
	std::vector<uint8_t> pixels(width*height);

	// Strip any whitespace, leaving only single-digit pixel values
	std::regex_replace(pix_nowspace.get(), pixin, pixin+maxFileSize, whitespaceRemoveRegex, "");

	auto firstDigitLocation = std::find_first_of(pix_nowspace.get(), pix_nowspace.get()+maxFileSize,
				digits.cbegin(), digits.cend());

	// Loop over all digits:
	// 0s and 1s are stored contiguously in memory
	for (size_t offset = 0; offset < width*height; offset++) {
		pixels[offset] = (*(firstDigitLocation+offset)) == '0' ? 255 : 0;
	}

	return pixels;
}

inline static std::vector<uint8_t> GetASCIIPGMPixels(const char* pixin, size_t width, size_t height, size_t maxFileSize) noexcept {
	static const std::regex multiWhitespaceRemoveRegex("\\s{2,}", std::regex_constants::ECMAScript | std::regex_constants::optimize);
	auto pix_nowspace = std::make_unique<char[]>(maxFileSize);
	std::vector<uint8_t> pixels(width*height);

	std::regex_replace(pix_nowspace.get(), pixin, pixin+maxFileSize, multiWhitespaceRemoveRegex, "");

	auto startLocation = std::find_first_of(pix_nowspace.get(), pix_nowspace.get()+maxFileSize,
				digits.cbegin(), digits.cend());

	auto endLocation = std::find(startLocation, startLocation+maxFileSize, '\0');

	// Loop over data until all pixels are processed
	for (size_t offset = 0; offset < width*height; offset++) {
		// Get location of next whitespace character
		auto wspaceLocation = std::find_first_of(startLocation, endLocation, whitespace.cbegin(), whitespace.cend());

		// Convert encountered number to unscaled pixel value
		std::from_chars(startLocation, wspaceLocation, pixels[offset]);

		// Next value starts one whitespace away from current location
		startLocation = wspaceLocation+1;
	}

	return pixels;
}

inline static std::vector<uint8_t> GetASCIIPPMPixels(const char* pixin, size_t width, size_t height, size_t maxFileSize) noexcept {
	static const std::regex multiWhitespaceRemoveRegex("\\s{2,}", std::regex_constants::ECMAScript | std::regex_constants::optimize);
	auto pix_nowspace = std::make_unique<char[]>(maxFileSize);
	std::vector<uint8_t> pixels(width*height*3);

	std::regex_replace(pix_nowspace.get(), pixin, pixin+maxFileSize, multiWhitespaceRemoveRegex, "");

	auto startLocation = std::find_first_of(pix_nowspace.get(), pix_nowspace.get()+maxFileSize,
				digits.cbegin(), digits.cend());

	auto endLocation = std::find(startLocation, startLocation+maxFileSize, '\0');

	// Loop over data until all pixels are processed
	for (size_t offset = 0; offset < width*height; offset++) {
		// Get location of next whitespace character
		auto wspaceLocation = std::find_first_of(startLocation, endLocation, whitespace.cbegin(), whitespace.cend());

		// Convert encountered number to unscaled pixel value
		std::from_chars(startLocation, wspaceLocation, pixels[offset]);

		// Next value starts one whitespace away from current location
		startLocation = wspaceLocation+1;
	}

	return pixels;
}

// TODO: Fix for PBM (per-bit instead of per-byte)
inline static std::vector<uint8_t> GetBinaryPNMPixels(const char* pixin, size_t width, size_t height, uint8_t chans) noexcept {
	std::vector<uint8_t> pixels(width*height*chans);

	for (size_t i = 0; i < pixels.size(); i++) {
		pixels[i] = *(pixin+i);
	}

	return pixels;
}

inline static std::vector<uint8_t> GetBinaryPBMPixels(const char* pixin, size_t width, size_t height, uint8_t chans) noexcept {
	std::vector<uint8_t> pixels(width*height);

	size_t iters = width / 8;
	size_t itersRem = width % 8;

	size_t totalOffset = 0;

	for (size_t h = 0; h < height; h++) {
		for (size_t w = 0; w < iters; w++) {
			const uint8_t tmp = *(pixin+w*h);

			pixels[totalOffset+0] = (tmp & 0b10000000) == 0 ? 255 : 0;
			pixels[totalOffset+1] = (tmp & 0b01000000) == 0 ? 255 : 0;
			pixels[totalOffset+2] = (tmp & 0b00100000) == 0 ? 255 : 0;
			pixels[totalOffset+3] = (tmp & 0b00010000) == 0 ? 255 : 0;
			pixels[totalOffset+4] = (tmp & 0b00001000) == 0 ? 255 : 0;
			pixels[totalOffset+5] = (tmp & 0b00000100) == 0 ? 255 : 0;
			pixels[totalOffset+6] = (tmp & 0b00000010) == 0 ? 255 : 0;
			pixels[totalOffset+7] = (tmp & 0b00000001) == 0 ? 255 : 0;

			totalOffset+=8;
		}

		// Remaining bits
		const uint8_t tmp = *(pixin+iters*h);
		for (size_t i = 0; i < itersRem; i++) {
			pixels[totalOffset+i] = (tmp & (1 << (7-i))) == 0 ? 255 : 0;
			totalOffset++;
		}
	}

	return pixels;
}

namespace wlib::image
{
	ImageInfoPNM LoadPNM(const std::string& path) {
		std::ifstream in(path, std::ios::binary);
		return LoadPNM(in);
	}

	ImageInfoPNM LoadPNM(std::ifstream& in) {
		in.seekg(0, std::ios::end);
		size_t fileSize = in.tellg();
		in.seekg(0);

		auto fileBytes = std::make_unique<uint8_t[]>(fileSize);
		in.read(reinterpret_cast<char*>(fileBytes.get()), fileSize);

		return LoadPNM(fileBytes.get(), fileSize);
	}

	ImageInfoPNM LoadPNM(const uint8_t* in, size_t size) {
		ImageInfoPNM info;
		size_t currentOffset = 0;

		char formatIdentifier_cstr[3];
		std::memcpy(formatIdentifier_cstr, in, 2);
		currentOffset+=2;
		formatIdentifier_cstr[2] = '\0';

		std::string formatIdentifier = formatIdentifier_cstr;

		std::regex commentRemoveRegex("(\\s*#+.*)",
			std::regex_constants::optimize | std::regex_constants::ECMAScript);

		// Make copy of data without comments
		auto in_nocomments = std::make_unique<uint8_t[]>(size+1);

		std::regex_replace(reinterpret_cast<char*>(in_nocomments.get()),
			reinterpret_cast<const char*>(in),
			reinterpret_cast<const char*>(in+size),
			commentRemoveRegex, "");

		// Get width
		auto digitLocation = std::find_first_of(in_nocomments.get()+currentOffset, in_nocomments.get()+currentOffset+128,
			digits.cbegin(), digits.cend());

		auto whitespaceLocation = std::find_first_of(in_nocomments.get()+currentOffset, in_nocomments.get()+currentOffset+128,
			whitespace.cbegin(), whitespace.cend());

		std::from_chars(reinterpret_cast<const char*>(digitLocation), reinterpret_cast<const char*>(whitespaceLocation), info.width);

		currentOffset = whitespaceLocation - in_nocomments.get() + 1;	// Update offset

		// Get height
		digitLocation = std::find_first_of(in_nocomments.get()+currentOffset, in_nocomments.get()+currentOffset+128,
			digits.cbegin(), digits.cend());

		whitespaceLocation = std::find_first_of(in_nocomments.get()+currentOffset, in_nocomments.get()+currentOffset+128,
			whitespace.cbegin(), whitespace.cend());

		std::from_chars(reinterpret_cast<const char*>(digitLocation), reinterpret_cast<const char*>(whitespaceLocation), info.height);

		currentOffset = whitespaceLocation - in_nocomments.get() + 1;	// Update offset

		// Split off into separate logic for PBM, PGM, and PPM
		if (formatIdentifier == "P1") {	// PBM ASCII
			info.colorChannels = 1;
			info.pixels = GetASCIIPBMPixels(reinterpret_cast<const char*>(in_nocomments.get())+currentOffset, info.width, info.height, size-currentOffset+1);

		} else if (formatIdentifier == "P2") {	// PGM ASCII
			info.colorChannels = 1;

			// Get max value
			digitLocation = std::find_first_of(in_nocomments.get()+currentOffset, in_nocomments.get()+currentOffset+128,
			digits.cbegin(), digits.cend());

			whitespaceLocation = std::find_first_of(in_nocomments.get()+currentOffset, in_nocomments.get()+currentOffset+128,
				whitespace.cbegin(), whitespace.cend());

			std::from_chars(reinterpret_cast<const char*>(digitLocation), reinterpret_cast<const char*>(whitespaceLocation), info.maxValue);

			currentOffset = whitespaceLocation - in_nocomments.get() + 1;

			info.pixels = GetASCIIPGMPixels(reinterpret_cast<const char*>(in_nocomments.get())+currentOffset, info.width, info.height, size-currentOffset+1);
		} else if (formatIdentifier == "P3") {	// PPM ASCII
			info.colorChannels = 3;

			// Get max value
			digitLocation = std::find_first_of(in_nocomments.get()+currentOffset, in_nocomments.get()+currentOffset+128,
			digits.cbegin(), digits.cend());

			whitespaceLocation = std::find_first_of(in_nocomments.get()+currentOffset, in_nocomments.get()+currentOffset+128,
				whitespace.cbegin(), whitespace.cend());

			std::from_chars(reinterpret_cast<const char*>(digitLocation), reinterpret_cast<const char*>(whitespaceLocation), info.maxValue);

			currentOffset = whitespaceLocation - in_nocomments.get() + 1;

			info.pixels = GetASCIIPPMPixels(reinterpret_cast<const char*>(in_nocomments.get())+currentOffset, info.width, info.height, size-currentOffset+1);
		} else if (formatIdentifier == "P4") { // PBM Binary
			info.colorChannels = 1;
			info.pixels = GetBinaryPNMPixels(reinterpret_cast<const char*>(in_nocomments.get())+currentOffset, info.width, info.height, info.colorChannels);
		} else if (formatIdentifier == "P5") { // PGM Binary
			info.colorChannels = 1;

			// Get max value
			digitLocation = std::find_first_of(in_nocomments.get()+currentOffset, in_nocomments.get()+currentOffset+128,
			digits.cbegin(), digits.cend());

			whitespaceLocation = std::find_first_of(in_nocomments.get()+currentOffset, in_nocomments.get()+currentOffset+128,
				whitespace.cbegin(), whitespace.cend());

			std::from_chars(reinterpret_cast<const char*>(digitLocation), reinterpret_cast<const char*>(whitespaceLocation), info.maxValue);

			currentOffset = whitespaceLocation - in_nocomments.get() + 1;

			info.pixels = GetBinaryPNMPixels(reinterpret_cast<const char*>(in_nocomments.get())+currentOffset, info.width, info.height, info.colorChannels);
		} else if (formatIdentifier == "P6") { // PPM Binary
			info.colorChannels = 3;

			// Get max value
			digitLocation = std::find_first_of(in_nocomments.get()+currentOffset, in_nocomments.get()+currentOffset+128,
			digits.cbegin(), digits.cend());

			whitespaceLocation = std::find_first_of(in_nocomments.get()+currentOffset, in_nocomments.get()+currentOffset+128,
				whitespace.cbegin(), whitespace.cend());

			std::from_chars(reinterpret_cast<const char*>(digitLocation), reinterpret_cast<const char*>(whitespaceLocation), info.maxValue);

			currentOffset = whitespaceLocation - in_nocomments.get() + 1;

			info.pixels = GetBinaryPNMPixels(reinterpret_cast<const char*>(in_nocomments.get())+currentOffset, info.width, info.height, info.colorChannels);
		}
		return info;
	}


	ImageInfoPAM LoadPAM(const std::string& path) {
		std::ifstream in(path, std::ios::binary);
		return LoadPAM(in);
	}

	ImageInfoPAM LoadPAM(std::ifstream& in) {
		in.seekg(0, std::ios::end);
		size_t fileSize = in.tellg();
		in.seekg(0);

		auto fileBytes = std::make_unique<uint8_t[]>(fileSize);
		in.read(reinterpret_cast<char*>(fileBytes.get()), fileSize);

		return LoadPAM(fileBytes.get(), fileSize);
	}

	ImageInfoPAM LoadPAM(const uint8_t* in, size_t size) {
		ImageInfoPAM info;

		std::regex commentRemoveRegex("(\\s*#+.*)",
			std::regex_constants::optimize | std::regex_constants::ECMAScript);

		// Make modifyable copy of data
		auto in_nocomments = std::make_unique<uint8_t[]>(size+1);

		std::regex_replace(reinterpret_cast<char*>(in_nocomments.get()),
			reinterpret_cast<const char*>(in),
			reinterpret_cast<const char*>(in+size),
			commentRemoveRegex, "");


		auto headerEnd = std::strstr(reinterpret_cast<const char*>(in_nocomments.get()), "ENDHDR")+7;

		char headerBuffer[256];
		std::copy(reinterpret_cast<const char*>(in_nocomments.get()), const_cast<const char*>(headerEnd), headerBuffer);

		// Get position of each parameter's data
		auto widthStr = std::strstr(headerBuffer, "WIDTH") + 6;
		auto heightStr = std::strstr(headerBuffer, "HEIGHT") + 7;
		auto depthStr = std::strstr(headerBuffer, "DEPTH") + 6;
		auto maxvalStr = std::strstr(headerBuffer, "MAXVAL") + 7;
		// auto tTypeStr = std::strstr(headerBuffer, "TUPLTYPE") + 8;

		// Find first whitespace following digit
		auto widthEnd = std::find_first_of(widthStr, widthStr+256, whitespace.cbegin(), whitespace.cend());
		auto heightEnd = std::find_first_of(heightStr, heightStr+256, whitespace.cbegin(), whitespace.cend());
		auto depthEnd = std::find_first_of(depthStr, depthStr+256, whitespace.cbegin(), whitespace.cend());
		auto maxvalEnd = std::find_first_of(maxvalStr, maxvalStr+256, whitespace.cbegin(), whitespace.cend());
		// auto tTypeEnd =

		// Convert ASCII values to ints
		std::from_chars(widthStr, widthEnd, info.width);
		std::from_chars(heightStr, heightEnd, info.height);
		std::from_chars(depthStr, depthEnd, info.colorChannels);
		std::from_chars(maxvalStr, maxvalEnd, info.maxValue);

		info.pixels.resize(info.width * info.height*info.colorChannels);

		if (info.maxValue == 1) {	// PBM-like (1 bit per pixel)
			size_t iters = info.width / 8;
			size_t itersRem = info.width % 8;

			size_t totalOffset = 0;

			for (size_t h = 0; h < info.height; h++) {
				for (size_t w = 0; w < iters; w++) {
					const uint8_t tmp = *(headerEnd+w*h);

					info.pixels[totalOffset+0] = (tmp & 0b10000000) == 0 ? 65535 : 0;
					info.pixels[totalOffset+1] = (tmp & 0b01000000) == 0 ? 65535 : 0;
					info.pixels[totalOffset+2] = (tmp & 0b00100000) == 0 ? 65535 : 0;
					info.pixels[totalOffset+3] = (tmp & 0b00010000) == 0 ? 65535 : 0;
					info.pixels[totalOffset+4] = (tmp & 0b00001000) == 0 ? 65535 : 0;
					info.pixels[totalOffset+5] = (tmp & 0b00000100) == 0 ? 65535 : 0;
					info.pixels[totalOffset+6] = (tmp & 0b00000010) == 0 ? 65535 : 0;
					info.pixels[totalOffset+7] = (tmp & 0b00000001) == 0 ? 65535 : 0;

					totalOffset+=8;
				}

				// Remaining bits
				const uint8_t tmp = *(headerEnd+iters*h);
				for (size_t i = 0; i < itersRem; i++) {
					info.pixels[totalOffset+i] = (tmp & (1 << (7-i))) == 0 ? 65535 : 0;
					totalOffset++;
				}
			}
		} else {	// PAM (PPM-like)
			if (info.maxValue <= 255) { // 8-bit
				for (size_t i = 0; i < info.width * info.height * info.colorChannels; i++) {
					info.pixels[i] = *(headerEnd+i);
				}
			} else { // 16-bit
				auto cvtptr = reinterpret_cast<uint16_t*>(headerEnd);
				for (size_t i = 0; i < info.width * info.height * info.colorChannels; i++) {
					info.pixels[i] = *(cvtptr+i);
				}
			}
		}

		return info;
	}
}
