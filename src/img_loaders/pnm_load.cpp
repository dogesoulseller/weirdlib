#include "image_format_loaders.hpp"
#include "img_loader_exceptions.hpp"

#include <tuple>
#include <array>
#include <memory>
#include <utility>
#include <stdexcept>
#include <cstring>
#include <string>
#include <algorithm>
#include <regex>
#include <charconv>
#include <type_traits>

#include "../../include/weirdlib_bitops.hpp"
#include "../../include/weirdlib_math.hpp"
#include "../common.hpp"

inline static std::vector<uint8_t> GetASCIIPBMPixels(const char* pixin, size_t width, size_t height, size_t maxFileSize) noexcept {
	static const std::regex whitespaceRemoveRegex(R"(\s+)", std::regex_constants::ECMAScript | std::regex_constants::optimize);

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
	static const std::regex multiWhitespaceRemoveRegex(R"(\s{2,})", std::regex_constants::ECMAScript | std::regex_constants::optimize);
	auto pix_nowspace = std::make_unique<char[]>(maxFileSize);
	std::vector<uint8_t> pixels(width*height);

	std::regex_replace(pix_nowspace.get(), pixin, pixin+maxFileSize, multiWhitespaceRemoveRegex, " ");

	auto startLocation = std::find_first_of(pix_nowspace.get(), pix_nowspace.get()+maxFileSize,
				digits.cbegin(), digits.cend());

	// Loop over data until all pixels are processed
	for (size_t offset = 0; offset < width*height; offset++) {
		// Get location of next whitespace character
		auto wspaceLocation = std::find_first_of(startLocation, startLocation+4, whitespace.cbegin(), whitespace.cend());

		// Convert encountered number to unscaled pixel value
		std::from_chars(startLocation, wspaceLocation, pixels[offset]);

		// Next value starts one whitespace away from current location
		startLocation = wspaceLocation+1;
	}

	return pixels;
}

inline static std::vector<uint8_t> GetASCIIPPMPixels(const char* pixin, size_t width, size_t height, size_t maxFileSize) noexcept {
	static const std::regex multiWhitespaceRemoveRegex(R"(\s\s+)", std::regex_constants::ECMAScript | std::regex_constants::optimize);
	auto pix_nowspace = std::make_unique<char[]>(maxFileSize);
	std::vector<uint8_t> pixels(width*height*3);

	std::regex_replace(pix_nowspace.get(), pixin, pixin+maxFileSize, multiWhitespaceRemoveRegex, " ");

	auto startLocation = std::find_first_of(pix_nowspace.get(), pix_nowspace.get()+maxFileSize,
				digits.cbegin(), digits.cend());

	// Loop over data until all pixels are processed
	for (size_t offset = 0; offset < width*height*3; offset++) {
		// Get location of next whitespace character
		auto wspaceLocation = std::find_first_of(startLocation, startLocation+4, whitespace.cbegin(), whitespace.cend());

		// Convert encountered number to unscaled pixel value
		std::from_chars(startLocation, wspaceLocation, pixels[offset]);

		// Next value starts one whitespace away from current location
		startLocation = wspaceLocation+1;
	}

	return pixels;
}

inline static std::vector<uint8_t> GetBinaryPNMPixels(const char* pixin, size_t width, size_t height, uint8_t chans) noexcept {
	std::vector<uint8_t> pixels(width*height*chans);

	for (size_t i = 0; i < pixels.size(); i++) {
		pixels[i] = *(pixin+i);
	}

	return pixels;
}

inline static std::vector<uint8_t> GetBinaryPBMPixels(const char* pixin, size_t width, size_t height) noexcept {
	std::vector<uint8_t> pixels(width*height);

	size_t iters = width / 8;
	size_t itersRem = width % 8;

	size_t totalOffset = 0;

	for (size_t h = 0; h < height; h++) {	// For each row
		for (size_t b = 0; b < iters; b++) {	// For each full byte
			const uint8_t val = *(pixin + (h*(iters+1)) + b);

			pixels[totalOffset+0] = wlib::bop::test(val, 7) ? 0 : 255;
			pixels[totalOffset+1] = wlib::bop::test(val, 6) ? 0 : 255;
			pixels[totalOffset+2] = wlib::bop::test(val, 5) ? 0 : 255;
			pixels[totalOffset+3] = wlib::bop::test(val, 4) ? 0 : 255;
			pixels[totalOffset+4] = wlib::bop::test(val, 3) ? 0 : 255;
			pixels[totalOffset+5] = wlib::bop::test(val, 2) ? 0 : 255;
			pixels[totalOffset+6] = wlib::bop::test(val, 1) ? 0 : 255;
			pixels[totalOffset+7] = wlib::bop::test(val, 0) ? 0 : 255;

			totalOffset += 8;
		}

		const uint8_t val = *(pixin + (h*(iters+1)) + iters);
		for (size_t i = 0; i < itersRem; i++) {	// For each bit in last byte
			pixels[totalOffset] = wlib::bop::test(val, 7-i) ? 0 : 255;
			totalOffset++;
		}
	}


	return pixels;
}

using namespace std::string_literals;

namespace wlib::image
{
	ImageInfoPNM LoadPNM(const std::string& path) {
		std::ifstream in(path, std::ios::binary);
		if (!in.good()) {
			throw except::file_open_error("PNM Loader: Failed to open file "s + path);
		}

		return LoadPNM(in);
	}

	ImageInfoPNM LoadPNM(std::ifstream& in) {
		in.seekg(0, std::ios::end);
		size_t fileSize = in.tellg();
		if (fileSize == 0) {
			throw except::file_open_error("PNM Loader: File size returned was 0");
		}

		in.seekg(0);

		auto fileBytes = std::make_unique<uint8_t[]>(fileSize);
		in.read(reinterpret_cast<char*>(fileBytes.get()), fileSize);

		return LoadPNM(fileBytes.get(), fileSize);
	}

	ImageInfoPNM LoadPNM(const uint8_t* in, size_t size) {
		if (in == nullptr) {
			throw except::invalid_image_data("PNM Loader: Image pointer was null");
		}

		ImageInfoPNM info;
		size_t currentOffset = 0;

		std::array<char, 3> formatIdentifier_cstr;
		std::memcpy(formatIdentifier_cstr.data(), in, 2);
		currentOffset+=2;
		formatIdentifier_cstr[2] = '\0';

		const std::string formatIdentifier(formatIdentifier_cstr.data());

		std::regex commentRemoveRegex(R"reg((\s*#+.*))reg",
			std::regex_constants::optimize | std::regex_constants::ECMAScript);

		// Make copy of data without comments
		auto in_nocomments = std::make_unique<uint8_t[]>(size+1);
		std::regex_replace(reinterpret_cast<char*>(in_nocomments.get()),
			reinterpret_cast<const char*>(in),
			reinterpret_cast<const char*>(in+size),
			commentRemoveRegex, "");

		// Get width
		const auto& [IGNORESB, IGNORESB, tmpWidth] = GetNextNumber<size_t>(in_nocomments.get(), currentOffset);
		info.width = tmpWidth;
		if (info.width == 0) {
			throw except::invalid_image_data("PNM Loader: Image width must not be 0");
		}

		// Get height
		const auto& [IGNORESB, IGNORESB, tmpHeight] = GetNextNumber<size_t>(in_nocomments.get(), currentOffset);
		info.height = tmpHeight;
		if (info.height == 0) {
			throw except::invalid_image_data("PNM Loader: Image height must not be 0");
		}

		// Split off into separate logic for PBM, PGM, and PPM
		if (formatIdentifier == "P1") {	// PBM ASCII
			info.colorChannels = 1;
			info.maxValue = 255;
			info.pixels = GetASCIIPBMPixels(reinterpret_cast<const char*>(in_nocomments.get())+currentOffset,
				info.width, info.height, size-currentOffset+1);

		} else if (formatIdentifier == "P2") {	// PGM ASCII
			info.colorChannels = 1;

			// Get max value
			const auto& [IGNORESB, IGNORESB, tmpMaxVal] = GetNextNumber<size_t>(in_nocomments.get(), currentOffset);
			info.maxValue = tmpMaxVal;

			info.pixels = GetASCIIPGMPixels(reinterpret_cast<const char*>(in_nocomments.get())+currentOffset,
				info.width, info.height, size-currentOffset+1);

		} else if (formatIdentifier == "P3") {	// PPM ASCII
			info.colorChannels = 3;

			// Get max value
			const auto& [IGNORESB, IGNORESB, tmpMaxVal] = GetNextNumber<size_t>(in_nocomments.get(), currentOffset);
			info.maxValue = tmpMaxVal;

			info.pixels = GetASCIIPPMPixels(reinterpret_cast<const char*>(in_nocomments.get())+currentOffset,
				info.width, info.height, size-currentOffset+1);

		} else if (formatIdentifier == "P4") {	// PBM Binary
			info.colorChannels = 1;
			info.maxValue = 255;
			info.pixels = GetBinaryPBMPixels(reinterpret_cast<const char*>(in_nocomments.get())+currentOffset,
				info.width, info.height);

		} else if (formatIdentifier == "P5") {	// PGM Binary
			info.colorChannels = 1;

			// Get max value
			const auto& [IGNORESB, IGNORESB, tmpMaxVal] = GetNextNumber<size_t>(in_nocomments.get(), currentOffset);
			info.maxValue = tmpMaxVal;

			info.pixels = GetBinaryPNMPixels(reinterpret_cast<const char*>(in_nocomments.get())+currentOffset,
				info.width, info.height, info.colorChannels);

		} else if (formatIdentifier == "P6") {	// PPM Binary
			info.colorChannels = 3;

			// Get max value
			const auto& [IGNORESB, IGNORESB, tmpMaxVal] = GetNextNumber<size_t>(in_nocomments.get(), currentOffset);
			info.maxValue = tmpMaxVal;

			info.pixels = GetBinaryPNMPixels(reinterpret_cast<const char*>(in_nocomments.get())+currentOffset,
				info.width, info.height, info.colorChannels);

		} else {
			throw except::unsupported_image_type("PNM Loader: Could not deduce image type from format magic number");
		}

		return info;
	}

}
