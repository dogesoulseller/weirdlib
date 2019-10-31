#include "image_format_loaders.hpp"
#include "img_loader_exceptions.hpp"
#include "../common.hpp"
#include "../../include/weirdlib_bitops.hpp"
#include "../../include/weirdlib_string.hpp"

#include <cstdint>
#include <regex>
#include <fstream>
#include <memory>
#include <charconv>

namespace wlib::image
{
	ImageInfoPAM LoadPAM(const std::string& path) {
		std::ifstream in(path, std::ios::binary);
		if (!in.good()) {
			throw except::file_open_error(std::string("PAM Loader: Failed to open file ") + path);
		}

		return LoadPAM(in);
	}

	ImageInfoPAM LoadPAM(std::ifstream& in) {
		in.seekg(0, std::ios::end);
		size_t fileSize = in.tellg();
		if (fileSize == 0) {
			throw except::file_open_error("PAM Loader: File size returned was 0");
		}

		in.seekg(0);

		auto fileBytes = std::make_unique<uint8_t[]>(fileSize);
		in.read(reinterpret_cast<char*>(fileBytes.get()), fileSize);

		return LoadPAM(fileBytes.get(), fileSize);
	}

	ImageInfoPAM LoadPAM(const uint8_t* in, size_t size) {
		ImageInfoPAM info;

		std::regex commentRemoveRegex("(\\s*#+.*)",
			std::regex_constants::optimize | std::regex_constants::ECMAScript);

		// Make copy of data without comments
		auto in_nocomments = std::make_unique<uint8_t[]>(size+1);

		std::regex_replace(reinterpret_cast<char*>(in_nocomments.get()),
			reinterpret_cast<const char*>(in),
			reinterpret_cast<const char*>(in+size),
			commentRemoveRegex, "");

		auto headerEnd = wlib::str::strstr(reinterpret_cast<const char*>(in_nocomments.get()), "ENDHDR", 0, 6)+7;

		std::array<char, 256> headerBuffer;
		std::copy(reinterpret_cast<const char*>(in_nocomments.get()), const_cast<const char*>(headerEnd), headerBuffer.begin());

		// Get position of each parameter's data
		auto widthStr = str::strstr(headerBuffer.data(), "WIDTH", 0, 5) + 6;
		auto heightStr = str::strstr(headerBuffer.data(), "HEIGHT", 0, 6) + 7;
		auto depthStr = str::strstr(headerBuffer.data(), "DEPTH", 0, 5) + 6;
		auto maxvalStr = str::strstr(headerBuffer.data(), "MAXVAL", 0, 6) + 7;

		// Find first whitespace following digit
		auto widthEnd = GetNextWhitespace(widthStr);
		auto heightEnd = GetNextWhitespace(heightStr);
		auto depthEnd = GetNextWhitespace(depthStr);
		auto maxvalEnd = GetNextWhitespace(maxvalStr);

		// Convert ASCII values to ints
		std::from_chars(widthStr, widthEnd, info.width);
		std::from_chars(heightStr, heightEnd, info.height);
		std::from_chars(depthStr, depthEnd, info.colorChannels);
		std::from_chars(maxvalStr, maxvalEnd, info.maxValue);

		if (info.width == 0) {
			throw except::invalid_image_data("PAM Loader: Image width must not be 0");
		}
		if (info.height == 0) {
			throw except::invalid_image_data("PAM Loader: Image height must not be 0");
		}
		if (info.colorChannels == 0) {
			throw except::invalid_image_data("PAM Loader: Image depth must not be 0");
		}

		info.pixels.resize(info.width * info.height*info.colorChannels);

		if (info.maxValue == 1) {	// PBM-like (1 bit per pixel)
			size_t iters = info.width / 8;
			size_t itersRem = info.width % 8;

			size_t totalOffset = 0;

			for (size_t h = 0; h < info.height; h++) {	// For each row
				for (size_t b = 0; b < iters; b++) {	// For each full byte
					const uint8_t val = *(headerEnd + (h*(iters+1)) + b);

					info.pixels[totalOffset+0] = wlib::bop::test(val, 7) ? 0 : 255;
					info.pixels[totalOffset+1] = wlib::bop::test(val, 6) ? 0 : 255;
					info.pixels[totalOffset+2] = wlib::bop::test(val, 5) ? 0 : 255;
					info.pixels[totalOffset+3] = wlib::bop::test(val, 4) ? 0 : 255;
					info.pixels[totalOffset+4] = wlib::bop::test(val, 3) ? 0 : 255;
					info.pixels[totalOffset+5] = wlib::bop::test(val, 2) ? 0 : 255;
					info.pixels[totalOffset+6] = wlib::bop::test(val, 1) ? 0 : 255;
					info.pixels[totalOffset+7] = wlib::bop::test(val, 0) ? 0 : 255;

					totalOffset += 8;
				}

				const uint8_t val = *(headerEnd + (h*(iters+1)) + iters);
				for (size_t i = 0; i < itersRem; i++) {	// For each bit in last byte
					info.pixels[totalOffset] = wlib::bop::test(val, 7-i) ? 0 : 255;
					totalOffset++;
				}
			}
		} else {	// PAM (PPM-like)
			if (info.maxValue <= 255) { // 8-bit
				for (size_t i = 0; i < info.width * info.height * info.colorChannels; i++) {
					info.pixels[i] = *(headerEnd+i);
				}
			} else { // 16-bit
				auto cvtptr = reinterpret_cast<const uint16_t*>(headerEnd);
				for (size_t i = 0; i < info.width * info.height * info.colorChannels; i++) {
					info.pixels[i] = *(cvtptr+i);
				}
			}
		}

		return info;
	}
} // namespace wlib::image
