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

static constexpr std::array<char, 10> digits = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'};
static constexpr std::array<char, 11> digitsOrNeg = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-'};
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
	static const std::regex multiWhitespaceRemoveRegex("\\s\\s+", std::regex_constants::ECMAScript | std::regex_constants::optimize);
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

// Get position of next digit
template<typename PtrT, typename IntT, typename = std::enable_if_t<std::is_pointer_v<PtrT> && std::is_integral_v<IntT>>>
inline static PtrT GetNextDigit(const PtrT ptr, IntT& offset) {
	PtrT out = std::find_first_of(ptr+offset, ptr+offset+128, digits.cbegin(), digits.cend());
	offset = out - ptr + 1;
	return out;
}

// Get position of next whitespace character
template<typename PtrT, typename IntT, typename = std::enable_if_t<std::is_pointer_v<PtrT> && std::is_integral_v<IntT>>>
inline static PtrT GetNextWhitespace(const PtrT ptr, IntT& offset) {
	PtrT out = std::find_first_of(ptr+offset, ptr+offset+128, whitespace.cbegin(), whitespace.cend());
	offset = out - ptr + 1;
	return out;
}

// Get position of next whitespace character without offset
template<typename PtrT, typename = std::enable_if_t<std::is_pointer_v<PtrT>>>
inline static PtrT GetNextWhitespace(const PtrT ptr) {
	PtrT out = std::find_first_of(ptr, ptr+128, whitespace.cbegin(), whitespace.cend());
	return out;
}

// Tuple of: <digitPos, whitespacePos, parsedNumber>
template<typename PtrT, typename IntT, typename = std::enable_if_t<std::is_pointer_v<PtrT> && std::is_integral_v<IntT>>>
using NumberParseResult = std::tuple<PtrT, PtrT, IntT>;

template<typename OutputT, typename PtrT, typename OffT>
inline static NumberParseResult<PtrT, OutputT> GetNextNumber(const PtrT startPtr, OffT& offset) {
	OutputT out;
	PtrT digitLocation = GetNextDigit(startPtr, offset);
	PtrT whitespaceLocation = GetNextWhitespace(startPtr, offset);

	std::from_chars(reinterpret_cast<const char*>(digitLocation), reinterpret_cast<const char*>(whitespaceLocation), out);

	return std::move(std::make_tuple(digitLocation, whitespaceLocation, out));
}

namespace wlib::image
{
	ImageInfoPNM LoadPNM(const std::string& path) {
		std::ifstream in(path, std::ios::binary);
		if (!in.good()) {
			throw except::file_open_error(std::string("PNM Loader: Failed to open file ") + path);
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

		std::regex commentRemoveRegex("(\\s*#+.*)",
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

		auto headerEnd = std::strstr(reinterpret_cast<const char*>(in_nocomments.get()), "ENDHDR")+7;

		std::array<char, 256> headerBuffer;
		std::copy(reinterpret_cast<const char*>(in_nocomments.get()), const_cast<const char*>(headerEnd), headerBuffer.begin());

		// Get position of each parameter's data
		auto widthStr = std::strstr(headerBuffer.data(), "WIDTH") + 6;
		auto heightStr = std::strstr(headerBuffer.data(), "HEIGHT") + 7;
		auto depthStr = std::strstr(headerBuffer.data(), "DEPTH") + 6;
		auto maxvalStr = std::strstr(headerBuffer.data(), "MAXVAL") + 7;

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


	ImageInfoPFM LoadPFM(const std::string& path) {
		std::ifstream in(path, std::ios::binary);
		return LoadPFM(in);
	}

	ImageInfoPFM LoadPFM(std::ifstream& in) {
		in.seekg(0, std::ios::end);
		size_t fileSize = in.tellg();
		in.seekg(0);

		auto fileBytes = std::make_unique<uint8_t[]>(fileSize);
		in.read(reinterpret_cast<char*>(fileBytes.get()), fileSize);

		return LoadPFM(fileBytes.get(), fileSize);
	}

	ImageInfoPFM LoadPFM(const uint8_t* in, size_t /*size*/) {
		ImageInfoPFM info;
		size_t currentOffset = 3;

		// Get width
		const auto& [IGNORESB, IGNORESB, tmpWidth] = GetNextNumber<size_t>(in, currentOffset);
		info.width = tmpWidth;

		// Get height
		const auto& [IGNORESB, IGNORESB, tmpHeight] = GetNextNumber<size_t>(in, currentOffset);
		info.height = tmpHeight;

		// Make space for data
		info.data.resize(info.width * info.height);

		// Get endianness
		std::array<char, 32> floatBuffer;

		auto digitLocation = std::find_first_of(in+currentOffset, in+currentOffset+128,
			digitsOrNeg.cbegin(), digitsOrNeg.cend());

		currentOffset = digitLocation - in + 1;

		auto whitespaceLocation = GetNextWhitespace(in, currentOffset);
		currentOffset = whitespaceLocation - in + 1;

		std::copy(digitLocation, whitespaceLocation, floatBuffer.begin());

		float endiannessValue = static_cast<float>(std::atof(floatBuffer.data()));

		info.isLittleEndian = math::float_eq(endiannessValue, -1.0f);

		// Get data
		auto inFloat = reinterpret_cast<const float*>(in+currentOffset);
		for (size_t i = 0; i < info.width * info.height; i++) {
			info.data[i] = *(inFloat+i);
		}

		return info;
	}

}
