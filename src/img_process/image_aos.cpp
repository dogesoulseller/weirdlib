#ifdef WEIRDLIB_ENABLE_IMAGE_OPERATIONS
#include "../../include/weirdlib_image.hpp"
#include "../../include/cpu_detection.hpp"
#include "../../include/weirdlib_traits.hpp"
#include "../../include/weirdlib_fileops.hpp"
#include "../img_loaders/image_format_loaders.hpp"
#include <fstream>
#include <cstring>

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Weverything"
#include "../../external/stb/stb_image.h"
#pragma clang diagnostic pop


namespace wlib::image
{
	Image::Image(const std::string& path, const bool isRawData, const uint64_t _width, const uint64_t _height, const ColorFormat requestedFormat) {
		LoadImage(path, isRawData,_width, _height, requestedFormat);
	}

	Image::Image(const uint8_t* _pixels, const uint64_t _width, const uint64_t _height, const ColorFormat _format) {
		LoadImage(_pixels, _width, _height, _format);
	}

	Image::Image(const float* _pixels, const uint64_t _width, const uint64_t _height, const ColorFormat _format) {
		LoadImage(_pixels, _width, _height, _format);
	}

	void Image::LoadImage(const std::string& path, const bool isRawData, const uint64_t _width, const uint64_t _height, const ColorFormat requestedFormat) {
		std::ifstream f(path, std::ios::binary | std::ios::ate);
		size_t fileSize = f.tellg();
		f.seekg(0);

		if (isRawData) {
			width = _width;
			height = _height;
			format = requestedFormat;
			pixels.resize(fileSize);
			pixels.shrink_to_fit();

			alignas(64) auto pixtmp = new uint8_t[fileSize];
			f.read(reinterpret_cast<char*>(pixtmp), fileSize);

			ConvertUint8ToFloat(pixtmp, pixels.data(), fileSize);
			delete[] pixtmp;
			return;
		}

		auto ftype = wlib::file::DetectFileType(path);
		switch (ftype)
		{
		  case wlib::file::FILETYPE_PBM:
		  case wlib::file::FILETYPE_PGM:
		  case wlib::file::FILETYPE_PPM: { // TODO: Scale input
			auto pnmInfo = LoadPNM(path);
			pnmInfo.colorChannels = static_cast<ColorFormat>(pnmInfo.colorChannels);
			width = pnmInfo.width;
			height = pnmInfo.height;
			pixels.resize(GetTotalImageSize(width, height, format));
			pixels.shrink_to_fit();
			ConvertUint8ToFloat(pnmInfo.pixels.data(), pixels.data(), GetTotalImageSize(width, height, format));
			break;
		  }
		  case wlib::file::FILETYPE_PAM: { // TODO: Scale input
			auto pamInfo = LoadPAM(path);
			pamInfo.colorChannels = static_cast<ColorFormat>(pamInfo.colorChannels);
			width = pamInfo.width;
			height = pamInfo.height;
			pixels.resize(GetTotalImageSize(width, height, format));
			pixels.shrink_to_fit();
			ConvertUint16ToFloat(pamInfo.pixels.data(), pixels.data(), GetTotalImageSize(width, height, format));
			break;
		  }
		  default: {
			int w;
			int h;
			int chan;
			uint8_t* stbpix = nullptr;

			switch (requestedFormat)
			{
			  case ColorFormat::F_Grayscale:
				format = requestedFormat;
				stbpix = stbi_load(path.c_str(), &w, &h, &chan, 1);
				break;
			  case ColorFormat::F_GrayAlpha:
				format = requestedFormat;
				stbpix = stbi_load(path.c_str(), &w, &h, &chan, 2);
				break;
			  case ColorFormat::F_RGB:
			  case ColorFormat::F_BGR:
				format = requestedFormat;
				stbpix = stbi_load(path.c_str(), &w, &h, &chan, 3);
				break;
			  case ColorFormat::F_RGBA:
			  case ColorFormat::F_BGRA:
				format = requestedFormat;
				stbpix = stbi_load(path.c_str(), &w, &h, &chan, 4);
				break;
			  case ColorFormat::F_Default:
				stbpix = stbi_load(path.c_str(), &w, &h, &chan, 0);
				switch (chan)
				{
				  case 4:
					format = F_RGBA;
					break;
				  case 3:
					format = F_RGB;
					break;
				  case 2:
					format = F_GrayAlpha;
					break;
				  case 1:
					format = F_Grayscale;
				  default:
					break;
				}
				break;
			}

			width = w;
			height = h;
			pixels.resize(GetTotalImageSize(width, height, format));
			pixels.shrink_to_fit();

			ConvertUint8ToFloat(stbpix, pixels.data(), GetTotalImageSize(width, height, format));

			switch (requestedFormat)
			{
			  case F_BGR:
				for (size_t i = 0; i < GetTotalImageSize(width, height, format); i+=3) {
					std::swap(pixels[0+i], pixels[2+i]);
				}
				break;
			  case F_BGRA:
				for (size_t i = 0; i < GetTotalImageSize(width, height, format); i+=4) {
					std::swap(pixels[0+i], pixels[2+i]);
				}
				break;
			  default:
				break;
			}

			free(stbpix);
			return;
		  }
		}
	}

	void Image::LoadImage(const uint8_t* _pixels, const uint64_t _width, const uint64_t _height, const ColorFormat _format) {
		width = _width;
		height = _height;
		format = _format;

		pixels.resize(0);
		pixels.shrink_to_fit();

		pixels.assign(_pixels, _pixels + GetTotalImageSize(width, height, format));
	}

	void Image::LoadImage(const float* _pixels, const uint64_t _width, const uint64_t _height, const ColorFormat _format) {
		width = _width;
		height = _height;
		format = _format;

		pixels.resize(0);
		pixels.shrink_to_fit();

		pixels.assign(_pixels, _pixels + GetTotalImageSize(width, height, format));
	}

	size_t Image::GetTotalImageSize(const uint64_t width, const uint64_t height, const ColorFormat format) noexcept {
		size_t result = width * height;

		switch (format)
		{
		  case F_GrayAlpha:
			result *= 2;
			break;
		  case F_RGB:
		  case F_BGR:
			result *= 3;
			break;
		  case F_RGBA:
		  case F_BGRA:
			result *= 4;
			break;
		  case F_Grayscale:
		  default:
			break;
		};

		return result;
	}

	size_t Image::GetTotalImageSize() const noexcept {
		size_t result = width * height;

		switch (format)
		{
		  case F_GrayAlpha:
			result *= 2;
			break;
		  case F_RGB:
		  case F_BGR:
			result *= 3;
			break;
		  case F_RGBA:
		  case F_BGRA:
			result *= 4;
			break;
		  case F_Grayscale:
		  default:
			break;
		};

		return result;
	}

	void Image::ConvertToRGBA() {
		Image::ConvertToRGBA(*this);
	}

	void Image::ConvertToBGR() {
		Image::ConvertToBGR(*this);
	}

	void Image::ConvertToRGB() {
		Image::ConvertToRGB(*this);
	}

	void Image::ConvertToBGRA() {
		Image::ConvertToBGRA(*this);
	}

	void Image::ConvertToRGB(Image& in) {
		switch (in.GetFormat())
		{
		  case F_BGR: {
			detail::swapRAndB_3c(in.GetPixels_Unsafe(), in.GetWidth() * in.GetHeight());
			break;
		  }
		  case F_RGBA: {
			auto tmp = detail::dropAlpha_4c(in);
			std::swap(in.AccessStorage(), tmp);
			break;
		  }
		  case F_BGRA: {
			detail::swapRAndB_4c(in.GetPixels_Unsafe(), in.GetWidth() * in.GetHeight());
			auto tmp = detail::dropAlpha_4c(in);
			std::swap(in.AccessStorage(), tmp);
			break;
		  }
		  case F_Grayscale: {
			auto tmp = detail::broadcastGray_to3c(in);
			std::swap(in.AccessStorage(), tmp);
			break;
		  }
		  case F_GrayAlpha: {
			auto tmp = detail::broadcastGrayAlpha_to3c(in);
			std::swap(in.AccessStorage(), tmp);
			break;
		  }
		  default:
			break;
		};

		in.SetFormat(F_RGB);
	}

	void Image::ConvertToBGR(Image& in) {
		switch (in.GetFormat())
		{
		  case F_RGB: {
			detail::swapRAndB_3c(in.GetPixels_Unsafe(), in.GetWidth() * in.GetHeight());
			break;
		  }
		  case F_RGBA: {
			detail::swapRAndB_4c(in.GetPixels_Unsafe(), in.GetWidth() * in.GetHeight());
			auto tmp = detail::dropAlpha_4c(in);
			std::swap(in.AccessStorage(), tmp);
			break;
		  }
		  case F_BGRA: {
			auto tmp = detail::dropAlpha_4c(in);
			std::swap(in.AccessStorage(), tmp);
			break;
		  }
		  case F_Grayscale: {
			auto tmp = detail::broadcastGray_to3c(in);
			std::swap(in.AccessStorage(), tmp);
			break;
		  }
		  case F_GrayAlpha: {
			auto tmp = detail::broadcastGrayAlpha_to3c(in);
			std::swap(in.AccessStorage(), tmp);
			break;
		  }
		  default:
			break;

		};

		in.SetFormat(F_BGR);
	}

	void Image::ConvertToRGBA(Image& in) {
		switch (in.GetFormat())
		{
		  case F_RGB: {
			auto tmp = detail::appendAlpha_3c(in);
			std::swap(in.AccessStorage(), tmp);
			break;
		  }
		  case F_BGR: {
			auto tmp = detail::appendAlpha_3c(in);
			detail::swapRAndB_4c(tmp.data(), in.GetWidth() * in.GetHeight());
			std::swap(in.AccessStorage(), tmp);
			break;
		  }
		  case F_BGRA: {
			detail::swapRAndB_4c(in.GetPixels_Unsafe(), in.GetWidth() * in.GetHeight());
			break;
		  }
		  case F_Grayscale: {
			auto tmp = detail::broadcastGray_to4c(in);
			std::swap(in.AccessStorage(), tmp);
			break;
		  }
		  case F_GrayAlpha: {
			auto tmp = detail::broadcastGrayAlpha_to4c(in);
			std::swap(in.AccessStorage(), tmp);
			break;
		  }
		  default:
			break;
		}

		in.SetFormat(F_RGBA);
	}

	void Image::ConvertToBGRA(Image& in) {
		switch (in.GetFormat())
		{
		  case F_RGB: {
			auto tmp = detail::appendAlpha_3c(in);
			detail::swapRAndB_4c(tmp.data(), in.GetWidth() * in.GetHeight());
			std::swap(in.AccessStorage(), tmp);
			break;
		  }
		  case F_RGBA: {
			detail::swapRAndB_4c(in.GetPixels_Unsafe(), in.GetWidth() * in.GetHeight());
			break;
		  }
		  case F_BGR: {
			auto tmp = detail::appendAlpha_3c(in);
			std::swap(in.AccessStorage(), tmp);
			break;
		  }
		  case F_Grayscale: {
			auto tmp = detail::broadcastGray_to4c(in);
			std::swap(in.AccessStorage(), tmp);
			break;
		  }
		  case F_GrayAlpha: {
			auto tmp = detail::broadcastGrayAlpha_to4c(in);
			std::swap(in.AccessStorage(), tmp);
			break;
		  }
		  default:
			break;
		}

		in.SetFormat(F_BGRA);
	}

	std::vector<uint8_t> Image::GetPixelsAsInt() {
		alignas(64) std::vector<uint8_t> pixelsOut;
		size_t dataLength = Image::GetTotalImageSize(width, height, format);
		pixelsOut.resize(dataLength);

		#if X86_SIMD_LEVEL >= LV_SSSE3
		// Works on 16 source floats at a time
		// Doing this allows full utilization of an XMM register for stores
		size_t iters = dataLength / 16;
		size_t itersRem = dataLength % 16;

		// Mask for moving least significant byte of 32-bit integers to start of register
		const __m128i shufmask = _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12, 8, 4, 0);

		alignas(16) std::array<uint8_t, 16> out_tmp;

		for (size_t i = 0; i < iters; i++) {
			// Load 16 floats into registers
			__m128 in0 = _mm_loadu_ps(pixels.data() + i*16);
			__m128 in1 = _mm_loadu_ps(pixels.data() + i*16 + 4);
			__m128 in2 = _mm_loadu_ps(pixels.data() + i*16 + 8);
			__m128 in3 = _mm_loadu_ps(pixels.data() + i*16 + 12);

			// Batch convert 4 floats into 32-bit integers
			__m128i converted0_32 = _mm_cvtps_epi32(in0);
			__m128i converted1_32 = _mm_cvtps_epi32(in1);
			__m128i converted2_32 = _mm_cvtps_epi32(in2);
			__m128i converted3_32 = _mm_cvtps_epi32(in3);

			// Perform shuffle that moves the least significant byte of 32-bit integers to the start of each register.
			// Because the color values are capped at 255 (maximum value representable by a single byte), it
			// effectively packs the set of 16 total values into 32-bit collections at the bottom of each register
			__m128i converted0 = _mm_shuffle_epi8(converted0_32, shufmask);
			__m128i converted1 = _mm_shuffle_epi8(converted1_32, shufmask);
			__m128i converted2 = _mm_shuffle_epi8(converted2_32, shufmask);
			__m128i converted3 = _mm_shuffle_epi8(converted3_32, shufmask);

			// Merge packed values:
			// Unpack first two values, creating XX10
			__m128i low0 = _mm_unpacklo_epi32(converted0, converted1);
			// Unpack remaining two values, creating 32XX
			__m128i low1 = _mm_unpacklo_epi32(converted2, converted3);

			// Merge two registers with data in their lower halves, forming 3210 in a single register
			__m128 output = _mm_shuffle_ps(reinterpret_cast<__m128>(low0), reinterpret_cast<__m128>(low1), 0b01000100);

			_mm_store_ps(reinterpret_cast<float*>(out_tmp.data()), output);

			std::memcpy(pixelsOut.data()+i*16, out_tmp.data(), 16);
		}

		for (size_t i = iters * 16; i < iters * 16 + itersRem; i++) {
			pixelsOut[i] = static_cast<uint8_t>(pixels[i]);
		}

		#else
		for (size_t i = 0; i < dataLength; i++) {
			pixelsOut[i] = static_cast<uint8_t>(pixels[i]);
		}

		#endif

		return pixelsOut;
	}
} // namespace wlib::image

#endif
