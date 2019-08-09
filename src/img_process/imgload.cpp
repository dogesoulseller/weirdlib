#include "../../include/weirdlib_image.hpp"
#include "../../include/cpu_detection.hpp"
#include "../../include/weirdlib_traits.hpp"
#include <fstream>

#ifdef WEIRDLIB_ENABLE_IMAGE_OPERATIONS

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Weverything"
#include "../../external/stb/stb_image.h"
#pragma clang diagnostic pop

#ifdef __x86_64__
	#include <immintrin.h>
#endif

namespace wlib::image
{
	Image::Image(const std::string& path, bool isRawData, const uint64_t _width, const uint64_t _height, ColorFormat requestedFormat) {
		LoadImage(path, isRawData,_width, _height, requestedFormat);
	}

	Image::Image(const uint8_t* _pixels, const uint64_t _width, const uint64_t _height, const ColorFormat _format) {
		LoadImage(_pixels, _width, _height, _format);
	}

	Image::Image(const float* _pixels, const uint64_t _width, const uint64_t _height, const ColorFormat _format) {
		LoadImage(_pixels, _width, _height, _format);
	}


	void Image::LoadImage(const std::string& path, bool isRawData, const uint64_t _width, const uint64_t _height, ColorFormat requestedFormat) {
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

			return;
		}

		int w;
		int h;
		int chan;
		uint8_t* stbpix;

		switch (requestedFormat)
		{
		case ColorFormat::F_Grayscale:
			stbpix = stbi_load(path.c_str(), &w, &h, &chan, 1);
			break;
		case ColorFormat::F_GrayAlpha:
			stbpix = stbi_load(path.c_str(), &w, &h, &chan, 2);
			break;
		case ColorFormat::F_RGB:
		case ColorFormat::F_BGR:
			stbpix = stbi_load(path.c_str(), &w, &h, &chan, 3);
			break;
		case ColorFormat::F_Default:
		case ColorFormat::F_RGBA:
		case ColorFormat::F_BGRA:
			stbpix = stbi_load(path.c_str(), &w, &h, &chan, 4);
			break;
		}

		width = w;
		height = h;
		format = requestedFormat;
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
		switch (format)
		{
		case ColorFormat::F_Grayscale:
			return width * height;
		case ColorFormat::F_GrayAlpha:
			return width * height * 2;
		case ColorFormat::F_RGB:
		case ColorFormat::F_BGR:
			return width * height * 3;
		case ColorFormat::F_Default:
		case ColorFormat::F_RGBA:
		case ColorFormat::F_BGRA:
			return width * height * 4;
		default:
			return 0;
		}
	}
} // namespace wlib::image
#else
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Weverything"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wterminate"
namespace wlib::image
{
	constexpr const char* errMsg = "This function is a stub - image ops module was disabled for this compilation";

	Image::Image(const std::string& /*path*/, bool /*isRawData*/, const uint64_t /*_width*/, const uint64_t /*_height*/, ColorFormat /*requestedFormat*/) {
		throw wlib::module_not_built(errMsg);
	}
	Image::Image(const uint8_t* /*_pixels*/, const uint64_t /*_width*/, const uint64_t /*_height*/, const ColorFormat /*_format*/) {
		throw wlib::module_not_built(errMsg);
	}
	Image::Image(const float* /*_pixels*/, const uint64_t /*_width*/, const uint64_t /*_height*/, const ColorFormat /*_format*/) {
		throw wlib::module_not_built(errMsg);
	}
	void Image::LoadImage(const std::string& /*path*/, bool /*isRawData*/, const uint64_t /*_width*/, const uint64_t /*_height*/, ColorFormat /*requestedFormat*/) {
		throw wlib::module_not_built(errMsg);
	}
	void Image::LoadImage(const uint8_t* /*_pixels*/, const uint64_t /*_width*/, const uint64_t /*_height*/, const ColorFormat /*_format*/) {
		throw wlib::module_not_built(errMsg);
	}
	void Image::LoadImage(const float* /*_pixels*/, const uint64_t /*_width*/, const uint64_t /*_height*/, const ColorFormat /*_format*/) {
		throw wlib::module_not_built(errMsg);
	}
	size_t Image::GetTotalImageSize(const uint64_t /*width*/, const uint64_t /*height*/, const ColorFormat /*format*/) noexcept {
		throw wlib::module_not_built(errMsg);
	}
} // namespace wlib::image
#pragma clang diagnostic pop
#pragma GCC diagnostic pop
#endif
