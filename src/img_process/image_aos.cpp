#ifdef WEIRDLIB_ENABLE_IMAGE_OPERATIONS
#include "../../include/weirdlib_image.hpp"
#include "../../include/cpu_detection.hpp"
#include "../../include/weirdlib_traits.hpp"
#include "../../include/weirdlib_fileops.hpp"
#include "../img_loaders/image_format_loaders.hpp"
#include <fstream>

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Weverything"
#include "../../external/stb/stb_image.h"
#pragma clang diagnostic pop

#ifdef __x86_64__
	#include <immintrin.h>
#endif

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
		case wlib::file::FILETYPE_BMP: {
			auto bmpInfo = LoadBMP(path);
			height = bmpInfo.height;
			width = bmpInfo.width;
			format = static_cast<ColorFormat>(bmpInfo.colorChannels);
			pixels.resize(GetTotalImageSize(width, height, format));
			pixels.shrink_to_fit();
			ConvertUint8ToFloat(bmpInfo.pixels.data(), pixels.data(), GetTotalImageSize(width, height, format));
			break;
		}
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

#endif
