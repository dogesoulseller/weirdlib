#ifdef WEIRDLIB_ENABLE_IMAGE_OPERATIONS
#include "../../include/weirdlib_image.hpp"
#include "../../include/weirdlib_traits.hpp"
#include "../../include/cpu_detection.hpp"

#include "../common.hpp"
#include <cmath>
#include <algorithm>
#include <thread>
#include <array>
#include <cstring>


#if WEIRDLIB_MULTITHREADING_MODE == WEIRDLIB_MTMODE_TBB
#include <tbb/tbb.h>
#endif

namespace wlib::image
{
	ImageSoA::ImageSoA(const ImageSoA& img) {
		*this = img;
	}

	ImageSoA::ImageSoA(ImageSoA&& img) {
		*this = std::move(img);
	}

	ImageSoA& ImageSoA::operator=(const ImageSoA& img) {
		if (&img == this) {
			return *this;
		}

		height = img.height;
		width = img.width;
		format = img.format;

		std::for_each(channels.begin(), channels.end(), [](auto ptr){delete[] ptr;});

		channels.resize(img.channels.size());
		channels.shrink_to_fit();

		for (size_t i = 0; i < channels.size(); i++) {
			auto chan = new float[width * height];
			std::copy(img.channels[i], img.channels[i] + width * height, chan);
			channels[i] = chan;
		}

		return *this;
	}

	ImageSoA& ImageSoA::operator=(ImageSoA&& img) {
		if (&img == this) {
			return *this;
		}

		height = img.height;
		width = img.width;
		format = img.format;

		std::for_each(channels.begin(), channels.end(), [](auto ptr){delete[] ptr;});

		channels.resize(img.channels.size());
		channels.shrink_to_fit();
		channels.assign(img.channels.begin(), img.channels.end());

		img.channels.assign(img.channels.size(), nullptr);

		return *this;
	}

	ImageSoA::~ImageSoA() {
		std::for_each(channels.begin(), channels.end(), [](auto ptr){delete[] ptr;});
	}

	void ImageSoA::ConvertToRGB(ImageSoA& in) {
		switch (in.format)
		{
		  case F_BGRA:
			std::swap(in.channels[0], in.channels[2]);
			[[fallthrough]];
		  case F_RGBA:
			delete[] in.channels[3];
			in.channels.erase(in.channels.begin() + 3);
			break;
		  case F_BGR:
			std::swap(in.channels[0], in.channels[2]);
			break;
		  case F_Grayscale: {
			in.channels.resize(3);
			detail::extendGSTo3Chan(in);
			break;
		  }
		  case F_GrayAlpha: {
			delete[] in.channels[1];
			in.channels.resize(3);
			detail::extendGSTo3Chan(in);
			break;
		  }
		  default:
			break;
		}

		in.format = F_RGB;
	}

	void ImageSoA::ConvertToBGR(ImageSoA& in) {
		switch (in.format)
		{
		  case F_RGBA:
			std::swap(in.channels[0], in.channels[2]);
			[[fallthrough]];
		  case F_BGRA:
			delete[] in.channels[3];
			in.channels.erase(in.channels.begin() + 3);
			break;
		  case F_RGB:
			std::swap(in.channels[0], in.channels[2]);
			break;
		  case F_Grayscale: {
			in.channels.resize(3);
			detail::extendGSTo3Chan(in);
			break;
		  }
		  case F_GrayAlpha: {
			delete[] in.channels[1];
			in.channels.resize(3);
			detail::extendGSTo3Chan(in);
			break;
		  }
		  default:
			break;
		}

		in.format = F_BGR;
	}

	void ImageSoA::ConvertToRGBA(ImageSoA& in) {
		switch (in.format)
		{
		  case F_BGRA:
			std::swap(in.channels[0], in.channels[2]);
			break;
		  case F_BGR:
			std::swap(in.channels[0], in.channels[2]);
			[[fallthrough]];
		  case F_RGB: {
			detail::appendConstantAlpha(in);
			break;
		  }
		  case F_Grayscale: {
			in.channels.resize(4);
			detail::extendGSTo4Chan(in, true);
			break;
		  }
		  case F_GrayAlpha: {
			in.channels.resize(4);
			detail::extendGSTo4Chan(in, false);
			break;
		  }
		  default:
			break;
		}

		in.format = F_RGBA;
	}

	void ImageSoA::ConvertToBGRA(ImageSoA& in) {
		switch (in.format)
		{
		  case F_RGBA:
			std::swap(in.channels[0], in.channels[2]);
			break;
		  case F_RGB:
			std::swap(in.channels[0], in.channels[2]);
			[[fallthrough]];
		  case F_BGR: {
			detail::appendConstantAlpha(in);
			break;
		  }
		  case F_Grayscale: {
			in.channels.resize(4);
			detail::extendGSTo4Chan(in, true);
			break;
		  }
		  case F_GrayAlpha: {
			in.channels.resize(4);
			detail::extendGSTo4Chan(in, false);
			break;
		  }
		  default:
			break;
		}

		in.format = F_BGRA;
	}

} // namespace wlib::image

#endif
