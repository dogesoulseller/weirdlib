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

		for (auto& ptr : channels) {
			if (ptr != nullptr) {
				delete[] ptr;
			}
		}

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

		for (auto& ptr : channels) {
			if (ptr != nullptr) {
				delete[] ptr;
			}
		}

		channels.resize(img.channels.size());
		channels.shrink_to_fit();
		channels.assign(img.channels.begin(), img.channels.end());

		for (auto& ptr: img.channels) {
			ptr = nullptr;
		}

		return *this;
	}

	ImageSoA::~ImageSoA() {
		for (auto& c : channels) {
			if (c != nullptr) {
				delete[] c;
			}
		}
	}

	void ImageSoA::ConvertToRGBA() {
		ImageSoA::ConvertToRGBA(*this);
	}

	void ImageSoA::ConvertToBGR() {
		ImageSoA::ConvertToBGR(*this);
	}

	void ImageSoA::ConvertToRGB() {
		ImageSoA::ConvertToRGB(*this);
	}

	void ImageSoA::ConvertToBGRA() {
		ImageSoA::ConvertToBGRA(*this);
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
			alignas(64) auto tmp0 = new float[in.width*in.height];
			alignas(64) auto tmp1 = new float[in.width*in.height];
			std::copy(in.channels[0], in.channels[0]+in.width*in.height, tmp0);
			std::copy(in.channels[0], in.channels[0]+in.width*in.height, tmp1);
			in.channels[1] = tmp0;
			in.channels[2] = tmp1;
			break;
		}
		case F_GrayAlpha: {
			delete[] in.channels[1];
			in.channels.resize(3);
			alignas(64) auto tmp0 = new float[in.width*in.height];
			alignas(64) auto tmp1 = new float[in.width*in.height];
			std::copy(in.channels[0], in.channels[0]+in.width*in.height, tmp0);
			std::copy(in.channels[0], in.channels[0]+in.width*in.height, tmp1);
			in.channels[1] = tmp0;
			in.channels[2] = tmp1;
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
			alignas(64) auto tmp0 = new float[in.width*in.height];
			alignas(64) auto tmp1 = new float[in.width*in.height];
			std::copy(in.channels[0], in.channels[0]+in.width*in.height, tmp0);
			std::copy(in.channels[0], in.channels[0]+in.width*in.height, tmp1);
			in.channels[1] = tmp0;
			in.channels[2] = tmp1;
			break;
		}
		case F_GrayAlpha: {
			delete[] in.channels[1];
			in.channels.resize(3);
			alignas(64) auto tmp0 = new float[in.width*in.height];
			alignas(64) auto tmp1 = new float[in.width*in.height];
			std::copy(in.channels[0], in.channels[0]+in.width*in.height, tmp0);
			std::copy(in.channels[0], in.channels[0]+in.width*in.height, tmp1);
			in.channels[1] = tmp0;
			in.channels[2] = tmp1;
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
		case F_RGB:
		{
			alignas(64) auto tmp = new float[in.width * in.height];
			std::uninitialized_fill(tmp, tmp + in.width * in.height, 255.0f);
			in.channels.push_back(tmp);
			break;
		}
		case F_Grayscale: {
			in.channels.resize(4);
			alignas(64) auto tmp0 = new float[in.width*in.height];
			alignas(64) auto tmp1 = new float[in.width*in.height];
			alignas(64) auto tmp2 = new float[in.width*in.height];
			std::copy(in.channels[0], in.channels[0]+in.width*in.height, tmp0);
			std::copy(in.channels[0], in.channels[0]+in.width*in.height, tmp1);
			std::uninitialized_fill(tmp2, tmp2 + in.width * in.height, 255.0f);
			in.channels[1] = tmp0;
			in.channels[2] = tmp1;
			in.channels[3] = tmp1;
			break;
		}
		case F_GrayAlpha: {
			in.channels.resize(4);
			alignas(64) auto tmp0 = new float[in.width*in.height];
			alignas(64) auto tmp1 = new float[in.width*in.height];
			std::copy(in.channels[0], in.channels[0]+in.width*in.height, tmp0);
			std::copy(in.channels[0], in.channels[0]+in.width*in.height, tmp1);
			in.channels[3] = in.channels[1];
			in.channels[1] = tmp0;
			in.channels[2] = tmp1;
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
			alignas(64) auto tmp = new float[in.width * in.height];
			std::uninitialized_fill(tmp, tmp + in.width * in.height, 255.0f);
			in.channels.push_back(tmp);
			break;
		}
		case F_Grayscale: {
			in.channels.resize(4);
			alignas(64) auto tmp0 = new float[in.width*in.height];
			alignas(64) auto tmp1 = new float[in.width*in.height];
			alignas(64) auto tmp2 = new float[in.width*in.height];
			std::copy(in.channels[0], in.channels[0]+in.width*in.height, tmp0);
			std::copy(in.channels[0], in.channels[0]+in.width*in.height, tmp1);
			std::uninitialized_fill(tmp2, tmp2 + in.width * in.height, 255.0f);
			in.channels[1] = tmp0;
			in.channels[2] = tmp1;
			in.channels[3] = tmp2;
			break;
		}
		case F_GrayAlpha: {
			in.channels.resize(4);
			alignas(64) auto tmp0 = new float[in.width*in.height];
			alignas(64) auto tmp1 = new float[in.width*in.height];
			std::copy(in.channels[0], in.channels[0]+in.width*in.height, tmp0);
			std::copy(in.channels[0], in.channels[0]+in.width*in.height, tmp1);
			in.channels[3] = in.channels[1];
			in.channels[1] = tmp0;
			in.channels[2] = tmp1;
			break;
		}
		default:
			break;
		}

		in.format = F_BGRA;
	}

} // namespace wlib::image

#endif
