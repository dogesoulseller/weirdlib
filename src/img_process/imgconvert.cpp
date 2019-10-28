#ifdef WEIRDLIB_ENABLE_IMAGE_OPERATIONS
#include "../../include/weirdlib_image.hpp"
#include "../../include/weirdlib_traits.hpp"
#include "../../include/cpu_detection.hpp"

#include <vector>

namespace wlib::image
{
	namespace detail
	{
		void swapRAndB_3c(float* in, size_t count) {
			for (size_t i = 0; i < count; i++) {
				auto tmp = in[i*3];

				in[i*3] = in[i*3+2];
				in[i*3+2] = tmp;
			}
		}

		void swapRAndB_4c(float* in, size_t count) {
			for (size_t i = 0; i < count; i++) {
				auto tmp = in[i*4];

				in[i*4] = in[i*4+2];
				in[i*4+2] = tmp;
			}
		}

		std::vector<float> dropAlpha_4c(Image& in) {
			std::vector<float> tmp(in.GetWidth() * in.GetHeight() * 3);
			auto inStorage = in.AccessStorage();

			for (size_t i = 0; i < in.GetWidth() * in.GetHeight(); i++) {
				tmp[i*3] = inStorage[i*4];
				tmp[i*3+1] = inStorage[i*4+1];
				tmp[i*3+2] = inStorage[i*4+2];
			}

			return tmp;
		}

		std::vector<float> dropAlpha_2c(Image& in) {
			std::vector<float> tmp(in.GetWidth() * in.GetHeight());
			auto inStorage = in.AccessStorage();

			for (size_t i = 0; i < in.GetWidth() * in.GetHeight(); i++) {
				tmp[i] = inStorage[i*2];
			}

			return tmp;
		}

		std::vector<float> appendAlpha_3c(Image& in) {
			std::vector<float> tmp(in.GetWidth() * in.GetHeight() * 4);
			auto inStorage = in.AccessStorage();

			for (size_t i = 0; i < in.GetWidth() * in.GetHeight(); i++) {
				tmp[i*4] = inStorage[i*3];
				tmp[i*4+1] = inStorage[i*3+1];
				tmp[i*4+2] = inStorage[i*3+2];
				tmp[i*4+3] = 255.0f;
			}

			return tmp;
		}

		std::vector<float> broadcastGray_to3c(Image& in) {
			std::vector<float> tmp(in.GetWidth() * in.GetHeight() * 3);
			auto inStorage = in.AccessStorage();

			for (size_t i = 0; i < in.GetWidth() * in.GetHeight(); i++) {
				tmp[i*3] = inStorage[i];
				tmp[i*3+1] = inStorage[i];
				tmp[i*3+2] = inStorage[i];
			}

			return tmp;
		}

		std::vector<float> broadcastGray_to4c(Image& in) {
			std::vector<float> tmp(in.GetWidth() * in.GetHeight() * 4);
			auto inStorage = in.AccessStorage();

			for (size_t i = 0; i < in.GetWidth() * in.GetHeight(); i++) {
				tmp[i*4] = inStorage[i];
				tmp[i*4+1] = inStorage[i];
				tmp[i*4+2] = inStorage[i];
				tmp[i*4+3] = 255.0f;
			}

			return tmp;
		}

		std::vector<float> broadcastGrayAlpha_to3c(Image& in) {
			std::vector<float> tmp(in.GetWidth() * in.GetHeight() * 3);
			auto inStorage = in.AccessStorage();

			for (size_t i = 0; i < in.GetWidth() * in.GetHeight(); i++) {
				tmp[i*3] = inStorage[i*2];
				tmp[i*3+1] = inStorage[i*2];
				tmp[i*3+2] = inStorage[i*2];
			}

			return tmp;
		}

		std::vector<float> broadcastGrayAlpha_to4c(Image& in) {
			std::vector<float> tmp(in.GetWidth() * in.GetHeight() * 4);
			auto inStorage = in.AccessStorage();

			for (size_t i = 0; i < in.GetWidth() * in.GetHeight(); i++) {
				tmp[i*4] = inStorage[i*2];
				tmp[i*4+1] = inStorage[i*2];
				tmp[i*4+2] = inStorage[i*2];
				tmp[i*4+3] = inStorage[i*2+1];
			}

			return tmp;
		}
	} // namespace detail


	void ConvertToRGB(ImageSoA& in) {
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

	void ConvertToBGR(ImageSoA& in) {
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

	void ConvertToRGBA(ImageSoA& in) {
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

	void ConvertToBGRA(ImageSoA& in) {
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

	void ConvertToRGB(Image& in) {
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

	void ConvertToBGR(Image& in) {
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

	void ConvertToRGBA(Image& in) {
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

	void ConvertToBGRA(Image& in) {
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

} // namespace wlib::image

#endif