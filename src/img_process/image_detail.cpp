#ifdef WEIRDLIB_ENABLE_IMAGE_OPERATIONS
#include "../../include/weirdlib_image.hpp"
#include <utility>
#include <algorithm>

namespace wlib::image::detail
{
	void swapRAndB_3c(float* in, size_t count) {
		for (size_t i = 0; i < count; i++) {
			std::swap(in[i*3], in[i*3+2]);
		}
	}

	void swapRAndB_4c(float* in, size_t count) {
		for (size_t i = 0; i < count; i++) {
			std::swap(in[i*4], in[i*4+2]);
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

		std::generate(tmp.begin(), tmp.end(), [i = 0, inStorage = in.AccessStorage()] () mutable { return inStorage[i*2]; } );

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

	void extendGSTo3Chan(ImageSoA& in) {
		for (int_fast8_t i = 1; i <= 2; i++) {
			alignas(64) auto tmp = new float[in.GetWidth()*in.GetHeight()];
			std::copy(in.AccessChannels()[0], in.AccessChannels()[0]+in.GetWidth()*in.GetHeight(), tmp);
			in.AccessChannels()[i] = tmp;
		}
	}

	void extendGSTo4Chan(ImageSoA& in, bool constantAlpha) {
		for (int_fast8_t i = 1; i <= 2; i++) {
			alignas(64) auto tmp = new float[in.GetWidth()*in.GetHeight()];
			std::copy(in.AccessChannels()[0], in.AccessChannels()[0]+in.GetWidth()*in.GetHeight(), tmp);
			in.AccessChannels()[i] = tmp;
		}

		if (constantAlpha) {
			alignas(64) auto tmp = new float[in.GetWidth()*in.GetHeight()];
			std::uninitialized_fill(tmp, tmp + in.GetWidth() * in.GetHeight(), 255.0f);
			in.AccessChannels()[3] = tmp;
		} else {
			in.AccessChannels()[3] = in.AccessChannels()[1];
		}
	}

	void appendConstantAlpha(ImageSoA& in) {
		alignas(64) auto tmp = new float[in.GetWidth() * in.GetHeight()];
		std::uninitialized_fill(tmp, tmp + in.GetWidth() * in.GetHeight(), 255.0f);
		in.AccessChannels().push_back(tmp);
	}

} // namespace wlib::image::detail
#endif
