#include "../../include/weirdlib_image.hpp"

namespace wlib::image::detail
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
} // namespace wlib::image::detail
