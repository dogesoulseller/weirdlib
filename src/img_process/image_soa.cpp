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
	ImageSoA::ImageSoA(const Image& img) {
		format = img.GetFormat();
		width = img.GetWidth();
		height = img.GetHeight();

		#if WEIRDLIB_MULTITHREADING_MODE == WEIRDLIB_MTMODE_TBB
		tbb::task_scheduler_init tSched(8);
		#endif

		switch (format)
		{
		case ColorFormat::F_Grayscale: {
			alignas(64) auto c0 = new float[width*height];
			auto source = img.GetPixels();

			#if WEIRDLIB_MULTITHREADING_MODE == WEIRDLIB_MTMODE_TBB
			tbb::parallel_for(tbb::blocked_range<size_t>(0, width*height, 10000), [&c0, &source](tbb::blocked_range<size_t>& i){
				for (size_t j = i.begin(); j != i.end(); j++) {
					c0[j] = static_cast<float>(source[j]);
				}
			});
			#else
			#if WEIRDLIB_MULTITHREADING_MODE == WEIRDLIB_MTMODE_OMP
			#pragma omp parallel for simd num_threads(8)
			#endif
			for (size_t i = 0; i < width * height; i++) {
				c0[i] = static_cast<float>(source[i]);
			}
			#endif

			channels.push_back(c0);

			break;
		}
		case ColorFormat::F_GrayAlpha: {
			alignas(64) auto c0 = new float[width*height];
			alignas(64) auto c1 = new float[width*height];
			auto source = img.GetPixels();

			#if WEIRDLIB_MULTITHREADING_MODE == WEIRDLIB_MTMODE_TBB
			tbb::parallel_for(tbb::blocked_range<size_t>(0, width*height, 10000), [&c0, &c1, &source](tbb::blocked_range<size_t>& i){
				for (size_t j = i.begin(); j != i.end(); j++) {
					c0[j] = static_cast<float>(source[j*2]);
					c1[j] = static_cast<float>(source[j*2+1]);
				}
			});
			#else
			#if WEIRDLIB_MULTITHREADING_MODE == WEIRDLIB_MTMODE_OMP
			#pragma omp parallel for simd num_threads(8)
			#endif
			for (size_t i = 0; i < width * height; i++) {
				c0[i] = static_cast<float>(source[i*2]);
				c1[i] = static_cast<float>(source[i*2+1]);
			}
			#endif

			channels.push_back(c0);
			channels.push_back(c1);

			break;
		}
		case ColorFormat::F_RGB:
		case ColorFormat::F_BGR: {
			alignas(64) auto c0 = new float[width*height];
			alignas(64) auto c1 = new float[width*height];
			alignas(64) auto c2 = new float[width*height];
			auto source = img.GetPixels();

			#if WEIRDLIB_MULTITHREADING_MODE == WEIRDLIB_MTMODE_TBB
			tbb::parallel_for(tbb::blocked_range<size_t>(0, width*height, 10000), [&c0, &c1, &c2, &source](tbb::blocked_range<size_t>& i){
				for (size_t j = i.begin(); j != i.end(); j++) {
					c0[j] = static_cast<float>(source[j*3]);
					c1[j] = static_cast<float>(source[j*3+1]);
					c2[j] = static_cast<float>(source[j*3+2]);
				}
			});
			#else
			#if WEIRDLIB_MULTITHREADING_MODE == WEIRDLIB_MTMODE_OMP
			#pragma omp parallel for simd num_threads(8)
			#endif
			for (size_t i = 0; i < width * height; i++) {
				c0[i] = static_cast<float>(source[i*3]);
				c1[i] = static_cast<float>(source[i*3+1]);
				c2[i] = static_cast<float>(source[i*3+2]);
			}
			#endif

			channels.push_back(c0);
			channels.push_back(c1);
			channels.push_back(c2);

			break;
		}
		case ColorFormat::F_RGBA:
		case ColorFormat::F_BGRA:
		case ColorFormat::F_Default: {
			alignas(64) auto c0 = new float[width*height];
			alignas(64) auto c1 = new float[width*height];
			alignas(64) auto c2 = new float[width*height];
			alignas(64) auto c3 = new float[width*height];
			auto source = img.GetPixels();

			#if WEIRDLIB_MULTITHREADING_MODE == WEIRDLIB_MTMODE_TBB
			tbb::parallel_for(tbb::blocked_range<size_t>(0, width*height, 10000), [&c0, &c1, &c2, &c3, &source](tbb::blocked_range<size_t>& i){
				for (size_t j = i.begin(); j != i.end(); j++) {
					c0[j] = static_cast<float>(source[j*4]);
					c1[j] = static_cast<float>(source[j*4+1]);
					c2[j] = static_cast<float>(source[j*4+2]);
					c3[j] = static_cast<float>(source[j*4+3]);
				}
			});
			#else
			#if WEIRDLIB_MULTITHREADING_MODE == WEIRDLIB_MTMODE_OMP
			#pragma omp parallel for simd num_threads(8)
			#endif
			for (size_t i = 0; i < width * height; i++) {
				c0[i] = static_cast<float>(source[i*4]);
				c1[i] = static_cast<float>(source[i*4+1]);
				c2[i] = static_cast<float>(source[i*4+2]);
				c3[i] = static_cast<float>(source[i*4+3]);
			}
			#endif

			channels.push_back(c0);
			channels.push_back(c1);
			channels.push_back(c2);
			channels.push_back(c3);

			break;
		}
		default:
			break;
		}
	}

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

	Image ImageSoA::ConvertToImage() {
		alignas(64) float* outputPix;

		switch (format)
		{
		case ColorFormat::F_Grayscale:
			outputPix = new float[width*height];
			std::copy(channels[0], channels[0]+width*height, outputPix);
			break;
		case ColorFormat::F_GrayAlpha:
			outputPix = new float[width*height*2];
			#pragma ivdep
			for (size_t i = 0; i < width * height; i++) {
				outputPix[i*2] = channels[0][i];
				outputPix[i*2+1] = channels[1][i];
			}
			break;
		case ColorFormat::F_RGB:
		case ColorFormat::F_BGR:
			outputPix = new float[width*height*3];
			#pragma ivdep
			for (size_t i = 0; i < width * height; i++) {
				outputPix[i*3] = channels[0][i];
				outputPix[i*3+1] = channels[1][i];
				outputPix[i*3+2] = channels[2][i];
			}
			break;
		case ColorFormat::F_Default:
		case ColorFormat::F_RGBA:
		case ColorFormat::F_BGRA:
			outputPix = new float[width*height*4];
			#pragma ivdep
			for (size_t i = 0; i < width * height; i++) {
				outputPix[i*4] = channels[0][i];
				outputPix[i*4+1] = channels[1][i];
				outputPix[i*4+2] = channels[2][i];
				outputPix[i*4+3] = channels[3][i];
			}
			break;
		}

		Image imgOut(outputPix, width, height, format);
		delete[] outputPix;
		return imgOut;
	}

} // namespace wlib::image

#endif
