#include "../../include/weirdlib_image.hpp"
#if WEIRDLIB_MULTITHREADING_MODE == WEIRDLIB_MTMODE_TBB
	#include <tbb/tbb.h>
#endif

namespace wlib::image
{
	Image MakeAoSFromSoA(const ImageSoA& in) {
		alignas(64) float* outputPix = nullptr;

		switch (in.format)
		{
		case ColorFormat::F_Grayscale:
			outputPix = new float[in.width*in.height];
			std::copy(in.channels[0], in.channels[0]+in.width*in.height, outputPix);
			break;
		case ColorFormat::F_GrayAlpha:
			outputPix = new float[in.width*in.height*2];
			#pragma ivdep
			for (size_t i = 0; i < in.width * in.height; i++) {
				outputPix[i*2] = in.channels[0][i];
				outputPix[i*2+1] = in.channels[1][i];
			}
			break;
		case ColorFormat::F_RGB:
		case ColorFormat::F_BGR:
			outputPix = new float[in.width*in.height*3];
			#pragma ivdep
			for (size_t i = 0; i < in.width * in.height; i++) {
				outputPix[i*3] = in.channels[0][i];
				outputPix[i*3+1] = in.channels[1][i];
				outputPix[i*3+2] = in.channels[2][i];
			}
			break;
		case ColorFormat::F_Default:
		case ColorFormat::F_RGBA:
		case ColorFormat::F_BGRA:
			outputPix = new float[in.width*in.height*4];
			#pragma ivdep
			for (size_t i = 0; i < in.width * in.height; i++) {
				outputPix[i*4] = in.channels[0][i];
				outputPix[i*4+1] = in.channels[1][i];
				outputPix[i*4+2] = in.channels[2][i];
				outputPix[i*4+3] = in.channels[3][i];
			}
			break;
		}

		Image imgOut(outputPix, in.width, in.height, in.format);
		delete[] outputPix;
		return imgOut;
	}

	ImageSoA MakeSoAFromAoS(const Image& img) {
		ImageSoA outimg;

		outimg.format = img.GetFormat();
		outimg.width = img.GetWidth();
		outimg.height = img.GetHeight();

		#if WEIRDLIB_MULTITHREADING_MODE == WEIRDLIB_MTMODE_TBB
		tbb::task_scheduler_init tSched(8);
		#endif

		switch (outimg.format)
		{
		case ColorFormat::F_Grayscale: {
			alignas(64) auto c0 = new float[outimg.width*outimg.height];
			auto source = img.GetPixels();

			#if WEIRDLIB_MULTITHREADING_MODE == WEIRDLIB_MTMODE_TBB
			tbb::parallel_for(tbb::blocked_range<size_t>(0, outimg.width*outimg.height, 10000), [&c0, &source](tbb::blocked_range<size_t>& i){
				for (size_t j = i.begin(); j != i.end(); j++) {
					c0[j] = static_cast<float>(source[j]);
				}
			});
			#else
			#if WEIRDLIB_MULTITHREADING_MODE == WEIRDLIB_MTMODE_OMP
			#pragma omp parallel for simd num_threads(8)
			#endif
			for (size_t i = 0; i < outimg.width * outimg.height; i++) {
				c0[i] = static_cast<float>(source[i]);
			}
			#endif

			outimg.channels.push_back(c0);

			break;
		}
		case ColorFormat::F_GrayAlpha: {
			alignas(64) auto c0 = new float[outimg.width*outimg.height];
			alignas(64) auto c1 = new float[outimg.width*outimg.height];
			auto source = img.GetPixels();

			#if WEIRDLIB_MULTITHREADING_MODE == WEIRDLIB_MTMODE_TBB
			tbb::parallel_for(tbb::blocked_range<size_t>(0, outimg.width*outimg.height, 10000), [&c0, &c1, &source](tbb::blocked_range<size_t>& i){
				for (size_t j = i.begin(); j != i.end(); j++) {
					c0[j] = static_cast<float>(source[j*2]);
					c1[j] = static_cast<float>(source[j*2+1]);
				}
			});
			#else
			#if WEIRDLIB_MULTITHREADING_MODE == WEIRDLIB_MTMODE_OMP
			#pragma omp parallel for simd num_threads(8)
			#endif
			for (size_t i = 0; i < outimg.width * outimg.height; i++) {
				c0[i] = static_cast<float>(source[i*2]);
				c1[i] = static_cast<float>(source[i*2+1]);
			}
			#endif

			outimg.channels.push_back(c0);
			outimg.channels.push_back(c1);

			break;
		}
		case ColorFormat::F_RGB:
		case ColorFormat::F_BGR: {
			alignas(64) auto c0 = new float[outimg.width*outimg.height];
			alignas(64) auto c1 = new float[outimg.width*outimg.height];
			alignas(64) auto c2 = new float[outimg.width*outimg.height];
			auto source = img.GetPixels();

			#if WEIRDLIB_MULTITHREADING_MODE == WEIRDLIB_MTMODE_TBB
			tbb::parallel_for(tbb::blocked_range<size_t>(0, outimg.width*outimg.height, 10000), [&c0, &c1, &c2, &source](tbb::blocked_range<size_t>& i){
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
			for (size_t i = 0; i < outimg.width * outimg.height; i++) {
				c0[i] = static_cast<float>(source[i*3]);
				c1[i] = static_cast<float>(source[i*3+1]);
				c2[i] = static_cast<float>(source[i*3+2]);
			}
			#endif

			outimg.channels.push_back(c0);
			outimg.channels.push_back(c1);
			outimg.channels.push_back(c2);

			break;
		}
		case ColorFormat::F_RGBA:
		case ColorFormat::F_BGRA:
		case ColorFormat::F_Default: {
			alignas(64) auto c0 = new float[outimg.width*outimg.height];
			alignas(64) auto c1 = new float[outimg.width*outimg.height];
			alignas(64) auto c2 = new float[outimg.width*outimg.height];
			alignas(64) auto c3 = new float[outimg.width*outimg.height];
			auto source = img.GetPixels();

			#if WEIRDLIB_MULTITHREADING_MODE == WEIRDLIB_MTMODE_TBB
			tbb::parallel_for(tbb::blocked_range<size_t>(0, outimg.width*outimg.height, 10000), [&c0, &c1, &c2, &c3, &source](tbb::blocked_range<size_t>& i){
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
			for (size_t i = 0; i < outimg.width * outimg.height; i++) {
				c0[i] = static_cast<float>(source[i*4]);
				c1[i] = static_cast<float>(source[i*4+1]);
				c2[i] = static_cast<float>(source[i*4+2]);
				c3[i] = static_cast<float>(source[i*4+3]);
			}
			#endif

			outimg.channels.push_back(c0);
			outimg.channels.push_back(c1);
			outimg.channels.push_back(c2);
			outimg.channels.push_back(c3);

			break;
		}
		default:
			break;
		}

		return outimg;
	}
} // namespace wlib::image
