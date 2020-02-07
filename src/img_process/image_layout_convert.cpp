#ifdef WEIRDLIB_ENABLE_IMAGE_OPERATIONS
#include "../../include/weirdlib_image.hpp"
#if WEIRDLIB_MULTITHREADING_MODE == WEIRDLIB_MTMODE_TBB
	#include <tbb/tbb.h>
#endif

namespace wlib::image
{
	Image MakeAoSFromSoA(const ImageSoA& in) {
		alignas(64) float* outputPix = nullptr;

		switch (in.GetFormat())
		{
		  case ColorFormat::F_Grayscale:
			outputPix = new float[in.GetWidth()*in.GetHeight()];
			std::copy(in.GetChannels()[0], in.GetChannels()[0]+in.GetWidth()*in.GetHeight(), outputPix);
			break;
		  case ColorFormat::F_GrayAlpha:
			outputPix = new float[in.GetWidth()*in.GetHeight()*2];
			#pragma ivdep
			for (size_t i = 0; i < in.GetWidth() * in.GetHeight(); i++) {
				outputPix[i*2] = in.GetChannels()[0][i];
				outputPix[i*2+1] = in.GetChannels()[1][i];
			}
			break;
		  case ColorFormat::F_RGB:
		  case ColorFormat::F_BGR:
			outputPix = new float[in.GetWidth()*in.GetHeight()*3];
			#pragma ivdep
			for (size_t i = 0; i < in.GetWidth() * in.GetHeight(); i++) {
				outputPix[i*3] = in.GetChannels()[0][i];
				outputPix[i*3+1] = in.GetChannels()[1][i];
				outputPix[i*3+2] = in.GetChannels()[2][i];
			}
			break;
		  case ColorFormat::F_Default:
		  case ColorFormat::F_RGBA:
		  case ColorFormat::F_BGRA:
			outputPix = new float[in.GetWidth()*in.GetHeight()*4];
			#pragma ivdep
			for (size_t i = 0; i < in.GetWidth() * in.GetHeight(); i++) {
				outputPix[i*4] = in.GetChannels()[0][i];
				outputPix[i*4+1] = in.GetChannels()[1][i];
				outputPix[i*4+2] = in.GetChannels()[2][i];
				outputPix[i*4+3] = in.GetChannels()[3][i];
			}
			break;
		}

		Image imgOut(outputPix, in.GetWidth(), in.GetHeight(), in.GetFormat());
		delete[] outputPix;
		return imgOut;
	}

	ImageSoA MakeSoAFromAoS(const Image& img) {
		ImageSoA outimg;

		outimg.SetFormat(img.GetFormat());
		outimg.SetWidth(img.GetWidth());
		outimg.SetHeight(img.GetHeight());

		#if WEIRDLIB_MULTITHREADING_MODE == WEIRDLIB_MTMODE_TBB
		tbb::task_scheduler_init tSched(8);
		#endif

		switch (outimg.GetFormat())
		{
		  case ColorFormat::F_Grayscale: {
			alignas(64) auto c0 = new float[outimg.GetWidth()*outimg.GetHeight()];
			auto source = img.GetPixels();

			#if WEIRDLIB_MULTITHREADING_MODE == WEIRDLIB_MTMODE_TBB
			tbb::parallel_for(tbb::blocked_range<size_t>(0, outimg.GetWidth()*outimg.GetHeight(), 10000), [&c0, &source](tbb::blocked_range<size_t>& i){
				for (size_t j = i.begin(); j != i.end(); j++) {
					c0[j] = static_cast<float>(source[j]);
				}
			});
			#else
			#if WEIRDLIB_MULTITHREADING_MODE == WEIRDLIB_MTMODE_OMP
			#pragma omp parallel for simd num_threads(8)
			#endif
			for (size_t i = 0; i < outimg.GetWidth() * outimg.GetHeight(); i++) {
				c0[i] = static_cast<float>(source[i]);
			}
			#endif

			outimg.AccessChannels().push_back(c0);

			break;
		  }
		  case ColorFormat::F_GrayAlpha: {
			alignas(64) auto c0 = new float[outimg.GetWidth()*outimg.GetHeight()];
			alignas(64) auto c1 = new float[outimg.GetWidth()*outimg.GetHeight()];
			auto source = img.GetPixels();

			#if WEIRDLIB_MULTITHREADING_MODE == WEIRDLIB_MTMODE_TBB
			tbb::parallel_for(tbb::blocked_range<size_t>(0, outimg.GetWidth()*outimg.GetHeight(), 10000), [&c0, &c1, &source](tbb::blocked_range<size_t>& i){
				for (size_t j = i.begin(); j != i.end(); j++) {
					c0[j] = static_cast<float>(source[j*2]);
					c1[j] = static_cast<float>(source[j*2+1]);
				}
			});
			#else
			#if WEIRDLIB_MULTITHREADING_MODE == WEIRDLIB_MTMODE_OMP
			#pragma omp parallel for simd num_threads(8)
			#endif
			for (size_t i = 0; i < outimg.GetWidth() * outimg.GetHeight(); i++) {
				c0[i] = static_cast<float>(source[i*2]);
				c1[i] = static_cast<float>(source[i*2+1]);
			}
			#endif

			outimg.AccessChannels().push_back(c0);
			outimg.AccessChannels().push_back(c1);

			break;
		  }
		  case ColorFormat::F_RGB:
		  case ColorFormat::F_BGR: {
			alignas(64) auto c0 = new float[outimg.GetWidth()*outimg.GetHeight()];
			alignas(64) auto c1 = new float[outimg.GetWidth()*outimg.GetHeight()];
			alignas(64) auto c2 = new float[outimg.GetWidth()*outimg.GetHeight()];
			auto source = img.GetPixels();

			#if WEIRDLIB_MULTITHREADING_MODE == WEIRDLIB_MTMODE_TBB
			tbb::parallel_for(tbb::blocked_range<size_t>(0, outimg.GetWidth()*outimg.GetHeight(), 10000), [&c0, &c1, &c2, &source](tbb::blocked_range<size_t>& i){
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
			for (size_t i = 0; i < outimg.GetWidth() * outimg.GetHeight(); i++) {
				c0[i] = static_cast<float>(source[i*3]);
				c1[i] = static_cast<float>(source[i*3+1]);
				c2[i] = static_cast<float>(source[i*3+2]);
			}
			#endif

			outimg.AccessChannels().push_back(c0);
			outimg.AccessChannels().push_back(c1);
			outimg.AccessChannels().push_back(c2);

			break;
		  }
		  case ColorFormat::F_RGBA:
		  case ColorFormat::F_BGRA:
		  case ColorFormat::F_Default: {
			alignas(64) auto c0 = new float[outimg.GetWidth()*outimg.GetHeight()];
			alignas(64) auto c1 = new float[outimg.GetWidth()*outimg.GetHeight()];
			alignas(64) auto c2 = new float[outimg.GetWidth()*outimg.GetHeight()];
			alignas(64) auto c3 = new float[outimg.GetWidth()*outimg.GetHeight()];
			auto source = img.GetPixels();

			#if WEIRDLIB_MULTITHREADING_MODE == WEIRDLIB_MTMODE_TBB
			tbb::parallel_for(tbb::blocked_range<size_t>(0, outimg.GetWidth()*outimg.GetHeight(), 10000), [&c0, &c1, &c2, &c3, &source](tbb::blocked_range<size_t>& i){
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
			for (size_t i = 0; i < outimg.GetWidth() * outimg.GetHeight(); i++) {
				c0[i] = static_cast<float>(source[i*4]);
				c1[i] = static_cast<float>(source[i*4+1]);
				c2[i] = static_cast<float>(source[i*4+2]);
				c3[i] = static_cast<float>(source[i*4+3]);
			}
			#endif

			outimg.AccessChannels().push_back(c0);
			outimg.AccessChannels().push_back(c1);
			outimg.AccessChannels().push_back(c2);
			outimg.AccessChannels().push_back(c3);

			break;
		  }
		  default:
			break;
		}

		return outimg;
	}
} // namespace wlib::image
#endif
