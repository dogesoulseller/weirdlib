#include "../../include/weirdlib_image.hpp"
#include "../../include/weirdlib_traits.hpp"
#include "../../include/cpu_detection.hpp"

#include "../common.hpp"
#include <cmath>
#include <algorithm>
#include <thread>
#include <array>

#ifdef WEIRDLIB_ENABLE_IMAGE_OPERATIONS

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

	ImageSoA& ImageSoA::operator=(const ImageSoA& img) {
		if (&img == this) {
			return *this;
		}

		height = img.height;
		width = img.width;
		format = img.format;

		for (auto& ptr : channels) {
			delete[] ptr;
		}

		if (channels.size() != img.channels.size()) {
			channels.resize(img.channels.size());
			channels.shrink_to_fit();
		}

		for (size_t i = 0; i < channels.size(); i++) {
			auto chan = new float[width * height];
			std::copy(img.channels[i], img.channels[i] + width * height, chan);
			channels[i] = chan;
		}

		return *this;
	}

	ImageSoA::~ImageSoA() {
		for (auto& c : channels) {
			delete[] c;
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

	ImageSoA& ConvertToGrayscale(ImageSoA& inImg, const bool preserveAlpha, const GrayscaleMethod method) {
		// Converting to grayscale from grayscale...
		if (inImg.format == F_GrayAlpha || inImg.format == F_Grayscale) {
			return inImg;
		}

		size_t channelRedN;
		size_t channelGreenN = 1;
		size_t channelBlueN;

		if (inImg.format == F_RGB || inImg.format == F_RGBA) {
			channelRedN = 0;
			channelBlueN = 2;
		} else {
			channelRedN = 2;
			channelBlueN = 0;
		}

		alignas(32) auto outGr = new float[inImg.width * inImg.height];

		switch (method)
		{
		case GrayscaleMethod::Luminosity:
		{
			// Processing with SIMD
			#if X86_SIMD_LEVEL >= LV_AVX
				const auto redMul = _mm256_set1_ps(0.2126f);
				const auto greenMul = _mm256_set1_ps(0.7152f);
				const auto blueMul = _mm256_set1_ps(0.0722f);
				const auto maxMask = _mm256_set1_ps(255.0f);

				const auto iter_AVX = (inImg.width * inImg.height) / 8;
				const auto iterRem_AVX = (inImg.width * inImg.height) % 8;

				for (size_t i = 0; i < iter_AVX; i++) {
					const auto rChan = _mm256_loadu_ps(inImg.channels[channelRedN]+i*8);
					const auto gChan = _mm256_loadu_ps(inImg.channels[channelGreenN]+i*8);
					const auto bChan = _mm256_loadu_ps(inImg.channels[channelBlueN]+i*8);

					#ifdef X86_SIMD_FMA
						const auto resultGr = _mm256_min_ps(_mm256_fmadd_ps(rChan, redMul, _mm256_fmadd_ps(gChan, greenMul, _mm256_mul_ps(bChan, blueMul))), maxMask);
					#else
						const auto resultGr = _mm256_min_ps(
							_mm256_add_ps(_mm256_mul_ps(rChan, redMul), _mm256_add_ps(_mm256_mul_ps(gChan, greenMul), _mm256_mul_ps(bChan, blueMul))), maxMask);
					#endif
						_mm256_storeu_ps(outGr+8*i, resultGr);
				}

				for (size_t i = iter_AVX * 8; i < iterRem_AVX + iter_AVX*8; i++) {
					if (const auto result = std::fma(inImg.channels[channelRedN][i], 0.2126f, std::fma(inImg.channels[channelGreenN][i], 0.7152f, inImg.channels[channelBlueN][i] * 0.0722f)); result > 255.0f) {
						outGr[i] = 255.0f;
					} else {
						outGr[i] = result;
					}
				}

				_mm256_zeroupper();
			#elif X86_SIMD_LEVEL >= LV_SSE
				const auto redMul = _mm_set1_ps(0.2126f);
				const auto greenMul = _mm_set1_ps(0.7152f);
				const auto blueMul = _mm_set1_ps(0.0722f);
				const auto maxMask = _mm_set1_ps(255.0f);

				const auto iter_SSE = (inImg.width * inImg.height) / 4;
				const auto iterRem_SSE = (inImg.width * inImg.height) % 4;

				for (size_t i = 0; i < iter_SSE; i++) {
					const auto rChan = _mm_loadu_ps(inImg.channels[channelRedN]+i*4);
					const auto gChan = _mm_loadu_ps(inImg.channels[channelGreenN]+i*4);
					const auto bChan = _mm_loadu_ps(inImg.channels[channelBlueN]+i*4);

					#ifdef X86_SIMD_FMA	// Technically shouldn't be possible
						const auto resultGr = _mm_min_ps(_mm_fmadd_ps(rChan, redMul, _mm256_fmadd_ps(gChan, greenMul, _mm_mul_ps(bChan, blueMul))), maxMask);
					#else
						const auto resultGr = _mm_min_ps(
							_mm_add_ps(_mm_mul_ps(rChan, redMul), _mm_add_ps(_mm_mul_ps(gChan, greenMul), _mm_mul_ps(bChan, blueMul))), maxMask);
					#endif
						_mm_storeu_ps(outGr+4*i, resultGr);
				}

				for (size_t i = iter_SSE * 4; i < iterRem_SSE + iter_SSE * 4; i++) {
					if (const auto result = std::fma(inImg.channels[channelRedN][i], 0.2126f, std::fma(inImg.channels[channelGreenN][i], 0.7152f, inImg.channels[channelBlueN][i] * 0.0722f)); result > 255.0f) {
						outGr[i] = 255.0f;
					} else {
						outGr[i] = result;
					}
				}
			#else
				for (size_t i = 0; i < inImg.width * inImg.height; i++) {
					if (const auto result = std::fma(inImg.channels[channelRedN][i], 0.2126f, std::fma(inImg.channels[channelGreenN][i], 0.7152f, inImg.channels[channelBlueN][i] * 0.0722f)); result > 255.0f) {
						outGr[i] = 255.0f;
					} else {
						outGr[i] = result;
					}
				}
			#endif
			break;
		}
		case GrayscaleMethod::Lightness:
		{
			// Processing with SIMD
			#if X86_SIMD_LEVEL >= LV_AVX
				const auto maxMask = _mm256_set1_ps(255.0f);
				const auto divMask = _mm256_set1_ps(0.5f);

				const auto iter_AVX = (inImg.width * inImg.height) / 8;
				const auto iterRem_AVX = (inImg.width * inImg.height) % 8;

				for (size_t i = 0; i < iter_AVX; i++) {
					const auto rChan = _mm256_loadu_ps(inImg.channels[channelRedN]+i*8);
					const auto gChan = _mm256_loadu_ps(inImg.channels[channelGreenN]+i*8);
					const auto bChan = _mm256_loadu_ps(inImg.channels[channelBlueN]+i*8);

					const auto maxValues = _mm256_max_ps(rChan, _mm256_max_ps(gChan, bChan));
					const auto minValues = _mm256_min_ps(rChan, _mm256_min_ps(gChan, bChan));

					#ifdef X86_SIMD_FMA
					const auto resultGr = _mm256_min_ps(_mm256_fmadd_ps(maxValues, minValues, divMask), maxMask);
					#else
					const auto resultGr = _mm256_min_ps(_mm256_mul_ps(_mm256_add_ps(maxValues, minValues), divMask), maxMask);
					#endif

					_mm256_storeu_ps(outGr+8*i, resultGr);
				}

				for (size_t i = iter_AVX * 8; i < iterRem_AVX + iter_AVX*8; i++) {
					const auto maxValue = std::max(std::max(inImg.channels[channelRedN][i], inImg.channels[channelGreenN][i]), inImg.channels[channelBlueN][i]);
					const auto minValue = std::min(std::min(inImg.channels[channelRedN][i], inImg.channels[channelGreenN][i]), inImg.channels[channelBlueN][i]);
					const auto result = (maxValue + minValue) * 0.5f;

					if (result > 255.0f) {
						outGr[i] = 255.0f;
					} else {
						outGr[i] = result;
					}
				}

				_mm256_zeroupper();
			#elif X86_SIMD_LEVEL >= LV_SSE
				const auto maxMask = _mm_set1_ps(255.0f);
				const auto divMask = _mm_set1_ps(0.5f);

				const auto iter_SSE = (inImg.width * inImg.height) / 4;
				const auto iterRem_SSE = (inImg.width * inImg.height) % 4;

				for (size_t i = 0; i < iter_SSE; i++) {
					const auto rChan = _mm_loadu_ps(inImg.channels[channelRedN]+i*4);
					const auto gChan = _mm_loadu_ps(inImg.channels[channelGreenN]+i*4);
					const auto bChan = _mm_loadu_ps(inImg.channels[channelBlueN]+i*4);

					const auto maxValues = _mm_max_ps(rChan, _mm_max_ps(gChan, bChan));
					const auto minValues = _mm_min_ps(rChan, _mm_min_ps(gChan, bChan));

					const auto resultGr = _mm_min_ps(_mm_mul_ps(_mm_add_ps(maxValues, minValues), divMask), maxMask);

					_mm_storeu_ps(outGr+4*i, resultGr);
				}

				for (size_t i = iter_SSE * 4; i < iterRem_SSE + iter_SSE*4; i++) {
					const auto maxValue = std::max(std::max(inImg.channels[channelRedN][i], inImg.channels[channelGreenN][i]), inImg.channels[channelBlueN][i]);
					const auto minValue = std::min(std::min(inImg.channels[channelRedN][i], inImg.channels[channelGreenN][i]), inImg.channels[channelBlueN][i]);
					const auto result = (maxValue + minValue) * 0.5f;

					if (result > 255.0f) {
						outGr[i] = 255.0f;
					} else {
						outGr[i] = result;
					}
				}

			#else
				for (size_t i = 0; i < inImg.width * inImg.height; i++) {
					const auto maxValue = std::max(std::max(inImg.channels[channelRedN][i], inImg.channels[channelGreenN][i]), inImg.channels[channelBlueN][i]);
					const auto minValue = std::min(std::min(inImg.channels[channelRedN][i], inImg.channels[channelGreenN][i]), inImg.channels[channelBlueN][i]);
					const auto result = (maxValue + minValue) * 0.5f;

					if (result > 255.0f) {
						outGr[i] = 255.0f;
					} else {
						outGr[i] = result;
					}
				}
			#endif
			break;
		}
		case GrayscaleMethod::Average:
		{
			// Processing with SIMD
			#if X86_SIMD_LEVEL >= LV_AVX
				const auto maxMask = _mm256_set1_ps(255.0f);
				const auto divMask = _mm256_set1_ps(0.3333333333f);

				const auto iter_AVX = (inImg.width * inImg.height) / 8;
				const auto iterRem_AVX = (inImg.width * inImg.height) % 8;

				for (size_t i = 0; i < iter_AVX; i++) {
					const auto rChan = _mm256_loadu_ps(inImg.channels[channelRedN]+i*8);
					const auto gChan = _mm256_loadu_ps(inImg.channels[channelGreenN]+i*8);
					const auto bChan = _mm256_loadu_ps(inImg.channels[channelBlueN]+i*8);

					#ifdef X86_SIMD_FMA
					const auto resultGr = _mm256_min_ps(_mm256_fmadd_ps(_mm256_add_ps(rChan, gChan), bChan, divMask), maxMask);
					#else
					const auto resultGr = _mm256_min_ps(_mm256_mul_ps(_mm256_add_ps(rChan, _mm256_add_ps(gChan, bChan)), divMask), maxMask);
					#endif

					_mm256_storeu_ps(outGr+8*i, resultGr);
				}

				for (size_t i = iter_AVX * 8; i < iterRem_AVX + iter_AVX*8; i++) {
					const auto result = (inImg.channels[channelRedN][i] + inImg.channels[channelGreenN][i] + inImg.channels[channelBlueN][i]) * 0.3333333333f;

					if (result > 255.0f) {
						outGr[i] = 255.0f;
					} else {
						outGr[i] = result;
					}
				}

				_mm256_zeroupper();
			#elif X86_SIMD_LEVEL >= LV_SSE
				const auto maxMask = _mm_set1_ps(255.0f);
				const auto divMask = _mm_set1_ps(0.3333333333f);

				const auto iter_AVX = (inImg.width * inImg.height) / 4;
				const auto iterRem_AVX = (inImg.width * inImg.height) % 4;

				for (size_t i = 0; i < iter_AVX; i++) {
					const auto rChan = _mm_loadu_ps(inImg.channels[channelRedN]+i*4);
					const auto gChan = _mm_loadu_ps(inImg.channels[channelGreenN]+i*4);
					const auto bChan = _mm_loadu_ps(inImg.channels[channelBlueN]+i*4);

					const auto resultGr = _mm_min_ps(_mm_mul_ps(_mm_add_ps(rChan, _mm_add_ps(gChan, bChan)), divMask), maxMask);

					_mm_storeu_ps(outGr+4*i, resultGr);
				}

				for (size_t i = iter_AVX * 4; i < iterRem_AVX + iter_AVX*4; i++) {
					const auto result = (inImg.channels[channelRedN][i] + inImg.channels[channelGreenN][i] + inImg.channels[channelBlueN][i]) * 0.3333333333f;

					if (result > 255.0f) {
						outGr[i] = 255.0f;
					} else {
						outGr[i] = result;
					}
				}

			#else
				for (size_t i = 0; i < inImg.width * inImg.height; i++) {
					const auto result = (inImg.channels[channelRedN][i] + inImg.channels[channelGreenN][i] + inImg.channels[channelBlueN][i]) * 0.3333333333f;

					if (result > 255.0f) {
						outGr[i] = 255.0f;
					} else {
						outGr[i] = result;
					}
				}
			#endif
			break;
		}
		case GrayscaleMethod::LuminosityBT601:
		{
			// Processing with SIMD
			#if X86_SIMD_LEVEL >= LV_AVX
				const auto redMul = _mm256_set1_ps(0.299f);
				const auto greenMul = _mm256_set1_ps(0.587f);
				const auto blueMul = _mm256_set1_ps(0.114f);
				const auto maxMask = _mm256_set1_ps(255.0f);

				const auto iter_AVX = (inImg.width * inImg.height) / 8;
				const auto iterRem_AVX = (inImg.width * inImg.height) % 8;

				for (size_t i = 0; i < iter_AVX; i++) {
					const auto rChan = _mm256_loadu_ps(inImg.channels[channelRedN]+i*8);
					const auto gChan = _mm256_loadu_ps(inImg.channels[channelGreenN]+i*8);
					const auto bChan = _mm256_loadu_ps(inImg.channels[channelBlueN]+i*8);

					#ifdef X86_SIMD_FMA
						const auto resultGr = _mm256_min_ps(_mm256_fmadd_ps(rChan, redMul, _mm256_fmadd_ps(gChan, greenMul, _mm256_mul_ps(bChan, blueMul))), maxMask);
					#else
						const auto resultGr = _mm256_min_ps(
							_mm256_add_ps(_mm256_mul_ps(rChan, redMul), _mm256_add_ps(_mm256_mul_ps(gChan, greenMul), _mm256_mul_ps(bChan, blueMul))), maxMask);
					#endif
						_mm256_storeu_ps(outGr+8*i, resultGr);
				}

				for (size_t i = iter_AVX * 8; i < iterRem_AVX + iter_AVX*8; i++) {
					if (const auto result = std::fma(inImg.channels[channelRedN][i], 0.299f, std::fma(inImg.channels[channelGreenN][i], 0.587f, inImg.channels[channelBlueN][i] * 0.114f)); result > 255.0f) {
						outGr[i] = 255.0f;
					} else {
						outGr[i] = result;
					}
				}

				_mm256_zeroupper();
			#elif X86_SIMD_LEVEL >= LV_SSE
				const auto redMul = _mm_set1_ps(0.299f);
				const auto greenMul = _mm_set1_ps(0.587f);
				const auto blueMul = _mm_set1_ps(0.114f);
				const auto maxMask = _mm_set1_ps(255.0f);

				const auto iter_SSE = (inImg.width * inImg.height) / 4;
				const auto iterRem_SSE = (inImg.width * inImg.height) % 4;

				for (size_t i = 0; i < iter_SSE; i++) {
					const auto rChan = _mm_loadu_ps(inImg.channels[channelRedN]+i*4);
					const auto gChan = _mm_loadu_ps(inImg.channels[channelGreenN]+i*4);
					const auto bChan = _mm_loadu_ps(inImg.channels[channelBlueN]+i*4);

					#ifdef X86_SIMD_FMA	// Technically shouldn't be possible
						const auto resultGr = _mm_min_ps(_mm_fmadd_ps(rChan, redMul, _mm256_fmadd_ps(gChan, greenMul, _mm_mul_ps(bChan, blueMul))), maxMask);
					#else
						const auto resultGr = _mm_min_ps(
							_mm_add_ps(_mm_mul_ps(rChan, redMul), _mm_add_ps(_mm_mul_ps(gChan, greenMul), _mm_mul_ps(bChan, blueMul))), maxMask);
					#endif
						_mm_storeu_ps(outGr+4*i, resultGr);
				}

				for (size_t i = iter_SSE * 4; i < iterRem_SSE + iter_SSE * 4; i++) {
					if (const auto result = std::fma(inImg.channels[channelRedN][i], 0.299f, std::fma(inImg.channels[channelGreenN][i], 0.587f, inImg.channels[channelBlueN][i] * 0.114f)); result > 255.0f) {
						outGr[i] = 255.0f;
					} else {
						outGr[i] = result;
					}
				}
			#else
				for (size_t i = 0; i < inImg.width * inImg.height; i++) {
					if (const auto result = std::fma(inImg.channels[channelRedN][i], 0.299f, std::fma(inImg.channels[channelGreenN][i], 0.587f, inImg.channels[channelBlueN][i] * 0.114f)); result > 255.0f) {
						outGr[i] = 255.0f;
					} else {
						outGr[i] = result;
					}
				}
			#endif
			break;
		}
		default:
			break;
		}

		// Copy channels
		if (inImg.format == F_RGB || inImg.format == F_BGR || !preserveAlpha) {
			inImg.format = F_Grayscale;

			for (auto& ptr : inImg.channels) {
				delete[] ptr;
			}

			inImg.channels.resize(1);
			inImg.channels.shrink_to_fit();
			inImg.channels[0] = outGr;
		} else if (inImg.format == F_RGBA || inImg.format == F_BGRA) {
			inImg.format = F_GrayAlpha;

			for (size_t i = 0; i < inImg.channels.size()-1; i++) {
				delete[] inImg.channels[i];
			}

			auto imgAlpha = inImg.channels[3];

			inImg.channels.resize(2);
			inImg.channels.shrink_to_fit();
			inImg.channels[0] = outGr;
			inImg.channels[1] = imgAlpha;
		}

		return inImg;
	}

	// TODO: Work on 16 bytes simultaneously instead of reading the same chunk four times
	void ConvertUint8ToFloat(const uint8_t* in, float* out, const size_t fileSize) {
		#if X86_SIMD_LEVEL >= LV_AVX512
			size_t iters = fileSize / 16;
			size_t itersRem = fileSize % 16;

			for (size_t i = 0; i < iters; i++) {
				const __m128i pix_AVX512_8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(in + i*16));
				const __m512i pix_AVX512_32 = _mm512_cvtepu8_epi32(pix_AVX512_8);
				const __m512 pixf_AVX512 = _mm512_cvtepi32_ps(pix_AVX512_32);
				_mm512_storeu_ps(out + i*16, pixf_AVX512);
			}

			for (size_t i = iters * 16; i < iters * 16 + itersRem; i++) {
				out[i] = static_cast<float>(in[i]);
			}
		#elif X86_SIMD_LEVEL >= LV_AVX2
			size_t iters = fileSize / 32;
			size_t itersRem = fileSize % 32;

			for (size_t i = 0; i < iters; i++) {
				// Work on 32 bytes at once
				const __m256i pixin = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(in + i*32));
				// 128-bit lane swapped bytes
				const __m256i pixin_swap = _mm256_permute2x128_si256(pixin, pixin, 1);

				// First 8 bytes
				__m128i pixin_0 = _mm256_castsi256_si128(pixin);
				// Second 8 bytes
				__m128i pixin_1 = _mm_bsrli_si128(_mm256_castsi256_si128(pixin), 8);
				// Third 8 bytes
				__m128i pixin_2 = _mm256_castsi256_si128(pixin_swap);
				// Last 8 bytes
				__m128i pixin_3 = _mm_bsrli_si128(_mm256_castsi256_si128(pixin_swap), 8);

				// Extend 8-bit values to 32-bit
				__m256i pix_0 = _mm256_cvtepu8_epi32(pixin_0);
				__m256i pix_1 = _mm256_cvtepu8_epi32(pixin_1);
				__m256i pix_2 = _mm256_cvtepu8_epi32(pixin_2);
				__m256i pix_3 = _mm256_cvtepu8_epi32(pixin_3);

				// Convert 32-bit values to floats
				__m256 pixf_0 = _mm256_cvtepi32_ps(pix_0);
				__m256 pixf_1 = _mm256_cvtepi32_ps(pix_1);
				__m256 pixf_2 = _mm256_cvtepi32_ps(pix_2);
				__m256 pixf_3 = _mm256_cvtepi32_ps(pix_3);

				_mm256_storeu_ps(out + i * 32, pixf_0);
				_mm256_storeu_ps(out + i * 32+8, pixf_1);
				_mm256_storeu_ps(out + i * 32+16, pixf_2);
				_mm256_storeu_ps(out + i * 32+24, pixf_3);
			}

			_mm256_zeroupper();

			for (size_t i = iters * 32; i < iters * 32 + itersRem; i++) {
				out[i] = static_cast<float>(in[i]);
			}
		#elif X86_SIMD_LEVEL >= LV_SSE41
			size_t iters = fileSize / 16;
			size_t itersRem = fileSize % 16;

			const __m128i shufmask0 = _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 2, 1, 0);
			const __m128i shufmask1 = _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 6, 5, 4);
			const __m128i shufmask2 = _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11, 10, 9, 8);
			const __m128i shufmask3 = _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 14, 13, 12);

			for (size_t i = 0; i < iters; i++) {
				__m128i pixin = _mm_loadu_si128(reinterpret_cast<const __m128i*>(in + i*16));

				__m128i pix0 = _mm_shuffle_epi8(pixin, shufmask0);
				__m128i pix1 = _mm_shuffle_epi8(pixin, shufmask1);
				__m128i pix2 = _mm_shuffle_epi8(pixin, shufmask2);
				__m128i pix3 = _mm_shuffle_epi8(pixin, shufmask3);

				pix0 = _mm_cvtepu8_epi32(pix0);
				pix1 = _mm_cvtepu8_epi32(pix1);
				pix2 = _mm_cvtepu8_epi32(pix2);
				pix3 = _mm_cvtepu8_epi32(pix3);

				__m128 pixf0 = _mm_cvtepi32_ps(pix0);
				__m128 pixf1 = _mm_cvtepi32_ps(pix1);
				__m128 pixf2 = _mm_cvtepi32_ps(pix2);
				__m128 pixf3 = _mm_cvtepi32_ps(pix3);

				_mm_storeu_ps(out + i*16, pixf0);
				_mm_storeu_ps(out + i*16+4, pixf1);
				_mm_storeu_ps(out + i*16+8, pixf2);
				_mm_storeu_ps(out + i*16+12, pixf3);
			}

			for (size_t i = iters * 16; i < iters * 16 + itersRem; i++) {
				out[i] = static_cast<float>(in[i]);
			}

		#else
			for (size_t i = 0; i < fileSize; i++) {
				out[i] = static_cast<float>(in[i]);
			}
		#endif
	}

	std::vector<uint8_t> Image::GetPixelsAsInt() {
		alignas(64) std::vector<uint8_t> pixelsOut;
		size_t dataLength = Image::GetTotalImageSize(width, height, format);
		pixelsOut.resize(dataLength);

		#if X86_SIMD_LEVEL >= LV_SSSE3
		// Works on 16 source floats at a time
		// Doing this allows full utilization of an XMM register for stores
		size_t iters = dataLength / 16;
		size_t itersRem = dataLength % 16;

		// Mask for moving least significant byte of 32-bit integers to start of register
		const __m128i shufmask = _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12, 8, 4, 0);

		alignas(16) std::array<uint8_t, 16> out_tmp;

		for (size_t i = 0; i < iters; i++) {
			// Load 16 floats into registers
			__m128 in0 = _mm_loadu_ps(pixels.data() + i*16);
			__m128 in1 = _mm_loadu_ps(pixels.data() + i*16 + 4);
			__m128 in2 = _mm_loadu_ps(pixels.data() + i*16 + 8);
			__m128 in3 = _mm_loadu_ps(pixels.data() + i*16 + 12);

			// Batch convert 4 floats into 32-bit integers
			__m128i converted0_32 = _mm_cvtps_epi32(in0);
			__m128i converted1_32 = _mm_cvtps_epi32(in1);
			__m128i converted2_32 = _mm_cvtps_epi32(in2);
			__m128i converted3_32 = _mm_cvtps_epi32(in3);

			// Perform shuffle that moves the least significant byte of 32-bit integers to the start of each register.
			// Because the color values are capped at 255 (maximum value representable by a single byte), it
			// effectively packs the set of 16 total values into 32-bit collections at the bottom of each register
			__m128i converted0 = _mm_shuffle_epi8(converted0_32, shufmask);
			__m128i converted1 = _mm_shuffle_epi8(converted1_32, shufmask);
			__m128i converted2 = _mm_shuffle_epi8(converted2_32, shufmask);
			__m128i converted3 = _mm_shuffle_epi8(converted3_32, shufmask);

			// Merge packed values:
			// Unpack first two values, creating XX10
			__m128i low0 = _mm_unpacklo_epi32(converted0, converted1);
			// Unpack remaining two values, creating 32XX
			__m128i low1 = _mm_unpacklo_epi32(converted2, converted3);

			// Merge two registers with data in their lower halves, forming 3210 in a single register
			__m128 output = _mm_shuffle_ps(reinterpret_cast<__m128i>(low0), reinterpret_cast<__m128i>(low1), 0b01000100);

			_mm_store_ps(reinterpret_cast<float*>(out_tmp.data()), output);

			std::memcpy(pixelsOut.data()+i*16, out_tmp.data(), 16);
		}

		for (size_t i = iters * 16; i < iters * 16 + itersRem; i++) {
			pixelsOut[i] = static_cast<uint8_t>(pixels[i]);
		}

		#else
		for (size_t i = 0; i < dataLength; i++) {
			pixelsOut[i] = static_cast<uint8_t>(pixels[i]);
		}

		#endif

		return pixelsOut;
	}

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
		}
		break;
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
		case F_BGR:
		{
			alignas(64) auto tmp = new float[in.width * in.height];
			std::uninitialized_fill(tmp, tmp + in.width * in.height, 255.0f);
			in.channels.push_back(tmp);
		}
		break;
		default:
			break;
		}

		in.format = F_BGRA;
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

	ImageSoA::ImageSoA(const Image& /*img*/) {
		throw wlib::module_not_built(errMsg);
	}
	ImageSoA::ImageSoA(const ImageSoA& /*img*/) {
		throw wlib::module_not_built(errMsg);
	}
	ImageSoA& ImageSoA::operator=(const ImageSoA& /*img*/) {
		throw wlib::module_not_built(errMsg);
	}
	ImageSoA::~ImageSoA() {
		throw wlib::module_not_built(errMsg);
	}
	Image ImageSoA::ConvertToImage() {
		throw wlib::module_not_built(errMsg);
	}
	ImageSoA& ConvertToGrayscale(ImageSoA& /*inImg*/, const bool /*preserveAlpha*/, const GrayscaleMethod /*method*/) {
		throw wlib::module_not_built(errMsg);
	}
	void ConvertUint8ToFloat(const uint8_t* /*in*/, float* /*out*/, const size_t /*fileSize*/) {
		throw wlib::module_not_built(errMsg);
	}
	void ConvertToRGB(ImageSoA& /*in*/) {
		throw wlib::module_not_built(errMsg);
	}
	void ConvertToBGR(ImageSoA& /*in*/) {
		throw wlib::module_not_built(errMsg);
	}
	void ConvertToRGBA(ImageSoA& /*in*/) {
		throw wlib::module_not_built(errMsg);
	}
	void ConvertToBGRA(ImageSoA& /*in*/) {
		throw wlib::module_not_built(errMsg);
	}
} // namespace wlib::image

#pragma clang diagnostic pop
#pragma GCC diagnostic pop

#endif
