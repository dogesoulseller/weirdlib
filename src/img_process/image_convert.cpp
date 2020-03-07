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
	ImageSoA& ConvertToGrayscale(ImageSoA& inImg, const bool preserveAlpha, const GrayscaleMethod method) {
		// Converting to grayscale from grayscale...
		if (inImg.GetFormat() == F_GrayAlpha || inImg.GetFormat() == F_Grayscale) {
			return inImg;
		}

		size_t channelRedN;
		size_t channelGreenN = 1;
		size_t channelBlueN;

		if (inImg.GetFormat() == F_RGB || inImg.GetFormat() == F_RGBA) {
			channelRedN = 0;
			channelBlueN = 2;
		} else {
			channelRedN = 2;
			channelBlueN = 0;
		}

		alignas(32) auto outGr = new float[inImg.GetWidth() * inImg.GetHeight()];

		auto ProcessLuminosity = [&](const float rWeight, const float gWeight, const float bWeight) {
			// Processing with SIMD
			#if X86_SIMD_LEVEL >= LV_AVX
				const auto redMul = _mm256_set1_ps(rWeight);
				const auto greenMul = _mm256_set1_ps(gWeight);
				const auto blueMul = _mm256_set1_ps(bWeight);
				const auto maxMask = _mm256_set1_ps(255.0f);

				const auto iter_AVX = (inImg.GetWidth() * inImg.GetHeight()) / 8;
				const auto iterRem_AVX = (inImg.GetWidth() * inImg.GetHeight()) % 8;

				for (size_t i = 0; i < iter_AVX; i++) {
					const auto rChan = _mm256_loadu_ps(inImg.AccessChannels()[channelRedN]+i*8);
					const auto gChan = _mm256_loadu_ps(inImg.AccessChannels()[channelGreenN]+i*8);
					const auto bChan = _mm256_loadu_ps(inImg.AccessChannels()[channelBlueN]+i*8);

					#ifdef X86_SIMD_FMA
						const auto resultGr = _mm256_min_ps(_mm256_fmadd_ps(rChan, redMul, _mm256_fmadd_ps(gChan, greenMul, _mm256_mul_ps(bChan, blueMul))), maxMask);
					#else
						const auto resultGr = _mm256_min_ps(
							_mm256_add_ps(_mm256_mul_ps(rChan, redMul), _mm256_add_ps(_mm256_mul_ps(gChan, greenMul), _mm256_mul_ps(bChan, blueMul))), maxMask);
					#endif
						_mm256_storeu_ps(outGr+8*i, resultGr);
				}

				for (size_t i = iter_AVX * 8; i < iterRem_AVX + iter_AVX*8; i++) {
					if (const auto result = std::fma(inImg.AccessChannels()[channelRedN][i], rWeight, std::fma(inImg.AccessChannels()[channelGreenN][i], gWeight, inImg.AccessChannels()[channelBlueN][i] * bWeight)); result > 255.0f) {
						outGr[i] = 255.0f;
					} else {
						outGr[i] = result;
					}
				}

				_mm256_zeroupper();
			#elif X86_SIMD_LEVEL >= LV_SSE
				const auto redMul = _mm_set1_ps(rWeight);
				const auto greenMul = _mm_set1_ps(gWeight);
				const auto blueMul = _mm_set1_ps(bWeight);
				const auto maxMask = _mm_set1_ps(255.0f);

				const auto iter_SSE = (inImg.GetWidth() * inImg.GetHeight()) / 4;
				const auto iterRem_SSE = (inImg.GetWidth() * inImg.GetHeight()) % 4;

				for (size_t i = 0; i < iter_SSE; i++) {
					const auto rChan = _mm_loadu_ps(inImg.AccessChannels()[channelRedN]+i*4);
					const auto gChan = _mm_loadu_ps(inImg.AccessChannels()[channelGreenN]+i*4);
					const auto bChan = _mm_loadu_ps(inImg.AccessChannels()[channelBlueN]+i*4);

					#ifdef X86_SIMD_FMA	// Technically shouldn't be possible
						const auto resultGr = _mm_min_ps(_mm_fmadd_ps(rChan, redMul, _mm256_fmadd_ps(gChan, greenMul, _mm_mul_ps(bChan, blueMul))), maxMask);
					#else
						const auto resultGr = _mm_min_ps(
							_mm_add_ps(_mm_mul_ps(rChan, redMul), _mm_add_ps(_mm_mul_ps(gChan, greenMul), _mm_mul_ps(bChan, blueMul))), maxMask);
					#endif
						_mm_storeu_ps(outGr+4*i, resultGr);
				}

				for (size_t i = iter_SSE * 4; i < iterRem_SSE + iter_SSE * 4; i++) {
					if (const auto result = std::fma(inImg.AccessChannels()[channelRedN][i], rWeight, std::fma(inImg.AccessChannels()[channelGreenN][i], gWeight, inImg.AccessChannels()[channelBlueN][i] * bWeight)); result > 255.0f) {
						outGr[i] = 255.0f;
					} else {
						outGr[i] = result;
					}
				}
			#else
				for (size_t i = 0; i < inImg.GetWidth() * inImg.GetHeight(); i++) {
					if (const auto result = std::fma(inImg.AccessChannels()[channelRedN][i], rWeight, std::fma(inImg.AccessChannels()[channelGreenN][i], gWeight, inImg.AccessChannels()[channelBlueN][i] * bWeight)); result > 255.0f) {
						outGr[i] = 255.0f;
					} else {
						outGr[i] = result;
					}
				}
			#endif
		};

		switch (method)
		{
		  case GrayscaleMethod::Luminosity: {
			ProcessLuminosity(0.2126f, 0.7152f, 0.0722f);
			break;
		  }
		  case GrayscaleMethod::LuminosityBT601: {
			ProcessLuminosity(0.299f, 0.587f, 0.114f);
			break;
		  }
		  case GrayscaleMethod::LuminosityBT2100: {
			ProcessLuminosity(0.2627f, 0.6780f, 0.0593f);
			break;
		  }
		  case GrayscaleMethod::Lightness: {
			// Processing with SIMD
			#if X86_SIMD_LEVEL >= LV_AVX
				const auto maxMask = _mm256_set1_ps(255.0f);
				const auto divMask = _mm256_set1_ps(0.5f);

				const auto iter_AVX = (inImg.GetWidth() * inImg.GetHeight()) / 8;
				const auto iterRem_AVX = (inImg.GetWidth() * inImg.GetHeight()) % 8;

				for (size_t i = 0; i < iter_AVX; i++) {
					const auto rChan = _mm256_loadu_ps(inImg.AccessChannels()[channelRedN]+i*8);
					const auto gChan = _mm256_loadu_ps(inImg.AccessChannels()[channelGreenN]+i*8);
					const auto bChan = _mm256_loadu_ps(inImg.AccessChannels()[channelBlueN]+i*8);

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
					const auto maxValue = std::max(std::max(inImg.AccessChannels()[channelRedN][i], inImg.AccessChannels()[channelGreenN][i]), inImg.AccessChannels()[channelBlueN][i]);
					const auto minValue = std::min(std::min(inImg.AccessChannels()[channelRedN][i], inImg.AccessChannels()[channelGreenN][i]), inImg.AccessChannels()[channelBlueN][i]);
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

				const auto iter_SSE = (inImg.GetWidth() * inImg.GetHeight()) / 4;
				const auto iterRem_SSE = (inImg.GetWidth() * inImg.GetHeight()) % 4;

				for (size_t i = 0; i < iter_SSE; i++) {
					const auto rChan = _mm_loadu_ps(inImg.AccessChannels()[channelRedN]+i*4);
					const auto gChan = _mm_loadu_ps(inImg.AccessChannels()[channelGreenN]+i*4);
					const auto bChan = _mm_loadu_ps(inImg.AccessChannels()[channelBlueN]+i*4);

					const auto maxValues = _mm_max_ps(rChan, _mm_max_ps(gChan, bChan));
					const auto minValues = _mm_min_ps(rChan, _mm_min_ps(gChan, bChan));

					const auto resultGr = _mm_min_ps(_mm_mul_ps(_mm_add_ps(maxValues, minValues), divMask), maxMask);

					_mm_storeu_ps(outGr+4*i, resultGr);
				}

				for (size_t i = iter_SSE * 4; i < iterRem_SSE + iter_SSE*4; i++) {
					const auto maxValue = std::max(std::max(inImg.AccessChannels()[channelRedN][i], inImg.AccessChannels()[channelGreenN][i]), inImg.AccessChannels()[channelBlueN][i]);
					const auto minValue = std::min(std::min(inImg.AccessChannels()[channelRedN][i], inImg.AccessChannels()[channelGreenN][i]), inImg.AccessChannels()[channelBlueN][i]);
					const auto result = (maxValue + minValue) * 0.5f;

					if (result > 255.0f) {
						outGr[i] = 255.0f;
					} else {
						outGr[i] = result;
					}
				}

			#else
				for (size_t i = 0; i < inImg.GetWidth() * inImg.GetHeight(); i++) {
					const auto maxValue = std::max(std::max(inImg.AccessChannels()[channelRedN][i], inImg.AccessChannels()[channelGreenN][i]), inImg.AccessChannels()[channelBlueN][i]);
					const auto minValue = std::min(std::min(inImg.AccessChannels()[channelRedN][i], inImg.AccessChannels()[channelGreenN][i]), inImg.AccessChannels()[channelBlueN][i]);
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
		  case GrayscaleMethod::Average: {
			// Processing with SIMD
			#if X86_SIMD_LEVEL >= LV_AVX
				const auto maxMask = _mm256_set1_ps(255.0f);
				const auto divMask = _mm256_set1_ps(0.3333333333f);

				const auto iter_AVX = (inImg.GetWidth() * inImg.GetHeight()) / 8;
				const auto iterRem_AVX = (inImg.GetWidth() * inImg.GetHeight()) % 8;

				for (size_t i = 0; i < iter_AVX; i++) {
					const auto rChan = _mm256_loadu_ps(inImg.AccessChannels()[channelRedN]+i*8);
					const auto gChan = _mm256_loadu_ps(inImg.AccessChannels()[channelGreenN]+i*8);
					const auto bChan = _mm256_loadu_ps(inImg.AccessChannels()[channelBlueN]+i*8);

					#ifdef X86_SIMD_FMA
					const auto resultGr = _mm256_min_ps(_mm256_fmadd_ps(_mm256_add_ps(rChan, gChan), bChan, divMask), maxMask);
					#else
					const auto resultGr = _mm256_min_ps(_mm256_mul_ps(_mm256_add_ps(rChan, _mm256_add_ps(gChan, bChan)), divMask), maxMask);
					#endif

					_mm256_storeu_ps(outGr+8*i, resultGr);
				}

				for (size_t i = iter_AVX * 8; i < iterRem_AVX + iter_AVX*8; i++) {
					const auto result = (inImg.AccessChannels()[channelRedN][i] + inImg.AccessChannels()[channelGreenN][i] + inImg.AccessChannels()[channelBlueN][i]) * 0.3333333333f;

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

				const auto iter_AVX = (inImg.GetWidth() * inImg.GetHeight()) / 4;
				const auto iterRem_AVX = (inImg.GetWidth() * inImg.GetHeight()) % 4;

				for (size_t i = 0; i < iter_AVX; i++) {
					const auto rChan = _mm_loadu_ps(inImg.AccessChannels()[channelRedN]+i*4);
					const auto gChan = _mm_loadu_ps(inImg.AccessChannels()[channelGreenN]+i*4);
					const auto bChan = _mm_loadu_ps(inImg.AccessChannels()[channelBlueN]+i*4);

					const auto resultGr = _mm_min_ps(_mm_mul_ps(_mm_add_ps(rChan, _mm_add_ps(gChan, bChan)), divMask), maxMask);

					_mm_storeu_ps(outGr+4*i, resultGr);
				}

				for (size_t i = iter_AVX * 4; i < iterRem_AVX + iter_AVX*4; i++) {
					const auto result = (inImg.AccessChannels()[channelRedN][i] + inImg.AccessChannels()[channelGreenN][i] + inImg.AccessChannels()[channelBlueN][i]) * 0.3333333333f;

					if (result > 255.0f) {
						outGr[i] = 255.0f;
					} else {
						outGr[i] = result;
					}
				}

			#else
				for (size_t i = 0; i < inImg.GetWidth() * inImg.GetHeight(); i++) {
					const auto result = (inImg.AccessChannels()[channelRedN][i] + inImg.AccessChannels()[channelGreenN][i] + inImg.AccessChannels()[channelBlueN][i]) * 0.3333333333f;

					if (result > 255.0f) {
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
		if (inImg.GetFormat() == F_RGB || inImg.GetFormat() == F_BGR || !preserveAlpha) {
			inImg.SetFormat(F_Grayscale);

			for (auto& ptr : inImg.AccessChannels()) {
				delete[] ptr;
			}

			inImg.AccessChannels().resize(1);
			inImg.AccessChannels().shrink_to_fit();
			inImg.AccessChannels()[0] = outGr;
		} else if (inImg.GetFormat() == F_RGBA || inImg.GetFormat() == F_BGRA) {
			inImg.SetFormat(F_GrayAlpha);

			for (size_t i = 0; i < inImg.AccessChannels().size()-1; i++) {
				delete[] inImg.AccessChannels()[i];
			}

			auto imgAlpha = inImg.AccessChannels()[3];

			inImg.AccessChannels().resize(2);
			inImg.AccessChannels().shrink_to_fit();
			inImg.AccessChannels()[0] = outGr;
			inImg.AccessChannels()[1] = imgAlpha;
		}

		return inImg;
	}

	void ConvertUint16ToFloat(const uint16_t* in, float* out, const size_t fileSize) {
		#if X86_SIMD_LEVEL >= LV_AVX2
			size_t iters = fileSize / 8;
			size_t itersRem = fileSize % 8;

			for (size_t i = 0; i < iters; i++) {
				__m128i pixin = _mm_loadu_si128(reinterpret_cast<const __m128i*>(in+i*8));
				__m256i pix0 = _mm256_cvtepu16_epi32(pixin);
				__m256 cvt = _mm256_cvtepi32_ps(pix0);

				_mm256_store_ps(out+i*8, cvt);
			}

			for (size_t i = iters * 8; i < iters * 8 + itersRem; i++) {
				out[i] = static_cast<float>(in[i]);
			}
		#elif X86_SIMD_LEVEL >= LV_SSE41
			size_t iters = fileSize / 4;
			size_t itersRem = fileSize % 4;

			for (size_t i = 0; i < iters; i++) {
				__m128i pixin = _mm_loadu_si128(reinterpret_cast<const __m128i*>(in+i*4));
				__m128i pix0 = _mm_cvtepu16_epi32(pixin);
				__m128 cvt = _mm_cvtepi32_ps(pix0);

				_mm_store_ps(out+i*4, cvt);
			}

			for (size_t i = iters * 4; i < iters * 4 + itersRem; i++) {
				out[i] = static_cast<float>(in[i]);
			}
		#else
			for (size_t i = 0; i < fileSize; i++) {
				out[i] = static_cast<float>(in[i]);
			}
		#endif
	}

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
} // namespace wlib::image

#endif