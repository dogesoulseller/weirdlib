#include "../../include/weirdlib_image.hpp"
#include "../../include/cpu_detection.hpp"

namespace wlib::image
{
	#if X86_SIMD_LEVEL >= 9
	static void FV_AVX512_GRAY_IGNORE_ALPHA(ImageSoA& in) {
		const size_t SIMDIter = (in.width * in.height) / 16;
		const size_t SIMDIRem = (in.width * in.height) % 16;

		const __m512 subt = _mm512_set1_ps(255.0f);
		for (size_t i = 0; i < SIMDIter; i++) {
			__m512 pix = _mm512_loadu_ps(in.channels[0]+(i*16));
			__m512 result = _mm512_sub_ps(subt, pix);
			_mm512_storeu_ps(in.channels[0]+(i*16), result);
		}

		// Remainder
		for (size_t i = 0; i < SIMDIRem; i++) {
			in.channels[0][SIMDIter*16+i] = 255.0f - in.channels[0][SIMDIter*16+i];
		}
	}

	static void FV_AVX512_RGB_IGNORE_ALPHA(ImageSoA& in) {
		const size_t SIMDIter = (in.width * in.height) / 16;
		const size_t SIMDIRem = (in.width * in.height) % 16;

		const __m512 subt = _mm512_set1_ps(255.0f);
		for (size_t c = 0; c < 3; c++)  {
			for (size_t i = 0; i < SIMDIter; i++) {
				__m512 pix = _mm512_loadu_ps(in.channels[c]+(i*16));
				__m512 result = _mm512_sub_ps(subt, pix);
				_mm512_storeu_ps(in.channels[c]+(i*16), result);
			}

			// Remainder
			for (size_t i = 0; i < SIMDIRem; i++) {
				in.channels[c][SIMDIter*16+i] = 255.0f - in.channels[c][SIMDIter*16+i];
			}
		}
	}

	static void FV_AVX512_ALL_PROCESS_ALPHA(ImageSoA& in) {
		const size_t SIMDIter = (in.width * in.height) / 16;
		const size_t SIMDIRem = (in.width * in.height) % 16;

		const __m512 subt = _mm512_set1_ps(255.0f);
		for (auto& c : in.channels)  {
			for (size_t i = 0; i < SIMDIter; i++) {
				__m512 pix = _mm512_loadu_ps(c+(i*16));
				__m512 result = _mm512_sub_ps(subt, pix);
				_mm512_storeu_ps(c+(i*16), result);
			}

			// Remainder
			for (size_t i = 0; i < SIMDIRem; i++) {
				c[SIMDIter*16+i] = 255.0f - c[SIMDIter*16+i];
			}
		}

	}
	#elif X86_SIMD_LEVEL >= 7
	static void FV_AVX_GRAY_IGNORE_ALPHA(ImageSoA& in) {
		const size_t SIMDIter = (in.width * in.height) / 8;
		const size_t SIMDIRem = (in.width * in.height) % 8;

		const __m256 subt = _mm256_set1_ps(255.0f);
		for (size_t i = 0; i < SIMDIter; i++) {
			__m256 pix = _mm256_loadu_ps(in.channels[0]+(i*8));
			__m256 result = _mm256_sub_ps(subt, pix);
			_mm256_storeu_ps(in.channels[0]+(i*8), result);
		}

		// Remainder
		for (size_t i = 0; i < SIMDIRem; i++) {
			in.channels[0][SIMDIter*8+i] = 255.0f - in.channels[0][SIMDIter*8+i];
		}
	}

	static void FV_AVX_RGB_IGNORE_ALPHA(ImageSoA& in) {
		const size_t SIMDIter = (in.width * in.height) / 8;
		const size_t SIMDIRem = (in.width * in.height) % 8;

		const __m256 subt = _mm256_set1_ps(255.0f);
		for (size_t c = 0; c < 3; c++)  {
			for (size_t i = 0; i < SIMDIter; i++) {
				__m256 pix = _mm256_loadu_ps(in.channels[c]+(i*8));
				__m256 result = _mm256_sub_ps(subt, pix);
				_mm256_storeu_ps(in.channels[c]+(i*8), result);
			}

			// Remainder
			for (size_t i = 0; i < SIMDIRem; i++) {
				in.channels[c][SIMDIter*8+i] = 255.0f - in.channels[c][SIMDIter*8+i];
			}
		}
	}

	static void FV_AVX_ALL_PROCESS_ALPHA(ImageSoA& in) {
		const size_t SIMDIter = (in.width * in.height) / 8;
		const size_t SIMDIRem = (in.width * in.height) % 8;

		const __m256 subt = _mm256_set1_ps(255.0f);
		for (auto& c : in.channels)  {
			for (size_t i = 0; i < SIMDIter; i++) {
				__m256 pix = _mm256_loadu_ps(c+(i*8));
				__m256 result = _mm256_sub_ps(subt, pix);
				_mm256_storeu_ps(c+(i*8), result);
			}

			// Remainder
			for (size_t i = 0; i < SIMDIRem; i++) {
				c[SIMDIter*8+i] = 255.0f - c[SIMDIter*8+i];
			}
		}

	}
	#elif X86_SIMD_LEVEL >= 1
	static void FV_SSE_GRAY_IGNORE_ALPHA(ImageSoA& in) {
		const size_t SIMDIter = (in.width * in.height) / 4;
		const size_t SIMDIRem = (in.width * in.height) % 4;

		const __m128 subt = _mm_set1_ps(255.0f);
		for (size_t i = 0; i < SIMDIter; i++) {
			__m128 pix = _mm_loadu_ps(in.channels[0]+(i*4));
			__m128 result = _mm_sub_ps(subt, pix);
			_mm_storeu_ps(in.channels[0]+(i*4), result);
		}

		// Remainder
		for (size_t i = 0; i < SIMDIRem; i++) {
			in.channels[0][SIMDIter*4+i] = 255.0f - in.channels[0][SIMDIter*4+i];
		}
	}

	static void FV_SSE_RGB_IGNORE_ALPHA(ImageSoA& in) {
		const size_t SIMDIter = (in.width * in.height) / 4;
		const size_t SIMDIRem = (in.width * in.height) % 4;

		const __m128 subt = _mm_set1_ps(255.0f);
		for (size_t c = 0; c < 3; c++)  {
			for (size_t i = 0; i < SIMDIter; i++) {
				__m128 pix = _mm_loadu_ps(in.channels[c]+(i*4));
				__m128 result = _mm_sub_ps(subt, pix);
				_mm_storeu_ps(in.channels[c]+(i*4), result);
			}

			// Remainder
			for (size_t i = 0; i < SIMDIRem; i++) {
				in.channels[c][SIMDIter*4+i] = 255.0f - in.channels[c][SIMDIter*4+i];
			}
		}
	}

	static void FV_SSE_ALL_PROCESS_ALPHA(ImageSoA& in) {
		const size_t SIMDIter = (in.width * in.height) / 4;
		const size_t SIMDIRem = (in.width * in.height) % 4;

		const __m128 subt = _mm_set1_ps(255.0f);
		for (auto& c : in.channels)  {
			for (size_t i = 0; i < SIMDIter; i++) {
				__m128 pix = _mm_loadu_ps(c+(i*4));
				__m128 result = _mm_sub_ps(subt, pix);
				_mm_storeu_ps(c+(i*4), result);
			}

			// Remainder
			for (size_t i = 0; i < SIMDIRem; i++) {
				c[SIMDIter*4+i] = 255.0f - c[SIMDIter*4+i];
			}
		}

	}
	#else
	static void FV_GENERIC_GRAY_IGNORE_ALPHA(ImageSoA& in) {
		for (size_t i = 0; i < in.width * in.height; i++) {
			in.channels[0][i] = 255.0f - in.channels[0][i];
		}
	}

	static void FV_GENERIC_RGB_IGNORE_ALPHA(ImageSoA& in) {
		for (size_t c = 0; c < 3; c++) {
			for (size_t i = 0; i < in.width * in.height; i++) {
				in.channels[c][i] = 255.0f - in.channels[c][i];
			}
		}
	}

	static void FV_GENERIC_ALL_PROCESS_ALPHA(ImageSoA& in) {
		for (auto& c : in.channels) {
			for (size_t i = 0 ; i < in.width * in.height; i++) {
				c[i] = 255.0f - c[i];
			}
		}
	}
	#endif

	void NegateValues(ImageSoA& in, bool withAlpha) {
		if (!withAlpha) {
			if (in.channels.size() == 2) {	// GA
				#if X86_SIMD_LEVEL >= 9
				FV_AVX512_GRAY_IGNORE_ALPHA(in);
				#elif X86_SIMD_LEVEL >= 7
				FV_AVX_GRAY_IGNORE_ALPHA(in);
				#elif X86_SIMD_LEVEL >= 1
				FV_SSE_GRAY_IGNORE_ALPHA(in);
				#else
				FV_GENERIC_GRAY_IGNORE_ALPHA(in);
				#endif
			} else if (in.channels.size() == 4) {	// xxxA
				#if X86_SIMD_LEVEL >= 9
				FV_AVX512_RGB_IGNORE_ALPHA(in);
				#elif X86_SIMD_LEVEL >= 7
				FV_AVX_RGB_IGNORE_ALPHA(in);
				#elif X86_SIMD_LEVEL >= 1
				FV_SSE_RGB_IGNORE_ALPHA(in);
				#else
				FV_GENERIC_RGB_IGNORE_ALPHA(in);
				#endif
			} else {
				#if X86_SIMD_LEVEL >= 9
				FV_AVX512_ALL_PROCESS_ALPHA(in);
				#elif X86_SIMD_LEVEL >= 7
				FV_AVX_ALL_PROCESS_ALPHA(in);
				#elif X86_SIMD_LEVEL >= 1
				FV_SSE_ALL_PROCESS_ALPHA(in);
				#else
				FV_GENERIC_ALL_PROCESS_ALPHA(in);
				#endif
			}
		} else {
			#if X86_SIMD_LEVEL >= 9
			FV_AVX512_ALL_PROCESS_ALPHA(in);
			#elif X86_SIMD_LEVEL >= 7
			FV_AVX_ALL_PROCESS_ALPHA(in);
			#elif X86_SIMD_LEVEL >= 1
			FV_SSE_ALL_PROCESS_ALPHA(in);
			#else
			FV_GENERIC_ALL_PROCESS_ALPHA(in);
			#endif
		}
	}

} // namespace wlib::image
