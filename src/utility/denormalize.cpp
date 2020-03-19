#include "../../include/weirdlib_utility.hpp"
#include "../../include/cpu_detection.hpp"

#include <algorithm>

namespace wlib::util
{
	void DenormalizeData(float* inout, size_t count, float maxval) {
		#if X86_SIMD_LEVEL >= LV_AVX
			auto muls = _mm256_set1_ps(maxval);

			size_t iters = count / 16;
			size_t itersRem = count % 16;

			// Unrolled 2 times
			for (size_t i = 0; i < iters; i++) {
				auto invec0 = _mm256_loadu_ps(inout + 16 * i);
				auto invec1 = _mm256_loadu_ps(inout + 16 * i + 8);

				_mm256_storeu_ps(inout + 16 * i, _mm256_mul_ps(invec0, muls));
				_mm256_storeu_ps(inout + 16 * i + 8, _mm256_mul_ps(invec1, muls));
			}

			if (itersRem != 0) {
				std::transform(inout+iters*16, inout+count, inout+iters*16, [&maxval] (auto val) {return maxval * val;});
			} else {
				return;
			}
		#elif X86_SIMD_LEVEL >= LV_SSE
			auto muls = _mm_set1_ps(maxval);

			size_t iters = count / 16;
			size_t itersRem = count % 16;

			// Unrolled 4 times
			for (size_t i = 0; i < iters; i++) {
				auto invec0 = _mm_loadu_ps(inout + 16 * i);
				auto invec1 = _mm_loadu_ps(inout + 16 * i + 4);
				auto invec2 = _mm_loadu_ps(inout + 16 * i + 8);
				auto invec3 = _mm_loadu_ps(inout + 16 * i + 12);

				_mm_storeu_ps(inout + 16 * i, _mm_mul_ps(invec0, muls));
				_mm_storeu_ps(inout + 16 * i + 4, _mm_mul_ps(invec1, muls));
				_mm_storeu_ps(inout + 16 * i + 8, _mm_mul_ps(invec2, muls));
				_mm_storeu_ps(inout + 16 * i + 12, _mm_mul_ps(invec3, muls));
			}

			if (itersRem != 0) {
				std::transform(inout+iters*16, inout+count, inout+iters*16, [&maxval] (auto val) {return maxval * val;});
			} else {
				return;
			}
		#else
			std::transform(inout, inout+count, inout, [&maxval] (auto val) {return maxval * val;});
		#endif
	}

	void DenormalizeData(double* inout, size_t count, double maxval) {
		#if X86_SIMD_LEVEL >= LV_AVX
			auto muls = _mm256_set1_pd(maxval);

			size_t iters = count / 8;
			size_t itersRem = count % 8;

			// Unrolled 2 times
			for (size_t i = 0; i < iters; i++) {
				auto invec0 = _mm256_loadu_pd(inout + 8 * i);
				auto invec1 = _mm256_loadu_pd(inout + 8 * i + 4);

				_mm256_storeu_pd(inout + 8 * i, _mm256_mul_pd(invec0, muls));
				_mm256_storeu_pd(inout + 8 * i + 4, _mm256_mul_pd(invec1, muls));
			}

			if (itersRem != 0) {
				std::transform(inout+iters*8, inout+count, inout+iters*8, [&maxval] (auto val) {return maxval * val;});
			} else {
				return;
			}
		#elif X86_SIMD_LEVEL >= LV_SSE2
			auto muls = _mm_set1_pd(maxval);

			size_t iters = count / 8;
			size_t itersRem = count % 8;

			// Unrolled 4 times
			for (size_t i = 0; i < iters; i++) {
				auto invec0 = _mm_loadu_pd(inout + 8 * i);
				auto invec1 = _mm_loadu_pd(inout + 8 * i + 2);
				auto invec2 = _mm_loadu_pd(inout + 8 * i + 4);
				auto invec3 = _mm_loadu_pd(inout + 8 * i + 6);

				_mm_storeu_pd(inout + 8 * i, _mm_mul_pd(invec0, muls));
				_mm_storeu_pd(inout + 8 * i + 2, _mm_mul_pd(invec1, muls));
				_mm_storeu_pd(inout + 8 * i + 4, _mm_mul_pd(invec2, muls));
				_mm_storeu_pd(inout + 8 * i + 6, _mm_mul_pd(invec3, muls));
			}

			if (itersRem != 0) {
				std::transform(inout+iters*8, inout+count, inout+iters*8, [&maxval] (auto val) {return maxval * val;});
			} else {
				return;
			}
		#else
			std::transform(inout, inout+count, inout, [&maxval] (auto val) {return maxval * val;});
		#endif
	}

	void DenormalizeData(long double* inout, size_t count, long double maxval) {
		std::transform(inout, inout+count, inout, [&maxval] (auto val) {return maxval * val;});
	}

} // namespace wlib::util
