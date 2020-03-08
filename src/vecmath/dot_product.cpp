#ifdef WEIRDLIB_ENABLE_VECTOR_MATH
#include "../../include/weirdlib_vecmath.hpp"
#include "../../include/weirdlib_simdhelper.hpp"
#include "../../include/cpu_detection.hpp"
#include <array>

namespace wlib::vecmath
{
	float DotProduct(const Vector2<float>& lhs, const Vector2<float>& rhs) {
		#if X86_SIMD_LEVEL >= LV_SSE
			auto lhs_vec = _mm_loadu_ps(&lhs.x);
			auto rhs_vec = _mm_loadu_ps(&rhs.x);

			auto prods = _mm_mul_ps(lhs_vec, rhs_vec);
			auto temp = _mm_shuffle_ps(prods, prods, _MM_SHUFFLE(0, 0, 0, 1));
			auto result = _mm_add_ss(temp, prods);

			alignas(16) std::array<float, 4> outvec;
			_mm_store_ps(outvec.data(), result);
			return outvec[0];
		#else
			return (lhs.x * rhs.x) + (lhs.y * rhs.y);
		#endif
	}

	double DotProduct(const Vector2<double>& lhs, const Vector2<double>& rhs) {
		#if X86_SIMD_LEVEL >= LV_SSE2
			auto lhs_vec = _mm_loadu_pd(&lhs.x);
			auto rhs_vec = _mm_loadu_pd(&rhs.x);

			auto prods = _mm_mul_pd(lhs_vec, rhs_vec);
			auto temp = _mm_shuffle_pd(prods, prods, _MM_SHUFFLE2(0, 1));
			auto result = _mm_add_sd(temp, prods);

			alignas(16) std::array<double, 2> outvec;
			_mm_store_pd(outvec.data(), result);
			return outvec[0];
		#else
			return (lhs.x * rhs.x) + (lhs.y * rhs.y);
		#endif
	}

	float DotProduct(const Vector3<float>& lhs, const Vector3<float>& rhs) {
		#if X86_SIMD_LEVEL >= LV_SSE
			auto lhs_vec = _mm_loadu_ps(&lhs.x);
			auto rhs_vec = _mm_loadu_ps(&rhs.x);

			auto prods = _mm_mul_ps(lhs_vec, rhs_vec);
			auto el1 = _mm_shuffle_ps(prods, prods, _MM_SHUFFLE(0, 0, 0, 1));
			auto el2 = _mm_shuffle_ps(prods, prods, _MM_SHUFFLE(0, 0, 0, 2));

			auto result = _mm_add_ss(_mm_add_ss(prods, el1), el2);

			alignas(16) std::array<float, 4> outvec;
			_mm_store_ps(outvec.data(), result);
			return outvec[0];
		#else
			return (lhs.x * rhs.x) + (lhs.y * rhs.y) + (lhs.z * rhs.z);
		#endif
	}

	double DotProduct(const Vector3<double>& lhs, const Vector3<double>& rhs) {
		#if X86_SIMD_LEVEL >= LV_AVX
			auto lhs_vec = _mm256_loadu_pd(&lhs.x);
			auto rhs_vec = _mm256_loadu_pd(&rhs.x);

			auto prod = _mm256_mul_pd(lhs_vec, rhs_vec);

			// We only use the result of positions 0+1
			auto temp_hadd = _mm256_hadd_pd(prod, prod);

			// Product is reversed lane-wise to get position 2 into position 0
			auto prod_reversed = wlib::simd::reverseLanes(prod);

			// Add the now aligned values
			auto result = _mm256_add_pd(prod_reversed, temp_hadd);

			alignas(32) std::array<double, 4> outvec;
			_mm256_store_pd(outvec.data(), result);
			return outvec[0];
		#elif X86_SIMD_LEVEL >= LV_SSE2
			auto lhs0 = _mm_loadu_pd(&lhs.x);
			auto rhs0 = _mm_loadu_pd(&rhs.x);

			auto lhs1 = _mm_loadu_pd(&lhs.z);
			auto rhs1 = _mm_loadu_pd(&rhs.z);

			auto prods0 = _mm_mul_pd(lhs0, rhs0);
			auto prods1 = _mm_mul_sd(lhs1, rhs1);

			auto temp = _mm_shuffle_pd(prods0, prods0, _MM_SHUFFLE2(0, 1));

			auto result = _mm_add_sd(_mm_add_sd(prods0, prods1), temp);

			alignas(16) std::array<double, 2> outvec;
			_mm_store_pd(outvec.data(), result);
			return outvec[0];
		#else
			return (lhs.x * rhs.x) + (lhs.y * rhs.y) + (lhs.z * rhs.z);
		#endif
	}

	float DotProduct(const Vector4<float>& lhs, const Vector4<float>& rhs) {
		#if X86_SIMD_LEVEL >= LV_SSE41
			auto lhs_vec = _mm_loadu_ps(&lhs.x);
			auto rhs_vec = _mm_loadu_ps(&rhs.x);

			alignas(16)	std::array<float, 4> outvec;
			_mm_store_ps(outvec.data(), _mm_dp_ps(lhs_vec, rhs_vec, 0xFF));
			return outvec[0];
		#elif X86_SIMD_LEVEL >= LV_SSE
			auto lhs_vec = _mm_loadu_ps(&lhs.x);
			auto rhs_vec = _mm_loadu_ps(&rhs.x);

			auto prods = _mm_mul_ps(lhs_vec, rhs_vec);

			auto el1 = _mm_shuffle_ps(prods, prods, _MM_SHUFFLE(0, 0, 3, 2));

			auto sumtemp = _mm_add_ps(prods, el1);

			auto el2 = _mm_shuffle_ps(sumtemp, sumtemp, _MM_SHUFFLE(0, 0, 0, 1));

			auto result = _mm_add_ss(el2, sumtemp);

			alignas(16)	std::array<float, 4> outvec;
			_mm_store_ps(outvec.data(), result);
			return outvec[0];
		#else
			return (lhs.x * rhs.x) + (lhs.y * rhs.y) + (lhs.z * rhs.z) + (lhs.w * rhs.w);
		#endif
	}

	double DotProduct(const Vector4<double>& lhs, const Vector4<double>& rhs) {
		#if X86_SIMD_LEVEL >= LV_AVX
			auto lhs_vec = _mm256_loadu_pd(&lhs.x);
			auto rhs_vec = _mm256_loadu_pd(&rhs.x);

			auto prod = _mm256_mul_pd(lhs_vec, rhs_vec);

			// Necessary results are in position 0 and 2
			auto hadd_result = _mm256_hadd_pd(prod, prod);

			// Reverse to get register with previous position 2 now in position 0
			auto hadd_result_reversed = wlib::simd::reverseLanes(hadd_result);

			// Add up results
			auto temp_fin = _mm256_add_pd(hadd_result_reversed, hadd_result);

			alignas(32) std::array<double, 4> outvec;
			_mm256_store_pd(outvec.data(), temp_fin);
			return outvec[0];
		#elif X86_SIMD_LEVEL >= LV_SSE2
			auto lhs0 = _mm_loadu_pd(&lhs.x);
			auto rhs0 = _mm_loadu_pd(&rhs.x);

			auto lhs1 = _mm_loadu_pd(&lhs.z);
			auto rhs1 = _mm_loadu_pd(&rhs.z);

			auto res0 = _mm_mul_pd(lhs0, rhs0);
			auto res1 = _mm_mul_pd(lhs1, rhs1);
			auto resCombined = _mm_add_pd(res0, res1);

			auto temp = _mm_shuffle_pd(resCombined, resCombined, _MM_SHUFFLE2(0, 1));

			auto result = _mm_add_sd(temp, resCombined);

			alignas(16) std::array<double, 2> outvec;
			_mm_store_pd(outvec.data(), result);
			return outvec[0];
		#else
			return (lhs.x * rhs.x) + (lhs.y * rhs.y) + (lhs.z * rhs.z) + (lhs.w * rhs.w);
		#endif
	}

} // namespace wlib::vecmath
#endif
