#include "../../include/weirdlib_vecmath.hpp"
#include "../../include/weirdlib_simdhelper.hpp"
#include "../../include/cpu_detection.hpp"
#include <array>
#include <cmath>

namespace wlib::vecmath
{
	static __m128 getPreAdd(const __m128 lhs, const __m128 rhs) {
		auto res_sub = _mm_sub_ps(lhs, rhs);
		return _mm_mul_ps(res_sub, res_sub);
	}

	float Distance(const Vector2<float>& lhs, const Vector2<float>& rhs) {
		#if X86_SIMD_LEVEL >= LV_SSE2
			auto lhs_vec = _mm_loadu_ps(&lhs.x);
			auto rhs_vec = _mm_loadu_ps(&rhs.x);
			auto pre_addition = getPreAdd(lhs_vec, rhs_vec);

			auto comp1 = _mm_castsi128_ps(_mm_bsrli_si128(_mm_castps_si128(pre_addition), 4));

			auto sum = _mm_add_ps(pre_addition, comp1);
			auto result = _mm_sqrt_ps(sum);

			alignas(16) std::array<float, 4> outvec;
			_mm_store_ps(outvec.data(), result);

			return outvec[0];
		#else
			auto xs = std::pow(lhs.x - rhs.x, 2);
			auto ys = std::pow(lhs.y - rhs.y, 2);

			return std::sqrt(xs + ys);
		#endif
	}

	double Distance(const Vector2<double>& lhs, const Vector2<double>& rhs) {
		#if X86_SIMD_LEVEL >= LV_SSE2
			auto lhs_vec = _mm_loadu_pd(&lhs.x);
			auto rhs_vec = _mm_loadu_pd(&rhs.x);

			auto res_sub = _mm_sub_pd(lhs_vec, rhs_vec);
			auto res_pow = _mm_mul_pd(res_sub, res_sub);

			auto comp1 = _mm_castsi128_pd(_mm_bsrli_si128(_mm_castpd_si128(res_pow), 8));

			auto sum = _mm_add_pd(res_pow, comp1);

			auto result = _mm_sqrt_pd(sum);

			alignas(16) std::array<double, 2> outvec;
			_mm_store_pd(outvec.data(), result);
			return outvec[0];
		#else
			auto xs = std::pow(lhs.x - rhs.x, 2);
			auto ys = std::pow(lhs.y - rhs.y, 2);
			return std::sqrt(xs + ys);
		#endif
	}

	float Distance(const Vector3<float>& lhs, const Vector3<float>& rhs) {
		#if X86_SIMD_LEVEL >= LV_SSE2
			auto lhs_vec = _mm_loadu_ps(&lhs.x);
			auto rhs_vec = _mm_loadu_ps(&rhs.x);
			auto pre_addition = getPreAdd(lhs_vec, rhs_vec);

			auto comp1 = _mm_castsi128_ps(_mm_bsrli_si128(_mm_castps_si128(pre_addition), 4));
			auto comp2 = _mm_castsi128_ps(_mm_bsrli_si128(_mm_castps_si128(pre_addition), 8));

			auto sum = _mm_add_ps(_mm_add_ps(pre_addition, comp1), comp2);
			auto result = _mm_sqrt_ps(sum);

			alignas(16) std::array<float, 4> outvec;
			_mm_store_ps(outvec.data(), result);

			return outvec[0];
		#else
			auto xs = std::pow(lhs.x - rhs.x, 2);
			auto ys = std::pow(lhs.y - rhs.y, 2);
			auto zs = std::pow(lhs.z - rhs.z, 2);

			return std::sqrt(xs + ys + zs);
		#endif
	}

	double Distance(const Vector3<double>& lhs, const Vector3<double>& rhs) {
		#if X86_SIMD_LEVEL >= LV_AVX
			auto lhs_vec = _mm256_loadu_pd(&lhs.x);
			auto rhs_vec = _mm256_loadu_pd(&rhs.x);

			auto res_sub = _mm256_sub_pd(lhs_vec, rhs_vec);
			auto res_pow = _mm256_mul_pd(res_sub, res_sub);

			auto res_hadd = _mm256_hadd_pd(res_pow, res_pow);
			auto comp1 = wlib::simd::reverseLanes(res_pow);

			auto sum = _mm256_add_pd(res_hadd, comp1);
			auto result = _mm256_sqrt_pd(sum);

			alignas(32) std::array<double, 4> outvec;
			_mm256_store_pd(outvec.data(), result);
			return outvec[0];
		#elif X86_SIMD_LEVEL >= LV_SSE2
			auto lhs_vec = _mm_loadu_pd(&lhs.x);
			auto rhs_vec = _mm_loadu_pd(&rhs.x);

			auto lhs_vec_1 = _mm_loadu_pd(&lhs.z);
			auto rhs_vec_1 = _mm_loadu_pd(&rhs.z);

			auto res_sub = _mm_sub_pd(lhs_vec, rhs_vec);
			auto res_pow = _mm_mul_pd(res_sub, res_sub);
			auto res_sub_1 = _mm_sub_pd(lhs_vec_1, rhs_vec_1);
			auto res_pow_1 = _mm_mul_pd(res_sub_1, res_sub_1);

			auto comp1 = _mm_castsi128_pd(_mm_bsrli_si128(_mm_castpd_si128(res_pow), 8));

			auto sum = _mm_add_pd(_mm_add_pd(res_pow, comp1), res_pow_1);

			auto result = _mm_sqrt_pd(sum);

			alignas(16) std::array<double, 2> outvec;
			_mm_store_pd(outvec.data(), result);
			return outvec[0];
		#else
			auto xs = std::pow(lhs.x - rhs.x, 2);
			auto ys = std::pow(lhs.y - rhs.y, 2);
			auto zs = std::pow(lhs.z - rhs.z, 2);

			return std::sqrt(xs + ys + zs);
		#endif
	}

	float Distance(const Vector4<float>& lhs, const Vector4<float>& rhs) {
		#if X86_SIMD_LEVEL >= LV_SSE2
			auto lhs_vec = _mm_loadu_ps(&lhs.x);
			auto rhs_vec = _mm_loadu_ps(&rhs.x);
			auto pre_addition = getPreAdd(lhs_vec, rhs_vec);

			auto comp1 = _mm_castsi128_ps(_mm_bsrli_si128(_mm_castps_si128(pre_addition), 4));
			auto comp2 = _mm_castsi128_ps(_mm_bsrli_si128(_mm_castps_si128(pre_addition), 8));
			auto comp3 = _mm_castsi128_ps(_mm_bsrli_si128(_mm_castps_si128(pre_addition), 12));

			auto sum = _mm_add_ps(_mm_add_ps(_mm_add_ps(pre_addition, comp1), comp2), comp3);
			auto result = _mm_sqrt_ps(sum);

			alignas(16) std::array<float, 4> outvec;
			_mm_store_ps(outvec.data(), result);
			return outvec[0];
		#else
			auto xs = std::pow(lhs.x - rhs.x, 2);
			auto ys = std::pow(lhs.y - rhs.y, 2);
			auto zs = std::pow(lhs.z - rhs.z, 2);
			auto ws = std::pow(lhs.w - rhs.w, 2);

			return std::sqrt(xs + ys + zs + ws);
		#endif
	}

	double Distance(const Vector4<double>& lhs, const Vector4<double>& rhs) {
		#if X86_SIMD_LEVEL >= LV_AVX
			auto lhs_vec = _mm256_loadu_pd(&lhs.x);
			auto rhs_vec = _mm256_loadu_pd(&rhs.x);

			auto res_sub = _mm256_sub_pd(lhs_vec, rhs_vec);
			auto res_pow = _mm256_mul_pd(res_sub, res_sub);

			auto res_hadd = _mm256_hadd_pd(res_pow, res_pow);
			auto res_hadd_rev = wlib::simd::reverseLanes(res_hadd);

			auto sum = _mm256_add_pd(res_hadd, res_hadd_rev);
			auto result = _mm256_sqrt_pd(sum);

			alignas(32) std::array<double, 4> outvec;
			_mm256_store_pd(outvec.data(), result);
			return outvec[0];
		#elif X86_SIMD_LEVEL >= LV_SSE2
			auto lhs_vec = _mm_loadu_pd(&lhs.x);
			auto rhs_vec = _mm_loadu_pd(&rhs.x);

			auto lhs_vec_1 = _mm_loadu_pd(&lhs.z);
			auto rhs_vec_1 = _mm_loadu_pd(&rhs.z);

			auto res_sub = _mm_sub_pd(lhs_vec, rhs_vec);
			auto res_pow = _mm_mul_pd(res_sub, res_sub);
			auto res_sub_1 = _mm_sub_pd(lhs_vec_1, rhs_vec_1);
			auto res_pow_1 = _mm_mul_pd(res_sub_1, res_sub_1);

			auto comp1 = _mm_castsi128_pd(_mm_bsrli_si128(_mm_castpd_si128(res_pow), 8));
			auto comp2 = _mm_castsi128_pd(_mm_bsrli_si128(_mm_castpd_si128(res_pow_1), 8));

			auto sum = _mm_add_pd(_mm_add_pd(_mm_add_pd(res_pow, comp1), res_pow_1), comp2);

			auto result = _mm_sqrt_pd(sum);

			alignas(16) std::array<double, 2> outvec;
			_mm_store_pd(outvec.data(), result);
			return outvec[0];
		#else
			auto xs = std::pow(lhs.x - rhs.x, 2);
			auto ys = std::pow(lhs.y - rhs.y, 2);
			auto zs = std::pow(lhs.z - rhs.z, 2);
			auto ws = std::pow(lhs.w - rhs.w, 2);

			return std::sqrt(xs + ys + zs + ws);
		#endif
	}

} // namespace wlib::vecmath
