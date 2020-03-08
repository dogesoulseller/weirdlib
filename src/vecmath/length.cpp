#ifdef WEIRDLIB_ENABLE_VECTOR_MATH
#include "../../include/weirdlib_vecmath.hpp"
#include "../../include/cpu_detection.hpp"

#include <cmath>
#include <array>


namespace wlib::vecmath
{
	float Length(const Vector2<float>& vec) {
		#if X86_SIMD_LEVEL >= LV_SSE
			auto vec_simd = _mm_loadu_ps(&vec.x);
			auto power2 = _mm_mul_ps(vec_simd, vec_simd);

			auto comp1 = _mm_shuffle_ps(power2, power2, _MM_SHUFFLE(0, 0, 0, 1));

			auto result = _mm_sqrt_ss(_mm_add_ss(power2, comp1));

			alignas(16) std::array<float, 4> outvec;
			_mm_store_ps(outvec.data(), result);
			return outvec[0];
		#else
			return std::sqrt(vec.x * vec.x + vec.y * vec.y);
		#endif
	}

	double Length(const Vector2<double>& vec) {
		#if X86_SIMD_LEVEL >= LV_SSE2
			auto vec_simd = _mm_loadu_pd(&vec.x);
			auto power2 = _mm_mul_pd(vec_simd, vec_simd);

			auto comp1 = _mm_shuffle_pd(power2, power2, _MM_SHUFFLE2(0, 1));
			auto result = _mm_sqrt_pd(_mm_add_sd(comp1, power2));

			alignas(16) std::array<double, 2> outvec;
			_mm_store_pd(outvec.data(), result);
			return outvec[0];
		#else
			return std::sqrt(vec.x * vec.x + vec.y * vec.y);
		#endif
	}

	float Length(const Vector3<float>& vec) {
		#if X86_SIMD_LEVEL >= LV_SSE
			auto vec_simd = _mm_loadu_ps(&vec.x);
			auto power2 = _mm_mul_ps(vec_simd, vec_simd);

			auto comp1 = _mm_shuffle_ps(power2, power2, _MM_SHUFFLE(0, 0, 0, 1));
			auto comp2 = _mm_shuffle_ps(power2, power2, _MM_SHUFFLE(0, 0, 0, 2));

			auto result = _mm_sqrt_ss(_mm_add_ss(comp1, _mm_add_ss(comp2, power2)));

			alignas(16) std::array<float, 4> outvec;
			_mm_store_ps(outvec.data(), result);
			return outvec[0];
		#else
			return std::sqrt(vec.x * vec.x + vec.y * vec.y + vec.z * vec.z);
		#endif
	}

	double Length(const Vector3<double>& vec) {
		#if X86_SIMD_LEVEL >= LV_SSE2
			auto vec_simd0 = _mm_loadu_pd(&vec.x);
			auto vec_simd1 = _mm_loadu_pd(&vec.z);

			auto power20 = _mm_mul_pd(vec_simd0, vec_simd0);
			auto power21 = _mm_mul_pd(vec_simd1, vec_simd1);

			auto comp1 = _mm_shuffle_pd(power20, power20, _MM_SHUFFLE2(0, 1));
			auto result = _mm_sqrt_pd(_mm_add_sd(_mm_add_sd(power21, power20), comp1));

			alignas(16) std::array<double, 2> outvec;
			_mm_store_pd(outvec.data(), result);
			return outvec[0];
		#else
			return std::sqrt(vec.x * vec.x + vec.y * vec.y + vec.z * vec.z);
		#endif
	}

	float Length(const Vector4<float>& vec) {
		#if X86_SIMD_LEVEL >= LV_SSE
			auto vec_simd = _mm_loadu_ps(&vec.x);
			auto power2 = _mm_mul_ps(vec_simd, vec_simd);

			// Addition in parallel, in first two positions
			auto comp23 = _mm_shuffle_ps(power2, power2, _MM_SHUFFLE(0, 0, 3, 2));

			auto partial_sum = _mm_add_ps(comp23, power2);

			auto comp_final = _mm_shuffle_ps(partial_sum, partial_sum, _MM_SHUFFLE(0, 0, 0, 1));

			auto result = _mm_sqrt_ss(_mm_add_ss(comp_final, partial_sum));

			alignas(16) std::array<float, 4> outvec;
			_mm_store_ps(outvec.data(), result);
			return outvec[0];
		#else
			return std::sqrt(vec.x * vec.x + vec.y * vec.y + vec.z * vec.z + vec.w * vec.w);
		#endif
	}

	double Length(const Vector4<double>& vec) {
		#if X86_SIMD_LEVEL >= LV_SSE2
			auto vec_simd0 = _mm_loadu_pd(&vec.x);
			auto vec_simd1 = _mm_loadu_pd(&vec.z);

			auto power20 = _mm_mul_pd(vec_simd0, vec_simd0);
			auto power21 = _mm_mul_pd(vec_simd1, vec_simd1);

			// Become x*z, y*w
			auto add0 = _mm_add_pd(power20, power21);

			auto comp1 = _mm_shuffle_pd(add0, add0, _MM_SHUFFLE2(0, 1));
			auto result = _mm_sqrt_pd(_mm_add_sd(comp1, add0));

			alignas(16) std::array<double, 2> outvec;
			_mm_store_pd(outvec.data(), result);
			return outvec[0];
		#else
			return std::sqrt(vec.x * vec.x + vec.y * vec.y + vec.z * vec.z + vec.w * vec.w);
		#endif
	}

} // namespace wlib::vecmath
#endif