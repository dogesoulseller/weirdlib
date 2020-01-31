#ifdef WEIRDLIB_ENABLE_VECTOR_MATH
#include "../../include/weirdlib_vecmath.hpp"
#include "../../include/cpu_detection.hpp"
#include "../../include/weirdlib_simdhelper.hpp"

#include <array>

namespace wlib::vecmath
{
	Vector2<double> Normalize(const Vector2<double>& vec) {
		auto magnitude = Length(vec);

		#if X86_SIMD_LEVEL >= LV_SSE2
			auto magnitude_vec = _mm_set1_pd(magnitude);
			auto vec_simd = _mm_loadu_pd(&vec.x);

			auto result = _mm_div_pd(vec_simd, magnitude_vec);

			alignas(16) std::array<double, 2> outvec;
			_mm_store_pd(outvec.data(), result);
			return Vector2(outvec[0], outvec[1]);
		#else
			return Vector2(vec.x / magnitude, vec.y / magnitude);
		#endif
	}

	Vector3<double> Normalize(const Vector3<double>& vec) {
		auto magnitude = Length(vec);
		#if X86_SIMD_LEVEL >= LV_AVX
			auto magnitude_vec = _mm256_set1_pd(magnitude);
			auto vec_simd = _mm256_loadu_pd(&vec.x);

			auto result = _mm256_div_pd(vec_simd, magnitude_vec);

			alignas(32) std::array<double, 4> outvec;
			_mm256_store_pd(outvec.data(), result);
			return Vector3(outvec[0], outvec[1], outvec[2]);
		#elif X86_SIMD_LEVEL >= LV_SSE2
			auto vec_simd_0 = _mm_loadu_pd(&vec.x);
			auto vec_simd_1 = _mm_loadu_pd(&vec.z);

			auto magnitude_vec = _mm_set1_pd(magnitude);

			auto result0 = _mm_div_pd(vec_simd_0, magnitude_vec);
			auto result1 = _mm_div_pd(vec_simd_1, magnitude_vec);

			alignas(16) std::array<double, 4> outvec;
			_mm_store_pd(outvec.data(), result0);
			_mm_store_pd(outvec.data()+2, result1);
			return Vector3(outvec[0], outvec[1], outvec[2]);
		#else
			return Vector3(vec.x / magnitude, vec.y / magnitude, vec.z / magnitude);
		#endif
	}

	Vector4<double> Normalize(const Vector4<double>& vec) {
		auto magnitude = Length(vec);
		#if X86_SIMD_LEVEL >= LV_AVX
			auto magnitude_vec = _mm256_set1_pd(magnitude);
			auto vec_simd = _mm256_loadu_pd(&vec.x);

			auto result = _mm256_div_pd(vec_simd, magnitude_vec);

			alignas(32) std::array<double, 4> outvec;
			_mm256_store_pd(outvec.data(), result);
			return Vector4(outvec[0], outvec[1], outvec[2], outvec[3]);
		#elif X86_SIMD_LEVEL >= LV_SSE2
			auto vec_simd_0 = _mm_loadu_pd(&vec.x);
			auto vec_simd_1 = _mm_loadu_pd(&vec.z);

			auto magnitude_vec = _mm_set1_pd(magnitude);

			auto result0 = _mm_div_pd(vec_simd_0, magnitude_vec);
			auto result1 = _mm_div_pd(vec_simd_1, magnitude_vec);

			alignas(16) std::array<double, 4> outvec;
			_mm_store_pd(outvec.data(), result0);
			_mm_store_pd(outvec.data()+2, result1);
			return Vector4(outvec[0], outvec[1], outvec[2], outvec[3]);
		#else
			return Vector4(vec.x / magnitude, vec.y / magnitude, vec.z / magnitude, vec.w / magnitude);
		#endif
	}

	Vector2<float> Normalize(const Vector2<float>& vec) {
		auto magnitude = Length(vec);

		#if X86_SIMD_LEVEL >= LV_SSE
			auto magnitude_vec = _mm_set1_ps(magnitude);
			auto vec_simd = _mm_loadu_ps(&vec.x);

			auto result = _mm_div_ps(vec_simd, magnitude_vec);

			alignas(16) std::array<float, 4> outvec;
			_mm_store_ps(outvec.data(), result);
			return Vector2(outvec[0], outvec[1]);
		#else
			return Vector2(vec.x / magnitude, vec.y / magnitude);
		#endif
	}

	Vector3<float> Normalize(const Vector3<float>& vec) {
		auto magnitude = Length(vec);

		#if X86_SIMD_LEVEL >= LV_SSE
			auto magnitude_vec = _mm_set1_ps(magnitude);
			auto vec_simd = _mm_loadu_ps(&vec.x);

			auto result = _mm_div_ps(vec_simd, magnitude_vec);

			alignas(16) std::array<float, 4> outvec;
			_mm_store_ps(outvec.data(), result);
			return Vector3(outvec[0], outvec[1], outvec[2]);
		#else
			return Vector3(vec.x / magnitude, vec.y / magnitude, vec.z / magnitude);
		#endif
	}

	Vector4<float> Normalize(const Vector4<float>& vec) {
		auto magnitude = Length(vec);

		#if X86_SIMD_LEVEL >= LV_SSE
			auto magnitude_vec = _mm_set1_ps(magnitude);
			auto vec_simd = _mm_loadu_ps(&vec.x);

			auto result = _mm_div_ps(vec_simd, magnitude_vec);

			alignas(16) std::array<float, 4> outvec;
			_mm_store_ps(outvec.data(), result);
			return Vector4(outvec[0], outvec[1], outvec[2], outvec[3]);
		#else
			return Vector4(vec.x / magnitude, vec.y / magnitude, vec.z / magnitude, vec.w / magnitude);
		#endif
	}

	Vector2<float> NormalizeApprox(const Vector2<float>& vec) {
		auto magnitude = Length(vec);

		#if X86_SIMD_LEVEL >= LV_SSE
			auto magnitude_vec_rcp = _mm_rcp_ps(_mm_set1_ps(magnitude));
			auto vec_simd = _mm_loadu_ps(&vec.x);

			auto result = _mm_mul_ps(magnitude_vec_rcp, vec_simd);

			alignas(16) std::array<float, 4> outvec;
			_mm_store_ps(outvec.data(), result);
			return Vector2(outvec[0], outvec[1]);
		#else
			return Vector2(vec.x / magnitude, vec.y / magnitude);
		#endif
	}

	Vector3<float> NormalizeApprox(const Vector3<float>& vec) {
		auto magnitude = Length(vec);

		#if X86_SIMD_LEVEL >= LV_SSE
			auto magnitude_vec_rcp = _mm_rcp_ps(_mm_set1_ps(magnitude));
			auto vec_simd = _mm_loadu_ps(&vec.x);

			auto result = _mm_mul_ps(magnitude_vec_rcp, vec_simd);

			alignas(16) std::array<float, 4> outvec;
			_mm_store_ps(outvec.data(), result);
			return Vector3(outvec[0], outvec[1], outvec[2]);
		#else
			return Vector3(vec.x / magnitude, vec.y / magnitude, vec.z / magnitude);
		#endif
	}

	Vector4<float> NormalizeApprox(const Vector4<float>& vec) {
		auto magnitude = Length(vec);

		#if X86_SIMD_LEVEL >= LV_SSE
			auto magnitude_vec_rcp = _mm_rcp_ps(_mm_set1_ps(magnitude));
			auto vec_simd = _mm_loadu_ps(&vec.x);

			auto result = _mm_mul_ps(magnitude_vec_rcp, vec_simd);

			alignas(16) std::array<float, 4> outvec;
			_mm_store_ps(outvec.data(), result);
			return Vector4(outvec[0], outvec[1], outvec[2], outvec[3]);
		#else
			return Vector4(vec.x / magnitude, vec.y / magnitude, vec.z / magnitude, vec.w / magnitude);
		#endif
	}
} // namespace wlib::vecmath
#endif
