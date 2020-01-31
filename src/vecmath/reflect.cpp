#ifdef WEIRDLIB_ENABLE_VECTOR_MATH
#include "../../include/weirdlib_vecmath.hpp"
#include "../../include/cpu_detection.hpp"
#include "../../include/weirdlib_simdhelper.hpp"

#include <array>

#if X86_SIMD_LEVEL >= LV_AVX
static const auto constTwosAVXFloat = _mm256_set1_ps(2.0f);
static const auto constTwosAVXDouble = _mm256_set1_pd(2.0);
#endif

#if X86_SIMD_LEVEL >= LV_SSE
static const auto constTwosSSEFloat = _mm_set1_ps(2.0f);
#endif

#if X86_SIMD_LEVEL >= LV_SSE2
static const auto constTwosSSEDouble = _mm_set1_pd(2.0);
#endif

#if X86_SIMD_LEVEL >= LV_SSE
	template<typename VecT>
	static std::array<float, 4> getReflectFloat(float DotP, const VecT& incident, const VecT& surfaceNormal) {
		auto incidentVec = _mm_loadu_ps(&incident.x);
		auto surfaceNormalVec = _mm_loadu_ps(&surfaceNormal.x);
		auto dotProduct = _mm_set1_ps(DotP);

		#if defined(X86_SIMD_FMA)
			auto result = _mm_fnmadd_ps(_mm_mul_ps(surfaceNormalVec, dotProduct), constTwosSSEFloat, incidentVec);
		#else
			auto product = _mm_mul_ps(_mm_mul_ps(surfaceNormalVec, dotProduct), constTwosSSEFloat);
			auto result = _mm_sub_ps(incidentVec, product);
		#endif

		alignas(16) std::array<float, 4> outvec;
		_mm_store_ps(outvec.data(), result);

		return outvec;
	}
#endif

#if X86_SIMD_LEVEL >= LV_SSE2
	template<typename VecT>
	static std::array<double, 4> getReflectDouble(double DotP, const VecT& incident, const VecT& surfaceNormal) {
		#if X86_SIMD_LEVEL >= LV_AVX
			auto incidentVec = _mm256_loadu_pd(&incident.x);
			auto surfaceNormalVec = _mm256_loadu_pd(&surfaceNormal.x);
			auto dotProduct = _mm256_set1_pd(DotP);

			#if defined(X86_SIMD_FMA)
				auto result = _mm256_fnmadd_pd(_mm256_mul_pd(surfaceNormalVec, dotProduct), constTwosAVXDouble, incidentVec);
			#else
				auto product = _mm256_mul_pd(_mm256_mul_pd(surfaceNormalVec, dotProduct), constTwosAVXDouble);
				auto result = _mm256_sub_pd(incidentVec, product);
			#endif

			alignas(32) std::array<double, 4> outvec;
			_mm256_store_pd(outvec.data(), result);

			return outvec;
		#elif X86_SIMD_LEVEL >= LV_SSE2
			auto incidentVec0 = _mm_loadu_pd(&incident.x);
			auto incidentVec1 = _mm_loadu_pd(&incident.z);
			auto surfaceNormalVec0 = _mm_loadu_pd(&surfaceNormal.x);
			auto surfaceNormalVec1 = _mm_loadu_pd(&surfaceNormal.z);
			auto dotProduct = _mm_set1_pd(DotP);

			#if defined(X86_SIMD_FMA)
				auto result0 = _mm_fnmadd_pd(_mm_mul_pd(surfaceNormalVec0, dotProduct), constTwosSSEDouble, incidentVec0);
				auto result1 = _mm_fnmadd_pd(_mm_mul_pd(surfaceNormalVec1, dotProduct), constTwosSSEDouble, incidentVec1);
			#else
				auto product0 = _mm_mul_pd(_mm_mul_pd(surfaceNormalVec0, dotProduct), constTwosSSEDouble);
				auto result0 = _mm_sub_pd(incidentVec0, product0);

				auto product1 = _mm_mul_pd(_mm_mul_pd(surfaceNormalVec1, dotProduct), constTwosSSEDouble);
				auto result1 = _mm_sub_pd(incidentVec1, product1);
			#endif

			alignas(16) std::array<double, 4> outvec;
			_mm_store_pd(outvec.data(), result0);
			_mm_store_pd(outvec.data()+2, result1);

			return outvec;
		#endif
	}
#endif


namespace wlib::vecmath
{
	Vector2<double> Reflect(const Vector2<double>& incident, const Vector2<double>& surfaceNormal) {
		auto DotP = DotProduct(surfaceNormal, incident);

		#if X86_SIMD_LEVEL >= LV_SSE2
			auto incidentVec = _mm_loadu_pd(&incident.x);
			auto surfaceNormalVec = _mm_loadu_pd(&surfaceNormal.x);
			auto dotProduct = _mm_set1_pd(DotP);

			#if defined(X86_SIMD_FMA)
				auto result = _mm_fnmadd_pd(_mm_mul_pd(surfaceNormalVec, dotProduct), constTwosSSEDouble, incidentVec);
			#else
				auto product = _mm_mul_pd(_mm_mul_pd(surfaceNormalVec, dotProduct), constTwosSSEDouble);
				auto result = _mm_sub_pd(incidentVec, product);
			#endif

			alignas(16) std::array<double, 2> outvec;
			_mm_store_pd(outvec.data(), result);

			return Vector2(outvec[0], outvec[1]);
		#else
			auto x = incident.x - 2.0 * DotP * surfaceNormal.x;
			auto y = incident.y - 2.0 * DotP * surfaceNormal.y;

			return Vector2(x, y);
		#endif
	}

	Vector3<double> Reflect(const Vector3<double>& incident, const Vector3<double>& surfaceNormal) {
		auto DotP = DotProduct(surfaceNormal, incident);

		#if X86_SIMD_LEVEL >= LV_SSE2
			auto outvec = getReflectDouble(DotP, incident, surfaceNormal);

			return Vector3(outvec[0], outvec[1], outvec[2]);
		#else
			auto x = incident.x - 2.0 * DotP * surfaceNormal.x;
			auto y = incident.y - 2.0 * DotP * surfaceNormal.y;
			auto z = incident.z - 2.0 * DotP * surfaceNormal.z;

			return Vector3(x, y, z);
		#endif
	}

	Vector4<double> Reflect(const Vector4<double>& incident, const Vector4<double>& surfaceNormal) {
		auto DotP = DotProduct(surfaceNormal, incident);

		#if X86_SIMD_LEVEL >= LV_SSE2
			auto outvec = getReflectDouble(DotP, incident, surfaceNormal);

			return Vector4(outvec[0], outvec[1], outvec[2], outvec[3]);
		#else
			auto x = incident.x - 2.0 * DotP * surfaceNormal.x;
			auto y = incident.y - 2.0 * DotP * surfaceNormal.y;
			auto z = incident.z - 2.0 * DotP * surfaceNormal.z;
			auto w = incident.w - 2.0 * DotP * surfaceNormal.w;

			return Vector4(x, y, z, w);
		#endif
	}

	Vector2<float> Reflect(const Vector2<float>& incident, const Vector2<float>& surfaceNormal) {
		auto DotP = DotProduct(surfaceNormal, incident);

		#if X86_SIMD_LEVEL >= LV_SSE2
			auto outvec = getReflectFloat(DotP, incident, surfaceNormal);

			return Vector2(outvec[0], outvec[1]);
		#else
			auto x = incident.x - 2.0f * DotP * surfaceNormal.x;
			auto y = incident.y - 2.0f * DotP * surfaceNormal.y;

			return Vector2(x, y);
		#endif
	}

	Vector3<float> Reflect(const Vector3<float>& incident, const Vector3<float>& surfaceNormal) {
		auto DotP = DotProduct(surfaceNormal, incident);

		#if X86_SIMD_LEVEL >= LV_SSE2
			auto outvec = getReflectFloat(DotP, incident, surfaceNormal);

			return Vector3(outvec[0], outvec[1], outvec[2]);
		#else
			auto x = incident.x - 2.0f * DotP * surfaceNormal.x;
			auto y = incident.y - 2.0f * DotP * surfaceNormal.y;
			auto z = incident.z - 2.0f * DotP * surfaceNormal.z;

			return Vector3(x, y, z);
		#endif
	}

	Vector4<float> Reflect(const Vector4<float>& incident, const Vector4<float>& surfaceNormal) {
		auto DotP = DotProduct(surfaceNormal, incident);

		#if X86_SIMD_LEVEL >= LV_SSE2
			auto outvec = getReflectFloat(DotP, incident, surfaceNormal);

			return Vector4(outvec[0], outvec[1], outvec[2], outvec[3]);
		#else
			auto x = incident.x - 2.0f * DotP * surfaceNormal.x;
			auto y = incident.y - 2.0f * DotP * surfaceNormal.y;
			auto z = incident.z - 2.0f * DotP * surfaceNormal.z;
			auto w = incident.w - 2.0f * DotP * surfaceNormal.w;

			return Vector4(x, y, z, w);
		#endif
	}

} // namespace wlib::vecmath
#endif
