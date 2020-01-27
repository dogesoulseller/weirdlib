#include "../../include/weirdlib_vecmath.hpp"
#include "../../include/cpu_detection.hpp"
#include "../../include/weirdlib_simdhelper.hpp"

#include <array>
#include <cmath>
#include <tuple>

namespace wlib::vecmath
{
	#if X86_SIMD_LEVEL >= LV_SSE
	static const auto SSE_ONE_MASK_FLT = _mm_set1_ps(1.0f);
	#endif

	#if X86_SIMD_LEVEL >= LV_SSE2
	static const auto SSE_ONE_MASK_DBL = _mm_set1_pd(1.0);
	#endif

	#if X86_SIMD_LEVEL >= LV_AVX
	static const auto AVX_ONE_MASK_FLT = _mm256_set1_ps(1.0f);
	static const auto AVX_ONE_MASK_DBL = _mm256_set1_pd(1.0);
	#endif

	#if X86_SIMD_LEVEL >= LV_SSE
		template<typename VecT>
		std::tuple<std::array<uint32_t, 4>, std::array<float, 4>> getRefractFloat(float eta, float dotProduct,
		  const VecT& incident, const VecT& surfaceNormal) {
			// Broadcast eta and the dot product onto vector
			auto eta_vec = _mm_set1_ps(eta);
			auto dotProduct_vec = _mm_set1_ps(dotProduct);

			auto incident_vec = _mm_loadu_ps(&incident.x);
			auto surfaceNormal_vec = _mm_loadu_ps(&surfaceNormal.x);

			#ifdef X86_SIMD_FMA
				// Calculate k, spreading it across the register
				auto parenth_k = _mm_fnmadd_ps(dotProduct_vec, dotProduct_vec, SSE_ONE_MASK_FLT);
				auto k_vec = _mm_fnmadd_ps(eta_vec, _mm_mul_ps(eta_vec, parenth_k), SSE_ONE_MASK_FLT);

				// Calculate result
				auto parenth_result = _mm_fmadd_ps(eta_vec, dotProduct_vec, _mm_sqrt_ps(k_vec));
				auto result = _mm_fmsub_ps(eta_vec, incident_vec, _mm_mul_ps(parenth_result, surfaceNormal_vec));
			#else
				// Calculate k, spreading it across the register
				auto dp_square = _mm_mul_ps(dotProduct_vec, dotProduct_vec);
				auto eta_square = _mm_mul_ps(eta_vec, eta_vec);
				auto parenth_k = _mm_sub_ps(SSE_ONE_MASK_FLT, dp_square);
				auto k_vec = _mm_sub_ps(SSE_ONE_MASK_FLT, _mm_mul_ps(eta_square, parenth_k));

				// Calculate result
				auto parenth_result = _mm_add_ps(_mm_mul_ps(eta_vec, dotProduct_vec), _mm_sqrt_ps(k_vec));
				auto result = _mm_sub_ps(_mm_mul_ps(eta_vec, incident_vec), _mm_mul_ps(parenth_result, surfaceNormal_vec));
			#endif

			// Make mask of negative k values
			auto compare_mask = _mm_cmplt_ps(k_vec, simd::SIMD128F_zeroMask);

			// Store result and mask in memory
			alignas(16) std::array<uint32_t, 4> maskvec;
			alignas(16) std::array<float, 4> outvec;
			_mm_store_ps(reinterpret_cast<float*>(maskvec.data()), compare_mask);
			_mm_store_ps(outvec.data(), result);

			return std::make_tuple(maskvec, outvec);
		}
	#endif

	#if X86_SIMD_LEVEL >= LV_SSE2
		template<typename VecT>
		std::tuple<std::array<uint64_t, 2>, std::array<double, 2>> getRefractDoubleSimple(double eta, double dotProduct,
		  const VecT& incident, const VecT& surfaceNormal) {
		    // Broadcast eta and the dot product onto vector
			auto eta_vec = _mm_set1_pd(eta);
			auto dotProduct_vec = _mm_set1_pd(dotProduct);

			auto incident_vec = _mm_loadu_pd(&incident.x);
			auto surfaceNormal_vec = _mm_loadu_pd(&surfaceNormal.x);

			#ifdef X86_SIMD_FMA
				// Calculate k, spreading it across the register
				auto parenth_k = _mm_fnmadd_pd(dotProduct_vec, dotProduct_vec, SSE_ONE_MASK_DBL);
				auto k_vec = _mm_fnmadd_pd(eta_vec, _mm_mul_pd(eta_vec, parenth_k), SSE_ONE_MASK_DBL);

				// Calculate result
				auto parenth_result = _mm_fmadd_pd(eta_vec, dotProduct_vec, _mm_sqrt_pd(k_vec));
				auto result = _mm_fmsub_pd(eta_vec, incident_vec, _mm_mul_pd(parenth_result, surfaceNormal_vec));
			#else
				// Calculate k, spreading it across the register
				auto dp_square = _mm_mul_pd(dotProduct_vec, dotProduct_vec);
				auto eta_square = _mm_mul_pd(eta_vec, eta_vec);
				auto parenth_k = _mm_sub_pd(SSE_ONE_MASK_DBL, dp_square);
				auto k_vec = _mm_sub_pd(SSE_ONE_MASK_DBL, _mm_mul_pd(eta_square, parenth_k));

				// Calculate result
				auto parenth_result = _mm_add_pd(_mm_mul_pd(eta_vec, dotProduct_vec), _mm_sqrt_pd(k_vec));
				auto result = _mm_sub_pd(_mm_mul_pd(eta_vec, incident_vec), _mm_mul_pd(parenth_result, surfaceNormal_vec));
			#endif

			// Make mask of negative k values
			auto compare_mask = _mm_cmplt_pd(k_vec, simd::SIMD128D_zeroMask);

			// Store result and mask in memory
			alignas(16) std::array<uint64_t, 2> maskvec;
			alignas(16) std::array<double, 2> outvec;
			_mm_store_pd(reinterpret_cast<double*>(maskvec.data()), compare_mask);
			_mm_store_pd(outvec.data(), result);

			return std::make_tuple(maskvec, outvec);
		}

		template<typename VecT>
		std::tuple<std::array<uint64_t, 4>, std::array<double, 4>> getRefractDoubleCompl(double eta, double dotProduct,
	  	  const VecT& incident, const VecT& surfaceNormal) {
			#if X86_SIMD_LEVEL >= LV_AVX
				// Broadcast eta and the dot product onto vector
				auto eta_vec = _mm256_set1_pd(eta);
				auto dotProduct_vec = _mm256_set1_pd(dotProduct);

				auto incident_vec = _mm256_loadu_pd(&incident.x);
				auto surfaceNormal_vec = _mm256_loadu_pd(&surfaceNormal.x);

				#ifdef X86_SIMD_FMA
					// Calculate k, spreading it across the register
					auto parenth_k = _mm256_fnmadd_pd(dotProduct_vec, dotProduct_vec, AVX_ONE_MASK_DBL);
					auto k_vec = _mm256_fnmadd_pd(eta_vec, _mm256_mul_pd(eta_vec, parenth_k), AVX_ONE_MASK_DBL);

					// Calculate result
					auto parenth_result = _mm256_fmadd_pd(eta_vec, dotProduct_vec, _mm256_sqrt_pd(k_vec));
					auto result = _mm256_fmsub_pd(eta_vec, incident_vec, _mm256_mul_pd(parenth_result, surfaceNormal_vec));
				#else
					// Calculate k, spreading it across the register
					auto dp_square = _mm256_mul_pd(dotProduct_vec, dotProduct_vec);
					auto eta_square = _mm256_mul_pd(eta_vec, eta_vec);
					auto parenth_k = _mm256_sub_pd(AVX_ONE_MASK_FLT, dp_square);
					auto k_vec = _mm256_sub_pd(AVX_ONE_MASK_FLT, _mm256_mul_pd(eta_square, parenth_k));

					// Calculate result
					auto parenth_result = _mm256_add_pd(_mm256_mul_pd(eta_vec, dotProduct_vec), _mm256_sqrt_pd(k_vec));
					auto result = _mm256_sub_pd(_mm256_mul_pd(eta_vec, incident_vec), _mm256_mul_pd(parenth_result, surfaceNormal_vec));
				#endif

				// Make mask of negative k values
				auto compare_mask = _mm256_cmp_pd(k_vec, simd::SIMD256D_zeroMask, _CMP_LT_OQ);

				// Store result and mask in memory
				alignas(32) std::array<uint64_t, 4> maskvec;
				alignas(32) std::array<double, 4> outvec;
				_mm256_store_pd(reinterpret_cast<double*>(maskvec.data()), compare_mask);
				_mm256_store_pd(outvec.data(), result);

				return std::make_tuple(maskvec, outvec);
			#elif X86_SIMD_LEVEL >= LV_SSE2
				// Broadcast eta and the dot product onto vector
				auto eta_vec = _mm_set1_pd(eta);
				auto dotProduct_vec = _mm_set1_pd(dotProduct);

				auto incident_vec0 = _mm_loadu_pd(&incident.x);
				auto incident_vec1 = _mm_loadu_pd(&incident.z);
				auto surfaceNormal_vec0 = _mm_loadu_pd(&surfaceNormal.x);
				auto surfaceNormal_vec1 = _mm_loadu_pd(&surfaceNormal.z);

				#ifdef X86_SIMD_FMA
					// Calculate k, spreading it across the register
					auto parenth_k = _mm_fnmadd_pd(dotProduct_vec, dotProduct_vec, SSE_ONE_MASK_DBL);
					auto k_vec = _mm_fnmadd_pd(eta_vec, _mm_mul_pd(eta_vec, parenth_k), SSE_ONE_MASK_DBL);

					// Calculate result
					auto parenth_result = _mm_fmadd_pd(eta_vec, dotProduct_vec, _mm_sqrt_pd(k_vec));
					auto result0 = _mm_fmsub_pd(eta_vec, incident_vec0, _mm_mul_pd(parenth_result, surfaceNormal_vec0));
					auto result1 = _mm_fmsub_pd(eta_vec, incident_vec1, _mm_mul_pd(parenth_result, surfaceNormal_vec1));
				#else
					// Calculate k, spreading it across the register
					auto dp_square = _mm_mul_pd(dotProduct_vec, dotProduct_vec);
					auto eta_square = _mm_mul_pd(eta_vec, eta_vec);
					auto parenth_k = _mm_sub_pd(SSE_ONE_MASK_DBL, dp_square);
					auto k_vec = _mm_sub_pd(SSE_ONE_MASK_DBL, _mm_mul_pd(eta_square, parenth_k));

					// Calculate result
					auto parenth_result = _mm_add_pd(_mm_mul_pd(eta_vec, dotProduct_vec), _mm_sqrt_pd(k_vec));
					auto result0 = _mm_sub_pd(_mm_mul_pd(eta_vec, incident_vec0), _mm_mul_pd(parenth_result, surfaceNormal_vec0));
					auto result1 = _mm_sub_pd(_mm_mul_pd(eta_vec, incident_vec1), _mm_mul_pd(parenth_result, surfaceNormal_vec1));
				#endif

				// Make mask of negative k values
				auto compare_mask = _mm_cmplt_pd(k_vec, simd::SIMD128D_zeroMask);

				// Store result and mask in memory
				alignas(16) std::array<uint64_t, 4> maskvec;
				alignas(16) std::array<double, 4> outvec;
				_mm_store_pd(reinterpret_cast<double*>(maskvec.data()), compare_mask);
				_mm_store_pd(outvec.data(), result0);
				_mm_store_pd(outvec.data()+2, result1);

				return std::make_tuple(maskvec, outvec);
			#endif
		}
	#endif

	Vector2<double> Refract(const Vector2<double>& incident, const Vector2<double>& surfaceNormal, double eta) {
		auto dotProduct = DotProduct(surfaceNormal, incident);

		#if X86_SIMD_LEVEL >= LV_SSE2
			auto [maskvec, outvec] = getRefractDoubleSimple(eta, dotProduct, incident, surfaceNormal);

			auto x = maskvec[0] == 0xFFFFFFFFFFFFFFFF ? 0.0 : outvec[0];
			auto y = maskvec[0] == 0xFFFFFFFFFFFFFFFF ? 0.0 : outvec[1];

			return Vector2(x, y);
		#else
			auto k = 1.0 - eta * eta * (1.0 - dotProduct * dotProduct);

			auto x = k < 0.0 ? 0.0 : eta * incident.x - (eta * dotProduct + std::sqrt(k)) * surfaceNormal.x;
			auto y = k < 0.0 ? 0.0 : eta * incident.y - (eta * dotProduct + std::sqrt(k)) * surfaceNormal.y;

			return Vector2(x, y);
		#endif
	}

	Vector3<double> Refract(const Vector3<double>& incident, const Vector3<double>& surfaceNormal, double eta) {
		auto dotProduct = DotProduct(surfaceNormal, incident);
		#if X86_SIMD_LEVEL >= LV_SSE2
			auto [maskvec, outvec] = getRefractDoubleCompl(eta, dotProduct, incident, surfaceNormal);

			auto x = maskvec[0] == 0xFFFFFFFFFFFFFFFF ? 0.0 : outvec[0];
			auto y = maskvec[0] == 0xFFFFFFFFFFFFFFFF ? 0.0 : outvec[1];
			auto z = maskvec[0] == 0xFFFFFFFFFFFFFFFF ? 0.0 : outvec[2];

			return Vector3(x, y, z);
		#else
			auto k = 1.0 - eta * eta * (1.0 - dotProduct * dotProduct);

			auto x = k < 0.0 ? 0.0 : eta * incident.x - (eta * dotProduct + std::sqrt(k)) * surfaceNormal.x;
			auto y = k < 0.0 ? 0.0 : eta * incident.y - (eta * dotProduct + std::sqrt(k)) * surfaceNormal.y;
			auto z = k < 0.0 ? 0.0 : eta * incident.z - (eta * dotProduct + std::sqrt(k)) * surfaceNormal.z;

			return Vector3(x, y, z);
		#endif
	}

	Vector4<double> Refract(const Vector4<double>& incident, const Vector4<double>& surfaceNormal, double eta) {
		auto dotProduct = DotProduct(surfaceNormal, incident);

		#if X86_SIMD_LEVEL >= LV_SSE2
			auto [maskvec, outvec] = getRefractDoubleCompl(eta, dotProduct, incident, surfaceNormal);

			auto x = maskvec[0] == 0xFFFFFFFFFFFFFFFF ? 0.0 : outvec[0];
			auto y = maskvec[0] == 0xFFFFFFFFFFFFFFFF ? 0.0 : outvec[1];
			auto z = maskvec[0] == 0xFFFFFFFFFFFFFFFF ? 0.0 : outvec[2];
			auto w = maskvec[0] == 0xFFFFFFFFFFFFFFFF ? 0.0 : outvec[3];

			return Vector4(x, y, z, w);
		#else
			auto k = 1.0 - eta * eta * (1.0 - dotProduct * dotProduct);

			auto x = k < 0.0 ? 0.0 : eta * incident.x - (eta * dotProduct + std::sqrt(k)) * surfaceNormal.x;
			auto y = k < 0.0 ? 0.0 : eta * incident.y - (eta * dotProduct + std::sqrt(k)) * surfaceNormal.y;
			auto z = k < 0.0 ? 0.0 : eta * incident.z - (eta * dotProduct + std::sqrt(k)) * surfaceNormal.z;
			auto w = k < 0.0 ? 0.0 : eta * incident.w - (eta * dotProduct + std::sqrt(k)) * surfaceNormal.w;

			return Vector4(x, y, z, w);
		#endif
	}

	Vector2<float> Refract(const Vector2<float>& incident, const Vector2<float>& surfaceNormal, float eta) {
		auto dotProduct = DotProduct(surfaceNormal, incident);
		#if X86_SIMD_LEVEL >= LV_SSE
			auto [maskvec, outvec] = getRefractFloat(eta, dotProduct, incident, surfaceNormal);

			auto x = maskvec[0] == 0xFFFFFFFF ? 0.0f : outvec[0];
			auto y = maskvec[0] == 0xFFFFFFFF ? 0.0f : outvec[1];

			return Vector2(x, y);
		#else
			auto k = 1.0f - eta * eta * (1.0f - dotProduct * dotProduct);

			auto x = k < 0.0f ? 0.0f : eta * incident.x - (eta * dotProduct + std::sqrt(k)) * surfaceNormal.x;
			auto y = k < 0.0f ? 0.0f : eta * incident.y - (eta * dotProduct + std::sqrt(k)) * surfaceNormal.y;

			return Vector2(x, y);
		#endif
	}

	Vector3<float> Refract(const Vector3<float>& incident, const Vector3<float>& surfaceNormal, float eta) {
		auto dotProduct = DotProduct(surfaceNormal, incident);
		#if X86_SIMD_LEVEL >= LV_SSE
			auto [maskvec, outvec] = getRefractFloat(eta, dotProduct, incident, surfaceNormal);

			auto x = maskvec[0] == 0xFFFFFFFF ? 0.0f : outvec[0];
			auto y = maskvec[0] == 0xFFFFFFFF ? 0.0f : outvec[1];
			auto z = maskvec[0] == 0xFFFFFFFF ? 0.0f : outvec[2];

			return Vector3(x, y, z);
		#else
			auto k = 1.0f - eta * eta * (1.0f - dotProduct * dotProduct);

			auto x = k < 0.0f ? 0.0f : eta * incident.x - (eta * dotProduct + std::sqrt(k)) * surfaceNormal.x;
			auto y = k < 0.0f ? 0.0f : eta * incident.y - (eta * dotProduct + std::sqrt(k)) * surfaceNormal.y;
			auto z = k < 0.0f ? 0.0f : eta * incident.z - (eta * dotProduct + std::sqrt(k)) * surfaceNormal.z;

			return Vector3(x, y, z);
		#endif
	}

	Vector4<float> Refract(const Vector4<float>& incident, const Vector4<float>& surfaceNormal, float eta) {
		auto dotProduct = DotProduct(surfaceNormal, incident);
		#if X86_SIMD_LEVEL >= LV_SSE
			auto [maskvec, outvec] = getRefractFloat(eta, dotProduct, incident, surfaceNormal);

			// Check result, zero out if k is < 0
			auto x = maskvec[0] == 0xFFFFFFFF ? 0.0f : outvec[0];
			auto y = maskvec[0] == 0xFFFFFFFF ? 0.0f : outvec[1];
			auto z = maskvec[0] == 0xFFFFFFFF ? 0.0f : outvec[2];
			auto w = maskvec[0] == 0xFFFFFFFF ? 0.0f : outvec[3];

			return Vector4(x, y, z, w);
		#else
			auto k = 1.0f - eta * eta * (1.0f - dotProduct * dotProduct);

			auto x = k < 0.0f ? 0.0f : eta * incident.x - (eta * dotProduct + std::sqrt(k)) * surfaceNormal.x;
			auto y = k < 0.0f ? 0.0f : eta * incident.y - (eta * dotProduct + std::sqrt(k)) * surfaceNormal.y;
			auto z = k < 0.0f ? 0.0f : eta * incident.z - (eta * dotProduct + std::sqrt(k)) * surfaceNormal.z;
			auto w = k < 0.0f ? 0.0f : eta * incident.w - (eta * dotProduct + std::sqrt(k)) * surfaceNormal.w;

			return Vector4(x, y, z, w);
		#endif
	}

} // namespace wlib::vecmath
