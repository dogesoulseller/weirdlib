#include "../../include/weirdlib_vecmath.hpp"
#include "../../include/cpu_detection.hpp"
#include "../../include/weirdlib_simdhelper.hpp"

#include <array>
namespace wlib::vecmath
{
	#if X86_SIMD_LEVEL >= LV_AVX2
		static const auto PERM_MASK_CROSSP_LHS = _mm256_set_epi32(0, 0, 1, 0, 2, 0, 2, 1);
		static const auto PERM_MASK_CROSSP_RHS = _mm256_set_epi32(0, 0, 0, 2, 1, 1, 0, 2);
		static const auto PERM_MASK_CROSSP_SUB = _mm256_set_epi32(0, 0, 5, 2, 4, 1, 3, 0);
	#endif

	Vector3<float> CrossProduct(const Vector3<float>& lhs, const Vector3<float>& rhs) {
		// TODO: This could be improved
		#if X86_SIMD_LEVEL >= LV_AVX2
			auto lhs_vec = _mm256_loadu_ps(&lhs.x);
			auto rhs_vec = _mm256_loadu_ps(&rhs.x);

			auto lhs_yzxzxy = _mm256_permutevar8x32_ps(lhs_vec, PERM_MASK_CROSSP_LHS);
			auto rhs_zxyyzx = _mm256_permutevar8x32_ps(rhs_vec, PERM_MASK_CROSSP_RHS);

			auto product = _mm256_mul_ps(lhs_yzxzxy, rhs_zxyyzx);

			// Set up for subtraction
			auto prod_hsub = _mm256_permutevar8x32_ps(product, PERM_MASK_CROSSP_SUB);
			auto result = _mm256_hsub_ps(prod_hsub, prod_hsub);

			alignas(32) std::array<float, 8> outvec;
			_mm256_store_ps(outvec.data(), result);

			return Vector3(outvec[0], outvec[1], outvec[4]);

		#elif X86_SIMD_LEVEL >= LV_SSE
			auto lhs_vec = _mm_loadu_ps(&lhs.x);
			auto rhs_vec = _mm_loadu_ps(&rhs.x);

			auto lhs_yzx = _mm_shuffle_ps(lhs_vec, lhs_vec, _MM_SHUFFLE(0, 0, 2, 1));
			auto lhs_zxy = _mm_shuffle_ps(lhs_vec, lhs_vec, _MM_SHUFFLE(0, 1, 0, 2));

			auto rhs_zxy = _mm_shuffle_ps(rhs_vec, rhs_vec, _MM_SHUFFLE(0, 1, 0, 2));
			auto rhs_yzx = _mm_shuffle_ps(rhs_vec, rhs_vec, _MM_SHUFFLE(0, 0, 2, 1));

			auto left_side  = _mm_mul_ps(lhs_yzx, rhs_zxy);
			auto right_side = _mm_mul_ps(lhs_zxy, rhs_yzx);

			auto result = _mm_sub_ps(left_side, right_side);
			alignas(16) std::array<float, 4> outvec;
			_mm_store_ps(outvec.data(), result);

			return Vector3(outvec[0], outvec[1], outvec[2]);
		#else
			float x = (lhs.y * rhs.z) - (lhs.z * rhs.y);
			float y = (lhs.z * rhs.x) - (lhs.x * rhs.z);
			float z = (lhs.x * rhs.y) - (lhs.y * rhs.x);

			return Vector3(x, y, z);
		#endif
	}

	Vector3<double> CrossProduct(const Vector3<double>& lhs, const Vector3<double>& rhs) {
		#if X86_SIMD_LEVEL >= LV_AVX2
			auto lhs_vec = _mm256_loadu_pd(&lhs.x);
			auto rhs_vec = _mm256_loadu_pd(&rhs.x);

			auto lhs_yzx = _mm256_permute4x64_pd(lhs_vec, 0b00001001);
			auto lhs_zxy = _mm256_permute4x64_pd(lhs_vec, 0b00010010);

			auto rhs_yzx = _mm256_permute4x64_pd(rhs_vec, 0b00001001);
			auto rhs_zxy = _mm256_permute4x64_pd(rhs_vec, 0b00010010);

			auto left_side = _mm256_mul_pd(lhs_yzx, rhs_zxy);
			auto right_side = _mm256_mul_pd(lhs_zxy, rhs_yzx);

			auto result = _mm256_sub_pd(left_side, right_side);

			alignas(32) std::array<double, 4> outvec;
			_mm256_store_pd(outvec.data(), result);

			return Vector3(outvec[0], outvec[1], outvec[2]);
		#elif X86_SIMD_LEVEL >= LV_SSE2
			auto lhs_vec_xy = _mm_loadu_pd(&lhs.x);
			auto lhs_vec_z_ = _mm_loadu_pd(&lhs.z);

			auto rhs_vec_xy = _mm_loadu_pd(&rhs.x);
			auto rhs_vec_z_ = _mm_loadu_pd(&rhs.z);

			auto lhs_yz = _mm_shuffle_pd(lhs_vec_xy, lhs_vec_z_, 0b00000001);
			auto lhs_zx = _mm_shuffle_pd(lhs_vec_z_, lhs_vec_xy, 0b00000000);
			auto lhs_x_ = _mm_shuffle_pd(lhs_vec_xy, lhs_vec_xy, 0b00000000);
			auto lhs_y_ = _mm_shuffle_pd(lhs_vec_xy, lhs_vec_xy, 0b00000001);

			auto rhs_yz = _mm_shuffle_pd(rhs_vec_xy, rhs_vec_z_, 0b00000001);
			auto rhs_zx = _mm_shuffle_pd(rhs_vec_z_, rhs_vec_xy, 0b00000000);
			auto rhs_x_ = _mm_shuffle_pd(rhs_vec_xy, rhs_vec_xy, 0b00000000);
			auto rhs_y_ = _mm_shuffle_pd(rhs_vec_xy, rhs_vec_xy, 0b00000001);

			auto l0 = _mm_mul_pd(lhs_yz, rhs_zx);
			auto r0 = _mm_mul_pd(lhs_zx, rhs_yz);
			auto l1 = _mm_mul_pd(lhs_x_, rhs_y_);
			auto r1 = _mm_mul_pd(lhs_y_, rhs_x_);

			auto result0 = _mm_sub_pd(l0, r0);
			auto result1 = _mm_sub_pd(l1, r1);

			alignas(16) std::array<double, 4> outvec;
			_mm_store_pd(outvec.data(), result0);
			_mm_store_pd(outvec.data()+2, result1);

			return Vector3(outvec[0], outvec[1], outvec[2]);
		#else
			double x = (lhs.y * rhs.z) - (lhs.z * rhs.y);
			double y = (lhs.z * rhs.x) - (lhs.x * rhs.z);
			double z = (lhs.x * rhs.y) - (lhs.y * rhs.x);

			return Vector3(x, y, z);
		#endif
	}
} // namespace wlib::vecmath
