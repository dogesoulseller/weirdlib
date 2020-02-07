#pragma once
#include "cpu_detection.hpp"
#include <type_traits>
#include <cstdint>
#include <algorithm>

namespace wlib
{
/// Operations commonly performed on SIMD types <br>
/// Compilers should manage to inline them
namespace simd
{
#if defined(__x86_64__) || defined(__X86__)
	#if defined(AVX512_BW)
		static const __m512i SIMD512B_zeroMask = _mm512_set1_epi8(0);
	#elif defined(AVX512_DQ)
		static const __m512i SIMD512B_zeroMask = _mm512_set1_epi32(0);
	#endif

	#if X86_SIMD_LEVEL >= LV_AVX512
		static const __m512 SIMD512F_zeroMask = _mm512_set1_ps(0.0f);
		static const __m512d SIMD512D_zeroMask = _mm512_set1_pd(0.0);
	#endif

	#if X86_SIMD_LEVEL >= LV_AVX2
		static const __m256i SIMD256B_zeroMask = _mm256_set1_epi8(0);
	#endif

	#if X86_SIMD_LEVEL >= LV_AVX
		static const __m256 SIMD256F_zeroMask = _mm256_set1_ps(0.0f);
		static const __m256d SIMD256D_zeroMask = _mm256_set1_pd(0.0);
		static const __m256 SIMD256_fastFloatNegateMask = _mm256_set1_ps(-0.0f);
		static const __m256d SIMD256D_fastDoubleNegateMask = _mm256_set1_pd(-0.0);
	#endif

	#if X86_SIMD_LEVEL >= LV_SSE2
		static const __m128d SIMD128D_fastDoubleNegateMask = _mm_set1_pd(-0.0);
		static const __m128d SIMD128D_zeroMask = _mm_set1_pd(0.0);
		static const __m128i SIMD128B_zeroMask = _mm_set1_epi8(0);
		inline const __m128i SIMD128B_byteReverseMask = _mm_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
		inline const __m128i SIMD128B_16bitReverseMask = _mm_set_epi8(1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14);
		inline const int SIMD128B_wordReverseMask = 177;
		inline constexpr int SIMD128_64bitReverseMask = 0x1;
	#endif

	#if X86_SIMD_LEVEL >= LV_SSE
		static const __m128 SIMD128_fastFloatNegateMask = _mm_set1_ps(-0.0f);
		inline constexpr int SIMD128_32bitReverseMask = 0x1B;
		static const __m128 SIMD128F_zeroMask = _mm_set1_ps(0.0f);
	#endif



	#if X86_SIMD_LEVEL >= LV_SSE
		/// Reverse order of single precision floating point elements in register
		inline __m128 reverse(const __m128& v) {
			return _mm_shuffle_ps(v, v, SIMD128_32bitReverseMask);
		}

		/// Negate all 32-bit elements in register
		inline __m128 negate(const __m128& v) {
			return _mm_xor_ps(v, SIMD128_fastFloatNegateMask);
		}
	#endif

	#if X86_SIMD_LEVEL >= LV_SSE2
		/// Reverse order of double precision floating point elements in register
		inline __m128d reverse(const __m128d& v) {
			return _mm_shuffle_pd(v, v, SIMD128_64bitReverseMask);
		}

		/// Reverse order of integer elements in register
		template<size_t IntegerBitSize>
		inline __m128i reverse(const __m128i& v) = delete;

		#if X86_SIMD_LEVEL >= LV_SSSE3
			/// Reverse order of 8-bit elements in register
			template<> inline __m128i reverse<8>(const __m128i& v) {
				return _mm_shuffle_epi8(v, SIMD128B_byteReverseMask);
			}
		#else
			/// Reverse order of 8-bit elements in register <br>
			/// Emulated using SSE2
			template<> inline __m128i reverse<8>(const __m128i& v) {
				alignas(16) uint8_t buf[16];
				_mm_store_si128(reinterpret_cast<__m128i*>(buf), v);
				std::reverse(buf, buf+16);
				return _mm_load_si128(reinterpret_cast<const __m128i*>(buf));
			}
		#endif

		#if X86_SIMD_LEVEL >= LV_SSSE3
			/// Reverse order of 16-bit elements in register <br>
			/// Specialized for presence of SSSE3's per-byte moves to avoid two instructions
			template<> inline __m128i reverse<16>(const __m128i& v) {
				return _mm_shuffle_epi8(v, SIMD128B_16bitReverseMask);
			}
		#else
			/// Reverse order of 16-bit elements in register
			template<> inline __m128i reverse<16>(const __m128i& v) {
				return _mm_shuffle_epi32(_mm_shufflehi_epi16(_mm_shufflelo_epi16(v, SIMD128B_wordReverseMask), SIMD128B_wordReverseMask), SIMD128_32bitReverseMask);
			}
		#endif

		/// Reverse order of 32-bit elements in register
		template<> inline __m128i reverse<32>(const __m128i& v) {
			return _mm_shuffle_epi32(v, SIMD128_32bitReverseMask);
		}

		/// Reverse order of 64-bit elements in register
		template<> inline __m128i reverse<64>(const __m128i& v) {
			return reinterpret_cast<__m128i>(_mm_shuffle_pd(reinterpret_cast<__m128d>(v), reinterpret_cast<__m128d>(v), SIMD128_64bitReverseMask));
		}

		/// Negate all 64-bit elements in register
		inline __m128d negate(const __m128d& v) {
			return _mm_xor_pd(v, SIMD128D_fastDoubleNegateMask);
		}
	#endif

	// AVX

	#if X86_SIMD_LEVEL >= LV_AVX

		/// Swap 128-bit lanes in register
		inline __m256 reverseLanes(const __m256& v) {
			return _mm256_permute2f128_ps(v, v, 1);
		}

		/// Swap 128-bit lanes in register
		inline __m256d reverseLanes(const __m256d& v) {
			return _mm256_permute2f128_pd(v, v, 1);
		}

		/// Negate all 32-bit elements in register
		inline __m256 negate(const __m256& v) {
			return _mm256_xor_ps(v, SIMD256_fastFloatNegateMask);
		}

		/// Negate all 64-bit elements in register
		inline __m256d negate(const __m256d& v) {
			return _mm256_xor_pd(v, SIMD256D_fastDoubleNegateMask);
		}
	#endif

	#if X86_SIMD_LEVEL >= LV_AVX2
		/// Swap 128-bit lanes in register
		inline __m256i reverseLanes(const __m256i& v) {
			return _mm256_permute2x128_si256(v, v, 1);
		}
	#endif

#endif
} // namespace simd
} // namespace wlib
