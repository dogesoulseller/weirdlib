#pragma once
#include "cpu_detection.hpp"

#include <type_traits>
#include <cstdint>

#if defined(AVX512_BW)
	const __m512i SIMD512B_zeroMask = _mm512_set1_epi8(0);
#elif defined(AVX512_DQ)
	const __m512i SIMD512B_zeroMask = _mm512_set1_epi32(0);
#endif

#if X86_SIMD_LEVEL >= 9	// AVX512F
	const __m512 SIMD512F_zeroMask = _mm512_set1_ps(0.0f);
	const __m512d SIMD512D_zeroMask = _mm512_set1_pd(0.0);
#endif

#if X86_SIMD_LEVEL >= 8 // AVX2
	const __m256i SIMD256B_zeroMask = _mm256_set1_epi8(0);
#endif

#if X86_SIMD_LEVEL >= 7	// AVX
	const __m256 SIMD256F_zeroMask = _mm256_set1_ps(0.0f);
	const __m256d SIMD256D_zeroMask = _mm256_set1_pd(0.0);
#endif

#if X86_SIMD_LEVEL >= 2 // SSE2
	const __m128i SIMD128B_zeroMask = _mm_set1_epi8(0);
#endif

#if X86_SIMD_LEVEL >= 1 // SSE
	const __m128 SIMD128F_zeroMask = _mm_set1_ps(0.0f);
	const __m128d SIMD128D_zeroMask = _mm_set1_pd(0.0);
#endif
