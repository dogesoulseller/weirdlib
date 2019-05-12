#pragma once

/**
 *	Check CPU feature set available in target platform and define it in a simpler way
 *	Currently supports x86 and x86_64 extensions
**/

// Unify x86 variant detection
#if (defined(_M_AMD64) || defined(_M_X64) || defined(__amd64) || defined(__amd64__) || defined(__x86_64)) && !defined(__x86_64__)
	#define __x86_64__
#elif (defined(i386) || defined(__i386) || defined(__i386__) || defined(_M_I86) || defined(_M_IX86) || defined(_X86_) || defined(__THW_INTEL__) || defined(__I86__) || defined(__INTEL__) || defined(__386)) && !defined(__X86__)
	#define __X86__
#endif
// Include intrinsics for x86 targets
#if defined(__x86_64__) || defined(__X86__)
	#include <immintrin.h>
#endif

// Detect x86 SIMD capabilities
#if defined(__AVX512F__)	// AVX512 is more complex and requires further detection
	#define X86_SIMD_LEVEL 9
	// Byte and Word
	#if defined(__AVX512BW__)
		#define AVX512_BW
	#endif

	// DWord and QWord
	#if defined(__AVX512DQ__)
		#define AVX512_DQ
	#endif

	// Vector Length
	#if defined(__AVX512VL__)
		#define AVX512_VL
	#endif
#elif defined(__AVX2__)
	#define X86_SIMD_LEVEL 8
#elif defined(__AVX__)
	#define X86_SIMD_LEVEL 7
#elif defined(__SSE4_2__)
	#define X86_SIMD_LEVEL 6
#elif defined(__SSE4_1__)
	#define X86_SIMD_LEVEL 5
#elif defined(__SSSE3__)
	#define X86_SIMD_LEVEL 4
#elif defined(__SSE3__)
	#define X86_SIMD_LEVEL 3
#elif defined(__SSE2__) || defined(__x86_64__)	// x64 guarantees at least SSE2 presence
	#define X86_SIMD_LEVEL 2
#elif defined(__SSE__)
	#define X86_SIMD_LEVEL 1
#else
	#define X86_SIMD_LEVEL 0
#endif

// Detect FMA presence separately (because AMD)
#ifdef __FMA__
	#define X86_SIMD_FMA
#endif
