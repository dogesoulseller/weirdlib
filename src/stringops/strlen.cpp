#ifdef WEIRDLIB_ENABLE_STRING_OPERATIONS
#include "../../include/weirdlib.hpp"
#include "../common.hpp"
#include <algorithm>
#include <cstring>

namespace wlib::str
{
	size_t strlen(const char* s) {
		size_t offset = 0;
		#if defined(AVX512_BW)
			if (reinterpret_cast<const size_t>(s) % 64 == 0) {
				// Move through bytes with a 64-byte stride
				while (true) {
					const auto chars = _mm512_load_si512(s + offset);

					if (const auto mask = _mm512_cmpeq_epi8_mask(chars, simd::SIMD512B_zeroMask); mask == 0) {
						offset += 64;
						continue;
					} else {
						return offset + static_cast<size_t>(bop::ctz(mask));
					}
				}
			} else {
				// Move through bytes with a 64-byte stride
				while (true) {
					const auto chars = _mm512_loadu_si512(s + offset);

					if (const auto mask = _mm512_cmpeq_epi8_mask(chars, simd::SIMD512B_zeroMask); mask == 0) {
						offset += 64;
						continue;
					} else {
						return offset + static_cast<size_t>(bop::ctz(mask));
					}
				}
			}
		#elif X86_SIMD_LEVEL >= LV_AVX2
			// Move through bytes with a 32-byte stride
			if (reinterpret_cast<size_t>(s) % 32 == 0) {
				while (true) {
					const auto chars = _mm256_load_si256(reinterpret_cast<const __m256i*>(s + offset));

					if (const auto mask = _mm256_movemask_epi8(_mm256_cmpeq_epi8(chars, simd::SIMD256B_zeroMask)); mask == 0) {
						offset += 32;
						continue;
					} else {
						_mm256_zeroupper();
						return offset + static_cast<size_t>(bop::ctz(mask));
					}
				}
			} else {
				while (true) {
					const auto chars = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(s + offset));

					if (const auto mask = _mm256_movemask_epi8(_mm256_cmpeq_epi8(chars, simd::SIMD256B_zeroMask)); mask == 0) {
						offset += 32;
						continue;
					} else {
						_mm256_zeroupper();
						return offset + static_cast<size_t>(bop::ctz(mask));
					}
				}
			}
		#elif X86_SIMD_LEVEL >= LV_SSE2
			// Move through bytes with a 16-byte stride
			if (reinterpret_cast<const size_t>(s) % 16 == 0) {
				while (true) {
					const auto chars = _mm_load_si128(reinterpret_cast<const __m128i*>(s + offset));

					// If no `\0` was found, check further
					if (const auto mask = _mm_movemask_epi8(_mm_cmpeq_epi8(chars, simd::SIMD128B_zeroMask)); mask == 0) {
						offset += 16;
						continue;
					} else {
						// If a `\0` was found, get the position through a bitscan
						return offset + static_cast<size_t>(bop::ctz(mask));
					}
				}
			} else {
				while (true) {
					const auto chars = _mm_loadu_si128(reinterpret_cast<const __m128i*>(s + offset));

					// If no `\0` was found, check further
					if (const auto mask = _mm_movemask_epi8(_mm_cmpeq_epi8(chars, simd::SIMD128B_zeroMask)); mask == 0) {
						offset += 16;
						continue;
					} else {
						// If a `\0` was found, get the position through a bitscan
						return offset + static_cast<size_t>(bop::ctz(mask));
					}
				}
			}
		#else
			return ::strlen(s);
		#endif
	}

} // wlib::str

#endif // WEIRDLIB_ENABLE_STRING_OPERATIONS
