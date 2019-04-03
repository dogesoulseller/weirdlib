#include "../include/weirdlib.hpp"
#include "common.hpp"
#include <algorithm>
#include <cstring>

namespace wlib
{
	size_t strlen(const char* s) noexcept {
		size_t offset = 0;
		#if defined(AVX512_BW)
			if (reinterpret_cast<const size_t>(s) % 64 == 0) {
				// Move through bytes with a 64-byte stride
				while (true) {
					const __m512i chars = _mm512_load_si512(s + offset);

					int64_t mask = _mm512_cmpeq_epi8_mask(chars, SIMD512B_zeroMask);
					if (mask == 0) {
						offset += 64;
						continue;
					} else {
						return offset + static_cast<size_t>(bop::ctz(mask));
					}
				}
			} else {
				// Move through bytes with a 64-byte stride
				while (true) {
					const __m512i chars = _mm512_loadu_si512(s + offset);

					int64_t mask = _mm512_cmpeq_epi8_mask(chars, SIMD512B_zeroMask);
					if (mask == 0) {
						offset += 64;
						continue;
					} else {
						return offset + static_cast<size_t>(bop::ctz(mask));
					}
				}
			}
		#elif X86_SIMD_LEVEL >= 8 	// AVX2
			// Move through bytes with a 32-byte stride
			if (reinterpret_cast<const size_t>(s) % 32 == 0) {
				while (true) {
					const __m256i chars = _mm256_load_si256(reinterpret_cast<const __m256i*>(s + offset));

					const int32_t mask = _mm256_movemask_epi8(_mm256_cmpeq_epi8(chars, SIMD256B_zeroMask));
					if (mask == 0) {
						offset += 32;
						continue;
					} else {
						_mm256_zeroupper();
						return offset + static_cast<size_t>(bop::ctz(mask));
					}
				}
			} else {
				while (true) {
					const __m256i chars = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(s + offset));

					const int32_t mask = _mm256_movemask_epi8(_mm256_cmpeq_epi8(chars, SIMD256B_zeroMask));
					if (mask == 0) {
						offset += 32;
						continue;
					} else {
						_mm256_zeroupper();
						return offset + static_cast<size_t>(bop::ctz(mask));
					}
				}
			}
		#elif X86_SIMD_LEVEL >= 2 // SSE2
			// Move through bytes with a 16-byte stride
			if (reinterpret_cast<const size_t>(s) % 16 == 0) {
				while (true) {
					const __m128i chars = _mm_load_si128(reinterpret_cast<const __m128i*>(s + offset));

					// If no `\0` was found, check further
					const int32_t mask = _mm_movemask_epi8(_mm_cmpeq_epi8(chars, SIMD128B_zeroMask));
					if (mask == 0) {
						offset += 16;
						continue;
					} else {
						// If a `\0` was found, get the position through a bitscan
						return offset + static_cast<size_t>(bop::ctz(mask));
					}
				}
			} else {
				while (true) {
					const __m128i chars = _mm_loadu_si128(reinterpret_cast<const __m128i*>(s + offset));

					// If no `\0` was found, check further
					const int32_t mask = _mm_movemask_epi8(_mm_cmpeq_epi8(chars, SIMD128B_zeroMask));
					if (mask == 0) {
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

	bool strcmp(const std::string& str0, const std::string& str1) {
		// If both strings are empty...
		if (str0.empty() && str1.empty()) {
			return true;
		}

		// If string lengths are not equal, the strings are different
		if (str0.size() != str1.size()) {
			return false;
		}

		return wlib::strcmp(str0, str1, str0.size());
	}

	// This is equal to strncmp
	bool strcmp(const std::string& str0, const std::string& str1, size_t len) {
		size_t totalOffset = 0;
		size_t iters = 0;

		// If length is 0, no reason to go further
        if (len == 0) {
            return true;
        }

		// If both strings are empty...
		if (str0.empty() && str1.empty()) {
			return true;
		}

		// Clamp length to minimum string length between compared strings
		len = std::min(std::min(str0.size(), str1.size()), len);

		#if defined(AVX512_BW)

		if (len >= 64) {
			iters = len / 64;

            // SIMD what is possible
            for (size_t i = 0; i < iters; i++) {
                const __m512i LS = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(str0.c_str() + (i * 64)));
                const __m512i RS = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(str1.c_str() + (i * 64)));

                const uint64_t eqMask = _mm512_cmpeq_epi8_mask(LS, RS);

                // Mask from bytes is in the first 64 bits
                if ((eqMask & 0xFFFFFFFFFFFFFFFF) != 0xFFFFFFFFFFFFFFFF) {
                    _mm256_zeroupper();
                    return false;
                }

            }
			len -= 64 * iters;
			totalOffset += 64 * iters;
            _mm256_zeroupper();
		}

		#endif

		#if X86_SIMD_LEVEL >= 8 // AVX2

		if (len >= 32) {
			iters = len / 32;

            // SIMD what is possible
            for (size_t i = 0; i < iters; i++) {
                const __m256i LS = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(str0.c_str() + (i * 32)));
                const __m256i RS = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(str1.c_str() + (i * 32)));

                const uint32_t eqMask = static_cast<uint32_t>(_mm256_movemask_epi8(_mm256_cmpeq_epi8(LS, RS)));

                // Mask from bytes is in the first 32 bits
                if ((eqMask & 0xFFFFFFFF) != 0xFFFFFFFF) {
                    _mm256_zeroupper();
                    return false;
                }

            }
			len -= 32 * iters;
			totalOffset += 32 * iters;

            _mm256_zeroupper();
		}

		#endif

		#if X86_SIMD_LEVEL >= 2 // SSE2

		if (len >= 16) {
			iters = len / 16;

            // SIMD what is possible
            for (size_t i = 0; i < iters; i++) {
                const __m128i LS = _mm_loadu_si128(reinterpret_cast<const __m128i*>(str0.c_str() + (i * 16)));
                const __m128i RS = _mm_loadu_si128(reinterpret_cast<const __m128i*>(str1.c_str() + (i * 16)));

                const uint32_t eqMask = static_cast<uint32_t>(_mm_movemask_epi8(_mm_cmpeq_epi8(LS, RS)));

                // Mask from bytes is in the first 16 bits
                if ((eqMask & 0xFFFF) != 0xFFFF) {
                    return false;
                }

            }
			len -= 16 * iters;
			totalOffset += 16 * iters;
		}

		#endif // Nothin'

		return ::strncmp(str0.c_str() + totalOffset, str1.c_str() + totalOffset, len) == 0;
	}
} // wlib
