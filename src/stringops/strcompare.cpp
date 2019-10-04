#ifdef WEIRDLIB_ENABLE_STRING_OPERATIONS
#include "../../include/weirdlib.hpp"
#include "../common.hpp"
#include <algorithm>
#include <cstring>

namespace wlib::str
{
	bool strcmp(const char* str0, const char* str1) {
		const size_t str0Size = wlib::str::strlen(str0);
		const size_t str1Size = wlib::str::strlen(str1);

		// If both strings are empty...
		if (str0Size == 0 && str1Size == 0) {
			return true;
		}

		// If string lengths are not equal, the strings are different
		if (str0Size != str1Size) {
			return false;
		}

		return wlib::str::strcmp(str0, str1, str0Size);
	}

	// This is equal to strncmp
	bool strcmp(const char* str0, const char* str1, size_t len) {
		size_t totalOffset = 0;
		size_t iters = 0;

		// If length is 0, no reason to go further
        if (len == 0) {
            return true;
        }

		const size_t str0Size = strlen(str0);
		const size_t str1Size = strlen(str1);

		// If both strings are empty...
		if (str0Size == 0 && str1Size == 0) {
			return true;
		}

		// Clamp length to minimum string length between compared strings
		len = std::min(std::min(str0Size, str1Size), len);

		#if defined(AVX512_BW)

		if (len >= 64) {
			iters = len / 64;

            // SIMD what is possible
            for (size_t i = 0; i < iters; i++) {
                const auto LS = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(str0 + (i * 64)));
                const auto RS = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(str1 + (i * 64)));

                const auto eqMask = _mm512_cmpeq_epi8_mask(LS, RS);

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

		#if X86_SIMD_LEVEL >= LV_AVX2

		if (len >= 32) {
			iters = len / 32;

            // SIMD what is possible
            for (size_t i = 0; i < iters; i++) {
                const auto LS = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(str0 + (i * 32)));
                const auto RS = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(str1 + (i * 32)));

                const auto eqMask = static_cast<uint32_t>(_mm256_movemask_epi8(_mm256_cmpeq_epi8(LS, RS)));

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

		#if X86_SIMD_LEVEL >= LV_SSE2

		if (len >= 16) {
			iters = len / 16;

            // SIMD what is possible
            for (size_t i = 0; i < iters; i++) {
                const auto LS = _mm_loadu_si128(reinterpret_cast<const __m128i*>(str0 + (i * 16)));
                const auto RS = _mm_loadu_si128(reinterpret_cast<const __m128i*>(str1 + (i * 16)));

                const auto eqMask = static_cast<uint32_t>(_mm_movemask_epi8(_mm_cmpeq_epi8(LS, RS)));

                // Mask from bytes is in the first 16 bits
                if ((eqMask & 0xFFFF) != 0xFFFF) {
                    return false;
                }

            }
			len -= 16 * iters;
			totalOffset += 16 * iters;
		}

		#endif // Nothin'

		return ::strncmp(str0 + totalOffset, str1 + totalOffset, len) == 0;
	}

	// strncmp is aliased to strcmp overloaded for a max len parameter
	bool strncmp(const char* str0, const char* str1, size_t len) {
		return wlib::str::strcmp(str0, str1, len);
	}

} // namespace wlib::str
#endif // WEIRDLIB_ENABLE_STRING_OPERATIONS
