#ifdef WEIRDLIB_ENABLE_STRING_OPERATIONS
#include "../../include/weirdlib_string.hpp"
#include "../../include/weirdlib_bitops.hpp"
#include "../../include/cpu_detection.hpp"
#include <algorithm>

namespace wlib::str
{
	const char* strstr(const char* str, const char* needle, size_t strLen, size_t needleLen) {
		if (strLen == 0) {
			strLen = wlib::str::strlen(str);
		}

		if (needleLen == 0) {
			needleLen = wlib::str::strlen(needle);
		}

		if (needleLen == 0) {
			return str;
		}

		#if defined(AVX512_BW)
			// Broadcast first and last character of substring to registers
			const auto start = _mm512_set1_epi8(needle[0]);
			const auto end = _mm512_set1_epi8(needle[needleLen-1]);

			// Process 32 bytes of string at once
			for (size_t i = 0; i < strLen; i+=64) {
				const auto first = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(str + i));
				const auto last = _mm512_loadu_si512(reinterpret_cast<const __m512i*>((str + i) + (needleLen - 1)));

				// Calculate mask from hits and misses
				auto mask = static_cast<uint64_t>(_mm512_movepi8_mask(_mm256_and_si256(_mm256_cmpeq_epi8(start, first), _mm256_cmpeq_epi8(end, last))));
				while (mask != 0) {
					// Get lowest set bit
					const auto pos = bop::bit_scan_reverse(mask);

					// Check for equality between str and needle search position
					const auto eq0 = str+i+pos+1;
					const auto eq1 = needle+1;
					if (std::equal(eq0, eq0+needleLen-2, eq1)) {
						return str + i + pos;
					} else {
						mask = bop::clear_leftmost_set(mask);
					}
				}
			}

			// If nothing was found, return null
			return nullptr;
		#elif X86_SIMD_LEVEL >= LV_AVX2
			// Broadcast first and last character of substring to registers
			const auto start = _mm256_set1_epi8(needle[0]);
			const auto end = _mm256_set1_epi8(needle[needleLen-1]);

			// Process 32 bytes of string at once
			for (size_t i = 0; i < strLen; i+=32) {
				const auto first = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(str + i));
				const auto last = _mm256_loadu_si256(reinterpret_cast<const __m256i*>((str + i) + (needleLen - 1)));

				// Calculate mask from hits and misses
				auto mask = _mm256_movemask_epi8(_mm256_and_si256(_mm256_cmpeq_epi8(start, first), _mm256_cmpeq_epi8(end, last)));
				while (mask != 0) {
					// Get lowest set bit
					const auto pos = bop::bit_scan_reverse(mask);

					// Check for equality between str and needle search position
					const auto eq0 = str+i+pos+1;
					const auto eq1 = needle+1;
					if (std::equal(eq0, eq0+needleLen-2, eq1)) {
						return str + i + pos;
					} else {
						mask = bop::clear_leftmost_set(mask);
					}
				}
			}

			_mm256_zeroupper();

			// If nothing was found, return null
			return nullptr;
		#elif X86_SIMD_LEVEL >= LV_SSE2
			// Broadcast first and last character of substring to registers
			const auto start = _mm_set1_epi8(needle[0]);
			const auto end = _mm_set1_epi8(needle[needleLen-1]);

			// Process 16 bytes of string at once
			for (size_t i = 0; i < strLen; i+=16) {
				const auto first = _mm_loadu_si128(reinterpret_cast<const __m128i*>(str + i));
				const auto last = _mm_loadu_si128(reinterpret_cast<const __m128i*>((str + i) + (needleLen - 1)));

				// Calculate mask from hits and misses
				auto mask = _mm_movemask_epi8(_mm_and_si128(_mm_cmpeq_epi8(start, first), _mm_cmpeq_epi8(end, last)));
				while (mask != 0) {
					// Get lowest set bit
					const auto pos = bop::bit_scan_reverse(mask);

					// Check for equality between str and needle search position
					const auto eq0 = str+i+pos+1;
					const auto eq1 = needle+1;
					if (std::equal(eq0, eq0+needleLen-2, eq1)) {
						return str + i + pos;
					} else {
						mask = bop::clear_leftmost_set(mask);
					}
				}
			}

			// If nothing was found, return null
			return nullptr;
		#else
			return std::strstr(str, needle);
		#endif
	}

	char* strstr(char* str, const char* needle, size_t strLen, size_t needleLen) {
		if (strLen == 0) {
			strLen = wlib::str::strlen(str);
		}

		if (needleLen == 0) {
			needleLen = wlib::str::strlen(needle);
		}

		if (needleLen == 0) {
			return str;
		}

		#if defined(AVX512_BW)
			// Broadcast first and last character of substring to registers
			const auto start = _mm512_set1_epi8(needle[0]);
			const auto end = _mm512_set1_epi8(needle[needleLen-1]);

			// Process 32 bytes of string at once
			for (size_t i = 0; i < strLen; i+=64) {
				const auto first = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(str + i));
				const auto last = _mm512_loadu_si512(reinterpret_cast<const __m512i*>((str + i) + (needleLen - 1)));

				// Calculate mask from hits and misses
				auto mask = static_cast<uint64_t>(_mm512_movepi8_mask(_mm256_and_si256(_mm256_cmpeq_epi8(start, first), _mm256_cmpeq_epi8(end, last))));
				while (mask != 0) {
					// Get lowest set bit
					const auto pos = bop::bit_scan_reverse(mask);

					// Check for equality between str and needle search position
					const auto eq0 = str+i+pos+1;
					const auto eq1 = needle+1;
					if (std::equal(eq0, eq0+needleLen-2, eq1)) {
						return str + i + pos;
					} else {
						mask = bop::clear_leftmost_set(mask);
					}
				}
			}

			// If nothing was found, return null
			return nullptr;
		#elif X86_SIMD_LEVEL >= LV_AVX2
			// Broadcast first and last character of substring to registers
			const auto start = _mm256_set1_epi8(needle[0]);
			const auto end = _mm256_set1_epi8(needle[needleLen-1]);

			// Process 32 bytes of string at once
			for (size_t i = 0; i < strLen; i+=32) {
				const auto first = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(str + i));
				const auto last = _mm256_loadu_si256(reinterpret_cast<const __m256i*>((str + i) + (needleLen - 1)));

				// Calculate mask from hits and misses
				auto mask = _mm256_movemask_epi8(_mm256_and_si256(_mm256_cmpeq_epi8(start, first), _mm256_cmpeq_epi8(end, last)));
				while (mask != 0) {
					// Get lowest set bit
					const auto pos = bop::bit_scan_reverse(mask);

					// Check for equality between str and needle search position
					const auto eq0 = str+i+pos+1;
					const auto eq1 = needle+1;
					if (std::equal(eq0, eq0+needleLen-2, eq1)) {
						return str + i + pos;
					} else {
						mask = bop::clear_leftmost_set(mask);
					}
				}
			}

			_mm256_zeroupper();

			// If nothing was found, return null
			return nullptr;
		#elif X86_SIMD_LEVEL >= LV_SSE2
			// Broadcast first and last character of substring to registers
			const auto start = _mm_set1_epi8(needle[0]);
			const auto end = _mm_set1_epi8(needle[needleLen-1]);

			// Process 32 bytes of string at once
			for (size_t i = 0; i < strLen; i+=32) {
				const auto first = _mm_loadu_si128(reinterpret_cast<const __m128i*>(str + i));
				const auto last = _mm_loadu_si128(reinterpret_cast<const __m128i*>((str + i) + (needleLen - 1)));

				// Calculate mask from hits and misses
				auto mask = _mm_movemask_epi8(_mm_and_si128(_mm_cmpeq_epi8(start, first), _mm_cmpeq_epi8(end, last)));
				while (mask != 0) {
					// Get lowest set bit
					const auto pos = bop::bit_scan_reverse(mask);

					// Check for equality between str and needle search position
					const auto eq0 = str+i+pos+1;
					const auto eq1 = needle+1;
					if (std::equal(eq0, eq0+needleLen-2, eq1)) {
						return str + i + pos;
					} else {
						mask = bop::clear_leftmost_set(mask);
					}
				}
			}

			// If nothing was found, return null
			return nullptr;
		#else
			return std::strstr(str, needle);
		#endif
	}

	const char* strstr(const std::string& str, const std::string& needle, size_t strLen, size_t needleLen) {
		strLen = strLen == 0 ? str.size() : strLen;
		needleLen = needleLen == 0 ? needle.size() : needleLen;
		return strstr(str.c_str(), needle.c_str(), strLen, needleLen);
	}

} // namespace wlib::str
#endif // WEIRDLIB_ENABLE_STRING_OPERATIONS
