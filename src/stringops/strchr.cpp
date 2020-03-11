#include "../../include/weirdlib_string.hpp"
#include "../../include/cpu_detection.hpp"
#include "../../include/weirdlib_bitops.hpp"

#include <cstring>

namespace wlib::str
{
	const char* strchr(const std::string& str, char search, size_t strLen) {
		strLen = strLen == 0 ? str.size() : strLen;
		return strchr(str.c_str(), search, strLen);
	}

	const char* strchr(const char* s, char search, size_t strLen) {
		size_t offset = 0;
		strLen = strLen == 0 ? wlib::str::strlen(s) : strLen;

		#if X86_SIMD_LEVEL >= LV_SSE
			size_t iters = strLen / 16;
			size_t itersRem = strLen % 16;
			const auto searchMask = _mm_set1_epi8(search);

			for (size_t i = 0; i < iters; i++) {
				auto inVec = _mm_loadu_si128(reinterpret_cast<const __m128i*>(s+offset));

				auto matchMask = _mm_movemask_epi8(_mm_cmpeq_epi8(inVec, searchMask));
				if (matchMask == 0) {
					offset += 16;
				} else {
					return s+(offset + static_cast<size_t>(wlib::bop::ctz(matchMask)));
				}
			}

			if (itersRem != 0) {
				return std::strchr(s+offset, search);
			} else {
				return nullptr;
			}
		#else
			return std::strchr(s, search);
		#endif
	}

	char* strchr(char* str, char search, size_t strLen) {
		return const_cast<char*>(wlib::str::strchr(const_cast<const char*>(str), search, strLen));
	}
} // namespace wlib::str
