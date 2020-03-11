#include "../../include/weirdlib_string.hpp"
#include "../../include/cpu_detection.hpp"
#include "../../include/weirdlib_bitops.hpp"

#include <vector>
#include <cstring>
#include <algorithm>
#include <numeric>
#include <functional>

namespace wlib::str
{
	using namespace std::placeholders;

	const char* strpbrk(const std::string& str, const char* needles, size_t strLen, size_t needlesLen) {
		strLen = strLen == 0 ? str.size() : strLen;
		needlesLen = needlesLen == 0 ? wlib::str::strlen(needles) : needlesLen;

		return wlib::str::strpbrk(str.c_str(), needles, strLen, needlesLen);
	}

	char* strpbrk(char* str, const char* needles, size_t strLen, size_t needlesLen) {
		strLen = strLen == 0 ? wlib::str::strlen(str) : strLen;
		needlesLen = needlesLen == 0 ? wlib::str::strlen(needles) : needlesLen;

		#if X86_SIMD_LEVEL >= LV_SSE
			std::vector<__m128i> searchMasks(needlesLen);
			std::generate(searchMasks.begin(), searchMasks.end(), [i=0, &needles] () mutable {
				return _mm_set1_epi8(needles[i++]);
			});

			size_t iters = strLen / 16;
			size_t itersRem = strLen % 16;
			size_t offset = 0;

			for (size_t i = 0; i < iters; i++) {
				auto searchVec = _mm_loadu_si128(reinterpret_cast<const __m128i*>(str+offset));

				__m128i mask = _mm_set1_epi8(0);
				for (const auto& m : searchMasks) {
					mask = _mm_or_si128(mask, _mm_cmpeq_epi8(searchVec, m));
				}

				auto matchMask = _mm_movemask_epi8(mask);
				if (matchMask == 0) {
					offset += 16;
				} else {
					return str+(offset + static_cast<size_t>(wlib::bop::ctz(matchMask)));
				}
			}

			if (itersRem != 0) {
				return std::strpbrk(str+offset, needles);
			} else {
				return nullptr;
			}
		#else
			return std::strpbrk(str, needles);
		#endif
	}

	const char* strpbrk(const char* str, const char* needles, size_t strLen, size_t needlesLen) {
		return const_cast<const char*>(wlib::str::strpbrk(const_cast<char*>(str), needles, strLen, needlesLen));
	}

} // namespace wlib::str
