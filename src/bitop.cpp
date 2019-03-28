#include "../include/weirdlib.hpp"

namespace wlib
{
namespace bop
{
	#ifdef __GNUC__
	template<> uint32_t count_trailing_zeros<uint32_t>(uint32_t x) noexcept {
		return __builtin_ctz(x);
	}

	template<> uint32_t count_leading_zeros<uint32_t>(uint32_t x) noexcept {
		return __builtin_clz(x);
	}

	template<> uint32_t count_trailing_zeros<int32_t>(int32_t x) noexcept {
		return __builtin_ctz(x);
	}

	template<> uint32_t count_leading_zeros<int32_t>(int32_t x) noexcept {
		return __builtin_clz(x);
	}


	template<> uint32_t count_trailing_zeros<uint64_t>(uint64_t x) noexcept {
		return __builtin_ctzll(x);
	}

	template<> uint32_t count_leading_zeros<uint64_t>(uint64_t x) noexcept {
		return __builtin_clzll(x);
	}

	template<> uint32_t count_trailing_zeros<int64_t>(int64_t x) noexcept {
		return __builtin_ctzll(x);
	}

	template<> uint32_t count_leading_zeros<int64_t>(int64_t x) noexcept {
		return __builtin_clzll(x);
	}
	#endif

} // bop

} // wlib
