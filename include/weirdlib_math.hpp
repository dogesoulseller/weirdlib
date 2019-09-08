#pragma once
#include <type_traits>
#include <cmath>
#include <numeric>
#include <limits>

namespace wlib
{
namespace math
{
	template<typename T, typename MulT, typename = std::enable_if_t<std::is_integral_v<T> && std::is_integral_v<MulT>>>
	auto next_multiple(const T n, const MulT multiple) {
		if (multiple == 0)
       		return n;

		const auto remainder = abs(n) % multiple;
		if (remainder == 0)
			return n;

		if (n < 0)
			return -(abs(n) - remainder);
		else
			return n + multiple - remainder;
	}

	template<typename T, typename MulT, typename = std::enable_if_t<std::is_integral_v<T> && std::is_integral_v<MulT>>>
	auto previous_multiple(const T n, const MulT multiple) {
		if (multiple == 0)
       		return n;

		const auto next_mul = next_multiple(n, multiple);
		if (next_mul == 0)
			return n;

		return next_mul - multiple == 0 ? next_mul : next_mul - multiple;
	}

	template<typename T, typename MulT, typename = std::enable_if_t<std::is_integral_v<T> && std::is_integral_v<MulT>>>
	auto nearest_multiple(const T n, const MulT multiple) {
		if (multiple == 0)
       		return n;

		const auto prev_mul = previous_multiple(n, multiple);
		const auto next_mul = next_multiple(n, multiple);

		if (std::abs(n - prev_mul) > std::abs(n - next_mul)) {
			return next_mul;
		} else {
			return prev_mul;
		}
	}

	template<typename TLHS, typename TRHS, typename = std::enable_if_t<std::is_floating_point_v<TLHS> && std::is_floating_point_v<TRHS>>>
	bool float_eq(const TLHS a, const TRHS b, const std::common_type_t<TLHS, TRHS> tolerance = 4.0) {
		using CommonT = std::common_type_t<TLHS, TRHS>;
		static_assert(std::is_floating_point_v<CommonT>, "Types for the float equality check are incompatible");

		return std::abs(a - b) <= std::min(std::abs(a), std::abs(b)) * std::numeric_limits<CommonT>::epsilon() * tolerance;
	}

} // namespace math
} // namespace wlib
