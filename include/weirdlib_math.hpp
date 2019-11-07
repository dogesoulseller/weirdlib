#pragma once
#include <type_traits>
#include <cmath>
#include <numeric>
#include <limits>
#include <utility>
#include <iterator>

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

	template<typename T>
	constexpr T max(T&& v) {
		return std::forward<T>(v);
	}

	template<typename T0, typename T1, typename... Args>
	constexpr std::common_type_t<T0, T1, Args...> max(T0&& v0, T1&& v1, Args... args) {
		return v1 > v0
		? max(v1, std::forward<Args>(args)...)
		: max(v0, std::forward<Args>(args)...);
	}

	template<typename T>
	constexpr T min(T&& v) {
		return std::forward<T>(v);
	}

	template<typename T0, typename T1, typename... Args>
	constexpr std::common_type_t<T0, T1, Args...> min(T0&& v0, T1&& v1, Args... args) {
		return v1 < v0
		? min(v1, std::forward<Args>(args)...)
		: min(v0, std::forward<Args>(args)...);
	}

	// Specialized for all integral
	template<typename T0, typename T1, typename... Args> constexpr
	std::enable_if_t<std::is_integral_v<T0> && std::is_integral_v<T1>
	&& (std::is_integral_v<Args> && ...),
	std::common_type_t<T0, T1, Args...>>
	average(T0 x, T1 y, Args... rest) {
		size_t tmp = x + y + (rest + ...);
		return tmp / (sizeof...(rest) + 2);
	}

	// Average with remainder
	template<typename T0, typename T1, typename... Args> constexpr
	std::enable_if_t<std::is_integral_v<T0> && std::is_integral_v<T1>
	&& (std::is_integral_v<Args> && ...),
	std::pair<std::common_type_t<T0, T1, Args...>, std::common_type_t<T0, T1, Args...>>>
	average_and_remainder(T0 x, T1 y, Args... rest) {
		size_t tmp = x + y + (rest + ...);
		return std::make_pair(tmp / (sizeof...(rest) + 2), tmp % (sizeof...(rest) + 2));
	}

	// Specialized for all floating
	template<typename T0, typename T1, typename... Args> constexpr
	std::enable_if_t<std::is_floating_point_v<T0> && std::is_floating_point_v<T1>
	&& (std::is_floating_point_v<Args> && ...),
	std::common_type_t<T0, T1, Args...>>
	average(T0 x, T1 y, Args... rest) {
		auto tmp = x + y + (rest + ...);
		return tmp / static_cast<std::common_type_t<T0, T1, Args...>>((sizeof...(rest) + 2));
	}

	// Specialized for mixed
	template<typename T0, typename T1, typename... Args> constexpr
	std::enable_if_t<(!(std::is_floating_point_v<T0> && std::is_floating_point_v<T1>
	&& (std::is_floating_point_v<Args> && ...))) && (!(std::is_integral_v<T0> && std::is_integral_v<T1>
	&& (std::is_integral_v<Args> && ...))), std::common_type_t<T0, T1, Args...>>
	average(T0 x, T1 y, Args... rest) {
		auto tmp = x + y + (rest + ...);
		return tmp / static_cast<std::common_type_t<T0, T1, Args...>>((sizeof...(rest) + 2));
	}

	// Specialized for iterators
	template<typename IterT> constexpr
	typename std::iterator_traits<IterT>::value_type average(const IterT start, const IterT end) {
		auto out = typename std::iterator_traits<IterT>::value_type {};
		for (auto i = start; i != end; i++) {
			out += (*i);
		}

		return out / std::distance(start, end);
	}

	// Specialized for iterators
	template<typename IterT> constexpr
	std::enable_if_t<std::is_integral_v<typename std::iterator_traits<IterT>::value_type>,
	std::pair<typename std::iterator_traits<IterT>::value_type, size_t>> average_and_remainder(const IterT start, const IterT end) {
		auto out = typename std::iterator_traits<IterT>::value_type {};
		for (auto i = start; i != end; i++) {
			out += (*i);
		}
		const auto distance = std::distance(start, end);
		return std::make_pair(out / distance, out % distance);
	}

} // namespace math
} // namespace wlib
