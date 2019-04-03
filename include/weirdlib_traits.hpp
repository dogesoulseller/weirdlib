#pragma once
#include <type_traits>
#include <utility>

namespace wlib
{
namespace traits
{
	template<typename...>
	using try_to_instantiate = void;

	template<typename T, typename Attempt = void>
	struct has_bitops : std::false_type{};

	template<typename T>
	struct has_bitops<T, try_to_instantiate<decltype(std::declval<T>() & 1,
		std::declval<T>() | 1,
		std::declval<T>() ^ 1,
		~std::declval<T>(),
		std::declval<T>() << 1)>> : std::true_type{};

	template<typename T>
	constexpr inline bool has_bitops_v = has_bitops<T>::value;
} // traits
} // wlib
