#pragma once
#include <type_traits>
#include <utility>

namespace wlib
{
namespace traits
{

	template<typename T, typename Attempt = void>
	struct has_bitops : std::false_type{};

	/// Is valid for a type that implements bitwise operators
	/// @see has_bitops_v
	template<typename T>
	struct has_bitops<T, std::void_t<decltype(std::declval<T>() & 1,
		std::declval<T>() | 1,
		std::declval<T>() ^ 1,
		~std::declval<T>(),
		std::declval<T>() << 1,
		std::declval<T>() >> 1)>> : std::true_type{};

	template<typename T>
	constexpr inline bool has_bitops_v = has_bitops<T>::value;
} // traits
} // wlib
