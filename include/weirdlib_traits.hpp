#pragma once
#include <type_traits>
#include <utility>
#include <stdexcept>
#include <string>

namespace wlib
{

/// SFINAE traits (mostly for internal use)
namespace traits
{
	template<typename T, typename AttemptLogical = void, typename AttemptShift = void>
	struct has_bitops : std::false_type{};

	/// Is valid for any type that implements bitwise operators
	/// @see has_bitops_v
	template<typename T>
	 struct has_bitops<T,
	 	std::void_t<decltype(((~std::declval<T>() | std::declval<T>()) & std::declval<T>()) ^ std::declval<T>())>,
		std::void_t<decltype(std::declval<T>() << std::declval<T>() >> std::declval<T>())>> : std::true_type{};

	template<typename T>
	constexpr inline bool has_bitops_v = has_bitops<T>::value;


	template<typename T, typename AttemptDerives = void>
	struct is_string_based : std::false_type{};

	/// Is valid for any type that can be accessed as a null-terminated C string
	/// @see is_string_based_v
	template<typename T>
	struct is_string_based<T, std::void_t<decltype(std::declval<T>().c_str())>> : std::true_type{};

	template<typename T>
	constexpr inline bool is_string_based_v = is_string_based<T>::value;




} // traits

	class module_not_built : public std::runtime_error
	{
		public:
		inline module_not_built(const char* m = "") : std::runtime_error(m) {}
	};
} // wlib
