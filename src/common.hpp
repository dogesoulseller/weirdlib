#pragma once
#include "../include/weirdlib_simdhelper.hpp"

#include <type_traits>
#include <cstdint>
#include <charconv>

#define PP_CAT(a, b) PP_CAT_I(a, b)
#define PP_CAT_I(a, b) PP_CAT_II(~, a ## b)
#define PP_CAT_II(p, res) res

#define IGNORESB PP_CAT(PP_CAT(_unused_sb_param__, __LINE__), __COUNTER__)


/// Get optimal thread count for processing image of given size
/// @param width image width
/// @param height image height
/// @return thread count
int getImagePreferredThreadCount(int width, int height);

inline constexpr std::array<char, 10> digits = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'};
inline constexpr std::array<char, 11> digitsOrNeg = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-'};
inline constexpr std::array<char, 4> whitespace = {' ', '\t', '\n', '\r'};

// Get position of next digit
template<typename PtrT, typename IntT, typename = std::enable_if_t<std::is_pointer_v<PtrT> && std::is_integral_v<IntT>>>
inline static PtrT GetNextDigit(const PtrT ptr, IntT& offset) {
	PtrT out = std::find_first_of(ptr+offset, ptr+offset+128, digits.cbegin(), digits.cend());
	offset = out - ptr + 1;
	return out;
}

// Get position of next whitespace character
template<typename PtrT, typename IntT, typename = std::enable_if_t<std::is_pointer_v<PtrT> && std::is_integral_v<IntT>>>
inline static PtrT GetNextWhitespace(const PtrT ptr, IntT& offset) {
	PtrT out = std::find_first_of(ptr+offset, ptr+offset+128, whitespace.cbegin(), whitespace.cend());
	offset = out - ptr + 1;
	return out;
}

// Get position of next whitespace character without offset
template<typename PtrT, typename = std::enable_if_t<std::is_pointer_v<PtrT>>>
inline static PtrT GetNextWhitespace(const PtrT ptr) {
	PtrT out = std::find_first_of(ptr, ptr+128, whitespace.cbegin(), whitespace.cend());
	return out;
}

// Tuple of: <digitPos, whitespacePos, parsedNumber>
template<typename PtrT, typename IntT, typename = std::enable_if_t<std::is_pointer_v<PtrT> && std::is_integral_v<IntT>>>
using NumberParseResult = std::tuple<PtrT, PtrT, IntT>;

template<typename OutputT, typename PtrT, typename OffT>
inline static NumberParseResult<PtrT, OutputT> GetNextNumber(const PtrT startPtr, OffT& offset) {
	OutputT out;
	PtrT digitLocation = GetNextDigit(startPtr, offset);
	PtrT whitespaceLocation = GetNextWhitespace(startPtr, offset);

	std::from_chars(reinterpret_cast<const char*>(digitLocation), reinterpret_cast<const char*>(whitespaceLocation), out);

	return std::move(std::make_tuple(digitLocation, whitespaceLocation, out));
}

template<typename ElemT, typename IterT>
inline static bool EqualToOneOf(const ElemT& elem, IterT start, IterT end) noexcept {
	while (start != end) {
		if (elem == *start) {
			return true;
		}
		++start;
	}
	return false;
}