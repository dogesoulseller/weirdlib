#pragma once

#include <type_traits>

#include "cpu_detection.hpp"
#include "weirdlib_traits.hpp"

namespace wlib
{

/// Bit operations on variable-length integers
///
/// - Inline assembly for GCC-compatible compilers is provided wherever possible
///	- Compiler builtins are used when they're available
namespace bop
{
	//-----------------------//
	// SINGLE BIT OPERATIONS //
	//-----------------------//

	/// Test bit at pos of value x
	/// @param x input value
	/// @param pos bit position (Little Endian)
	/// @return boolean true if bit set, false if bit reset
	template<typename T, typename = std::enable_if_t<traits::has_bitops_v<T>>>
	constexpr bool test(const T x, const uint8_t pos) noexcept {
		return ((x >> pos) & 1) != 0;
	}

	/// Set bit at pos of value x (in-place)
	/// @param x value to modify
	/// @param pos bit position (Little Endian)
	template<typename T, typename = std::enable_if_t<traits::has_bitops_v<T>>>
	void set_ip(T& x, const uint8_t pos) noexcept {
		x |= (1u << pos);
	}

	/// Set bit at pos of value x
	/// @param x input value
	/// @param pos bit position (Little Endian)
	/// @return modified x
	template<typename T, typename = std::enable_if_t<traits::has_bitops_v<T>>>
	constexpr T set(const T x, const uint8_t pos) noexcept {
		return x | T(1u << pos);
	}

	/// Reset bit at pos of value x (in-place)
	/// @param x value to modify
	/// @param pos bit position (Little Endian)
	template<typename T, typename = std::enable_if_t<traits::has_bitops_v<T>>>
	void reset_ip(T& x, const uint8_t pos) noexcept {
		x &= ~(1u << pos);
	}

	/// Reset bit at pos of value x
	/// @param x input value
	/// @param pos bit position (Little Endian)
	/// @return modified x
	template<typename T, typename = std::enable_if_t<traits::has_bitops_v<T>>>
	constexpr T reset(const T x, const uint8_t pos) noexcept {
		return x & ~(1u << pos);
	}

	/// Toggle (complement) bit at pos of value x (in-place)
	/// @param x value to modify
	/// @param pos bit position (Little Endian)
	template<typename T, typename = std::enable_if_t<traits::has_bitops_v<T>>>
	void toggle_ip(T& x, const uint8_t pos) noexcept {
		x ^= 1u << pos;
	}

	/// Complement bit at pos of value x
	/// @param x input value
	/// @param pos bit position (Little Endian)
	/// @return modified x
	template<typename T, typename = std::enable_if_t<traits::has_bitops_v<T>>>
	constexpr T toggle(const T x, const uint8_t pos) noexcept {
		return x ^ T(1u << pos);
	}

	/// Reset leftmost bit of x
	/// @param x input value
	/// @return modified x
	template<typename T, typename = std::enable_if_t<traits::has_bitops_v<T>>>
	constexpr T clear_leftmost_set(const T x) noexcept {
		return x & (x - 1);
	}

	//-------------------------//
	// BIT SCANNING OPERATIONS //
	//-------------------------//

	/// Count contiguous reset bits starting from least significant end
	/// @param x tested value
	/// @return count of trailing zeros
	/// @see count_leading_zeros
	template<typename T, typename = std::enable_if_t<traits::has_bitops_v<T>>>
	uint32_t count_trailing_zeros(T x) noexcept {
		if (x == 0) return (sizeof(T) * 8);

		#if defined(__GNUC__) && !defined(__clang__) && defined(__BMI__)
			[[maybe_unused]] T output;
			if constexpr (std::is_same_v<T, uint16_t>) {
				asm (
					"tzcnt ax, ax;"
					: "=a"(output)
					: "a"(x)
				);
				return output;
			} else if constexpr (std::is_same_v<T, uint32_t>) {
				asm (
					"tzcnt eax, eax;"
					: "=a"(output)
					: "a"(x)
				);
				return output;
			} else if constexpr (std::is_same_v<T, uint64_t>) {
				asm (
					"tzcnt rax, rax;"
					: "=a"(output)
					: "a"(x)
				);
				return output;
			}
		#elif defined(__BMI__) || X86_SIMD_LEVEL >= LV_AVX2
			if constexpr (std::is_same_v<T, uint32_t>) {
				return _tzcnt_u32(x);
			} else if constexpr (std::is_same_v<T, uint64_t>) {
				return _tzcnt_u64(x);
			}
		#endif

		#ifdef __GNUC__
			if constexpr (std::is_same_v<T, uint32_t>) {
				return static_cast<uint32_t>(__builtin_ctz(x));
			} else if constexpr (std::is_same_v<T, uint64_t>) {
				return static_cast<uint32_t>(__builtin_ctzll(x));
			}
		#elif defined(_MSC_VER)
			int result = 0;
			if constexpr (sizeof(T) <= sizeof(uint32_t)) {
				_BitScanForward(&result, x);
				return result;
			} else if constexpr (sizeof(T) == sizeof(uint64_t)) {
				_BitScanForward64(&result, x);
				return result;
			}
		#endif

		uint32_t count = 0;
		while ((x & 1) == 0) {
			x = x >> 1;
			count++;
		}

		return count;
	}

	/// Count contiguous reset bits starting from most significant end
	/// @param x tested value
	/// @return count of leading zeros
	/// @see count_trailing_zeros
	template<typename T, typename = std::enable_if_t<traits::has_bitops_v<T>>>
	uint32_t count_leading_zeros(T x) noexcept {
		if (x == 0) return sizeof(T) * 8;

		#if defined(__GNUC__) && !defined(__clang__) && (defined(__ABM__) || defined(__BMI__))
			[[maybe_unused]] T output;
			if constexpr (std::is_same_v<T, uint16_t>) {
				asm (
					"lzcnt ax, ax;"
					: "=a"(output)
					: "a"(x)
				);
				return output;
			} else if constexpr (std::is_same_v<T, uint32_t>) {
				asm (
					"lzcnt eax, eax;"
					: "=a"(output)
					: "a"(x)
				);
				return output;
			} else if constexpr (std::is_same_v<T, uint64_t>) {
				asm (
					"lzcnt rax, rax;"
					: "=a"(output)
					: "a"(x)
				);
				return output;
			}
		#elif (defined(__ABM__) || defined(__BMI__)) || X86_SIMD_LEVEL >= LV_AVX2
			if constexpr (std::is_same_v<T, uint32_t>) {
				return _lzcnt_u32(x);
			} else if constexpr (std::is_same_v<T, uint64_t>) {
				return _lzcnt_u64(x);
			}
		#endif

		#ifdef __GNUC__
			if constexpr (sizeof(T) == sizeof(uint32_t)) {
				return static_cast<uint32_t>(__builtin_clz(x));
			} else if constexpr (sizeof(T) == sizeof(uint64_t)) {
				return static_cast<uint32_t>(__builtin_clzll(x));
			}
		#elif defined(_MSC_VER) && defined(__AVX__)
			if constexpr (sizeof(T) == sizeof(uint32_t)) {
				return _lzcnt_u32(x);
			} else if constexpr (sizeof(T) == sizeof(uint64_t)) {
				return _lzcnt_u64(x);
			}
		#endif

		uint32_t total_bits = sizeof(T) * 8;
		uint32_t res = 0;
		while (!(x & (1 << (total_bits - 1)))) {
			x <<= 1;
			res++;
		}

		return res;
	}

	/// Find last set bit (from least significant)
	/// @param x tested value
	/// @return 0-based index of last set bit
	/// @see bit_scan_reverse
	template<typename T, typename = std::enable_if_t<traits::has_bitops_v<T>>>
	uint32_t bit_scan_forward(T x) noexcept {
		#if defined(__GNUC__) && !defined(__clang__) && defined(__x86_64__)
			[[maybe_unused]] [[maybe_unused]] T output;
			if constexpr (std::is_same_v<T, uint16_t>) {
				asm (
					"bsf ax, ax;"
					"cmovz ax, bx;"
					: "=a"(output)
					: "a"(x), "b"(0)
				);
				return output;
			} else if constexpr (std::is_same_v<T, uint32_t>) {
				asm (
					"bsf eax, eax;"
					"cmovz eax, ebx;"
					: "=a"(output)
					: "a"(x), "b"(0)
				);
				return output;
			} else if constexpr (std::is_same_v<T, uint64_t>) {
				asm (
					"bsf rax, rax;"
					"cmovz rax, rbx;"
					: "=a"(output)
					: "a"(x), "b"(0)
				);
				return output;
			}
		#endif

		if (x == 0) return 0;

		#ifdef __GNUC__
		if constexpr (sizeof(T) <= sizeof(uint32_t)) {
			return static_cast<uint32_t>(__builtin_ctz(x));
		} else if constexpr (sizeof(T) == sizeof(uint64_t)) {
			return static_cast<uint32_t>(__builtin_ctzll(x));
		}
		#elif defined(_MSC_VER)
		int result = 0;
		if constexpr (sizeof(T) <= sizeof(uint32_t)) {
			_BitScanForward(&result, x);
			return result;
		} else if constexpr (sizeof(T) == sizeof(uint64_t)) {
			_BitScanForward64(&result, x);
			return result;
		}
		#endif

		return count_trailing_zeros<T>(x);
	}

	/// Find first set bit (from least significant)
	/// @param x tested value
	/// @return 0-based index of first set bit
	/// @see bit_scan_forward
	template<typename T, typename = std::enable_if_t<traits::has_bitops_v<T>>>
	uint32_t bit_scan_reverse(T x) noexcept {
		#if defined(__GNUC__) && !defined(__clang__) && defined(__x86_64__)
			[[maybe_unused]] T output;
			if constexpr (std::is_same_v<T, uint16_t>) {
				asm (
					"bsr ax, ax;"
					"cmovz ax, bx;"
					: "=a"(output)
					: "a"(x), "b"(0)
					:
				);

				return output;
			} else if constexpr (std::is_same_v<T, uint32_t>) {
				asm (
					"bsr eax, eax;"
					"cmovz eax, ebx;"
					: "=a"(output)
					: "a"(x), "b"(0)
					:
				);

				return output;
			} else if constexpr (std::is_same_v<T, uint64_t>) {
				asm (
					"bsr rax, rax;"
					"cmovz rax, rbx;"
					: "=a"(output)
					: "a"(x), "b"(0)
					:
				);

				return output;
			}
		#endif

		if (x == 0) return 0;

		T temp = sizeof(T)*8 - 1;
		while (test(x, temp) == false) {
			temp--;
		}

		return static_cast<uint32_t>(temp);
	}

	/// Count set bits
	/// @param x tested value
	/// @return number of set bits
	template<typename T, typename = std::enable_if_t<traits::has_bitops_v<T>>>
	uint32_t population_count(T x) noexcept {
		#if (defined(__POPCNT__) || defined(__ABM__)) && defined(__GNUC__) && !defined(__clang__)
			[[maybe_unused]] T output;
			if constexpr (std::is_same_v<T, uint16_t>) {
				asm (
					"popcnt ax, ax;"
					: "=a"(output)
					: "a"(x)
				);
				return output;
			} else if constexpr (std::is_same_v<T, uint32_t>) {
				asm (
					"popcnt eax, eax;"
					: "=a"(output)
					: "a"(x)
				);
				return output;
			} else if constexpr (std::is_same_v<T, uint64_t>) {
				asm (
					"popcnt rax, rax;"
					: "=a"(output)
					: "a"(x)
				);
				return output;
			}

		#elif defined(__POPCNT__) || defined(__ABM__)
			if constexpr (std::is_same_v<T, uint32_t>) {
				return _mm_popcnt_u32(x);
			} else if constexpr (std::is_same_v<T, uint64_t>) {
				return _mm_popcnt_u64(x);
			}
		#endif

		#ifdef __GNUC__
			if constexpr (sizeof(T) == sizeof(uint32_t)) {
				return static_cast<uint32_t>(__builtin_popcount(x));
			} else if constexpr (sizeof(T) == sizeof(uint64_t)) {
				return static_cast<uint32_t>(__builtin_popcountll(x));
			}
		#elif defined(_MSC_VER) && defined(__AVX__)
			if constexpr (sizeof(T) == sizeof(uint32_t)) {
				return _mm_popcnt_u32(x);
			}
			else if constexpr (sizeof(T) == sizeof(uint64_t)) {
				return _mm_popcnt_u64(x);
			}
		#endif

		// Some compilers will detect this as a popcnt and replace it with a single instruction
		// Notably clang - love that beast
		uint32_t c = 0;
		while (x) {
			c++;
			x &= (x-1);
		}

		return c;
	}

	//-------------------------//
	// BIT ROTATION OPERATIONS //
	//-------------------------//

	/// Perform a series of bit rotation to the left
	/// @param x input value
	/// @param c rotations steps to perform
	/// @return modified value
	/// @see rotate_right
	template<typename T, typename = std::enable_if_t<std::conjunction_v<std::is_unsigned<T>, traits::has_bitops<T>>>>
	T rotate_left(const T x, uint8_t c) noexcept {
		#if defined(__GNUC__) && !defined(__clang__) && defined(__x86_64__)
			[[maybe_unused]] T output;
			if constexpr (std::is_same_v<T, uint8_t>) {
				asm (
					"rol al, cl;"
				: "=a"(output)
				: "a"(x), "c"(c)
				);
				return output;
			} else if constexpr (std::is_same_v<T, uint16_t>) {
				asm (
					"rol ax, cl;"
				: "=a"(output)
				: "a"(x), "c"(c)
				);
				return output;
			} else if constexpr (std::is_same_v<T, uint32_t>) {
				asm (
					"rol eax, cl;"
				: "=a"(output)
				: "a"(x), "c"(c)
				);
				return output;
			} else if constexpr (std::is_same_v<T, uint64_t>) {
				asm (
					"rol rax, cl;"
				: "=a"(output)
				: "a"(x), "c"(c)
				);
			}
		#endif
		const T mask = (sizeof(T) * 8 - 1);

		c &= mask;
		return (x << c) | (x >> ((-c) & mask));
	}

	/// Perform a series bit rotation to the left
	/// @param x input value
	/// @param c rotations steps to perform
	/// @return modified value
	/// @see rotate_left
	template<typename T, typename = std::enable_if_t<std::conjunction_v<std::is_unsigned<T>, traits::has_bitops<T>>>>
	T rotate_right(const T x, uint8_t c) noexcept {
		#if defined(__GNUC__) && !defined(__clang__) && defined(__x86_64__)
			[[maybe_unused]] T output;
			if constexpr (std::is_same_v<T, uint8_t>) {
				asm (
					"ror al, cl;"
				: "=a"(output)
				: "a"(x), "c"(c)
				);
				return output;
			} else if constexpr (std::is_same_v<T, uint16_t>) {
				asm (
					"ror ax, cl;"
				: "=a"(output)
				: "a"(x), "c"(c)
				);
				return output;
			} else if constexpr (std::is_same_v<T, uint32_t>) {
				asm (
					"ror eax, cl;"
				: "=a"(output)
				: "a"(x), "c"(c)
				);
				return output;
			} else if constexpr (std::is_same_v<T, uint64_t>) {
				asm (
					"ror rax, cl;"
				: "=a"(output)
				: "a"(x), "c"(c)
				);
			}
		#endif
		const T mask = (sizeof(T) * 8 - 1);

		c &= mask;
		return (x >> c) | ( x<< ((-c) & mask));
	}

	//-------------------//
	// OPERATION ALIASES //
	//-------------------//

	/// Alias for {@link population_count}
	template<typename T, typename = std::enable_if_t<traits::has_bitops_v<T>>>
	uint32_t popcnt(T&& x) noexcept {
		return population_count(std::forward<T>(x));
	}

	/// Alias for {@link count_trailing_zeros}
	template<typename T, typename = std::enable_if_t<traits::has_bitops_v<T>>>
	uint32_t ctz(T&& x) noexcept {
		return count_trailing_zeros(std::forward<T>(x));
	}

	/// Alias for {@link count_leading_zeros}
	template<typename T, typename = std::enable_if_t<traits::has_bitops_v<T>>>
	uint32_t clz(T&& x) noexcept {
		return count_leading_zeros(std::forward<T>(x));
	}

	/// Alias for @{link bit_scan_forward}
	template<typename T, typename = std::enable_if_t<traits::has_bitops_v<T>>>
	uint32_t bsf(T&& x) noexcept {
		return bit_scan_forward(std::forward<T>(x));
	}

	/// Alias for @{link bit_scan_reverse}
	template<typename T, typename = std::enable_if_t<traits::has_bitops_v<T>>>
	uint32_t bsr(T&& x) noexcept {
		return bit_scan_reverse(std::forward<T>(x));
	}

} // bop
} // wlib
