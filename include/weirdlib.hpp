#pragma once
#include <string>
#include <type_traits>

namespace wlib
{
	size_t strlen(const char* s) noexcept;

	bool strcmp(const std::string& str0, const std::string& str1);
	bool strcmp(const std::string& str0, const std::string& str1, const size_t len);

	// strncmp is aliased to strcmp overloaded for a max len parameter
	template<typename SType>
	bool strncmp(SType&& str0, SType&& str1, const size_t len) {
		return wlib::strcmp(std::forward<SType>(str0), std::forward<SType>(str1), len);
	}

	namespace bop
	{
		template<typename T, typename TPos>
		constexpr inline bool test(const T x, const TPos pos) noexcept {
			return (x & (1u << pos)) != 0;
		}

		template<typename T, typename TPos>
		constexpr inline void set(T* x, const TPos pos) noexcept {
			*x |= (1u << pos);
		}

		template<typename T, typename TPos>
		constexpr inline T set(const T x, const TPos pos) noexcept {
			return x | (1u << pos);
		}

		template<typename T, typename TPos>
		constexpr inline void reset(T* x, const TPos pos) noexcept {
			*x &= ~(1u << pos);
		}

		template<typename T, typename TPos>
		constexpr inline T reset(const T x, const TPos pos) noexcept {
			return x & ~(1u << pos);
		}

		template<typename T, typename TPos>
		constexpr inline void toggle(T* x, const TPos pos) noexcept {
			*x ^= 1u << pos;
		}

		template<typename T, typename TPos>
		constexpr inline T toggle(const T x, const TPos pos) noexcept {
			return x ^ 1u << pos;
		}

		template<typename T>
		constexpr inline T clear_leftmost_set(const T x) noexcept {
			return x & (x - 1);
		}

		template<typename T>
		constexpr uint32_t count_trailing_zeros(T x) noexcept {
			if (x == 0) return (sizeof(T) * 8);

			uint32_t count = 0;
			while ((x & 1) == 0) {
					x = x >> 1;
					count++;
			}
			return count;
		}

		template<typename T>
		constexpr uint32_t count_leading_zeros(T x) noexcept {
			if (x == 0) return sizeof(x) * 8;

			uint32_t count = 0;
			while (!(x < 0)) {
				count++;
				x <<= 1;
			}
			return count;
		}

		// Alias for count_trailing_zeros
		template<typename T>
		constexpr uint32_t ctz(T&& x) noexcept {
			return count_trailing_zeros(std::forward<T>(x));
		}

		// Alias for count_leading_zeros
		template<typename T>
		constexpr uint32_t clz(T&& x) noexcept {
			return count_leading_zeros(std::forward<T>(x));
		}

	} // bop



} // wlib
