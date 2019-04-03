#pragma once
#include <string>
#include <type_traits>
#include <limits>

#if defined(_MSC_VER)
#include <intrin.h>
	#if defined(__AVX__)
		#include <immintrin.h>
	#endif
#endif

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
		template<typename T, typename TPos, typename = std::enable_if_t<traits::has_bitops_v<T>>>
		constexpr bool test(const T x, const TPos pos) noexcept {
			return (x & (1u << pos)) != 0;
		}

		template<typename T, typename TPos, typename = std::enable_if_t<traits::has_bitops_v<T>>>
		void set_ip(T& x, const TPos pos) noexcept {
			x |= (1u << pos);
		}

		template<typename T, typename TPos, typename = std::enable_if_t<traits::has_bitops_v<T>>>
		constexpr T set(const T x, const TPos pos) noexcept {
			return x | (1u << pos);
		}

		template<typename T, typename TPos, typename = std::enable_if_t<traits::has_bitops_v<T>>>
		void reset_ip(T& x, const TPos pos) noexcept {
			x &= ~(1u << pos);
		}

		template<typename T, typename TPos, typename = std::enable_if_t<traits::has_bitops_v<T>>>
		constexpr T reset(const T x, const TPos pos) noexcept {
			return x & ~(1u << pos);
		}

		template<typename T, typename TPos, typename = std::enable_if_t<traits::has_bitops_v<T>>>
		void toggle_ip(T& x, const TPos pos) noexcept {
			x ^= 1u << pos;
		}

		template<typename T, typename TPos, typename = std::enable_if_t<traits::has_bitops_v<T>>>
		constexpr T toggle(const T x, const TPos pos) noexcept {
			return x ^ 1u << pos;
		}

		template<typename T, typename = std::enable_if_t<traits::has_bitops_v<T>>>
		constexpr T clear_leftmost_set(const T x) noexcept {
			return x & (x - 1);
		}

		template<typename T, typename = std::enable_if_t<traits::has_bitops_v<T>>>
		uint32_t count_trailing_zeros(T x) noexcept {
			if (x == 0) return (sizeof(T) * 8);

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

			uint32_t count = 0;
			while ((x & 1) == 0) {
				x = x >> 1;
				count++;
			}

			return count;
		}

		template<typename T, typename = std::enable_if_t<traits::has_bitops_v<T>>>
		uint32_t count_leading_zeros(T x) noexcept {
			if (x == 0) return sizeof(T) * 8;

			#ifdef __GNUC__
			if constexpr (sizeof(T) == sizeof(uint32_t)) {
				return static_cast<uint32_t>(__builtin_clz(x));
			} else if constexpr (sizeof(T) == sizeof(uint64_t)) {
				return static_cast<uint32_t>(__builtin_clzll(x));
			}
			#elif defined(_MSC_VER) && defined(__AVX__)
			if constexpr (sizeof(T) == sizeof(uint32_t)) {
				return _lzcnt_u32(x);
			}
			#ifdef _M_X64
			else if constexpr (sizeof(T) == sizeof(uint64_t)) {
				return _lzcnt_u64(x);
			}
			#endif
			#endif

			uint32_t total_bits = sizeof(T) * 8;
			uint32_t res = 0;
			while (!(x & (1 << (total_bits - 1)))) {
				x <<= 1;
				res++;
			}

			return res;
		}

		template<typename T, typename = std::enable_if_t<traits::has_bitops_v<T>>>
		uint32_t bit_scan_forward(T x) noexcept {
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

		template<typename T, typename = std::enable_if_t<traits::has_bitops_v<T>>>
		uint32_t bit_scan_reverse(T x) noexcept {
			if (x == 0 || test(x, sizeof(x) * 8 - 1)) return 0;

			return sizeof(T) * 8 - count_leading_zeros<T>(x);
		}

		template<typename T, typename = std::enable_if_t<traits::has_bitops_v<T>>>
		uint32_t population_count(T x) noexcept {
			#ifdef __GNUC__
			if constexpr (sizeof(T) == sizeof(uint32_t)) {
				return static_cast<uint32_t>(__builtin_popcount(x));
			} else if constexpr (sizeof(T) == sizeof(uint64_t)) {
				return static_cast<uint32_t>(__bulitin_popcountll(x));
			}
			#elif defined(_MSC_VER) && defined(__AVX__)
			if constexpr (sizeof(T) == sizeof(uint32_t)) {
				return _mm_popcnt_u32(x);
			}
			#ifdef _M_X64
			else if constexpr (sizeof(T) == sizeof(uint64_t)) {
				return _mm_popcnt_u64(x);
			}
			#endif
			#endif

			// Some compilers will detect this as a popcnt and replace it with a single instruction
			// Notably clang - love that beast
			unsigned int c;
			for (c = 0; x; c++) {
				x &= x - 1;
			}

			return c;
		}

		// Alias for population_count
		template<typename T, typename = std::enable_if_t<traits::has_bitops_v<T>>>
		uint32_t popcnt(T&& x) noexcept {
			return population_count(std::forward<T>(x));
		}

		// Alias for count_trailing_zeros
		template<typename T, typename = std::enable_if_t<traits::has_bitops_v<T>>>
		uint32_t ctz(T&& x) noexcept {
			return count_trailing_zeros(std::forward<T>(x));
		}

		// Alias for count_leading_zeros
		template<typename T, typename = std::enable_if_t<traits::has_bitops_v<T>>>
		uint32_t clz(T&& x) noexcept {
			return count_leading_zeros(std::forward<T>(x));
		}

		// Alias for bit_scan_forward
		template<typename T, typename = std::enable_if_t<traits::has_bitops_v<T>>>
		uint32_t bsf(T&& x) noexcept {
			return bit_scan_forward(std::forward<T>(x));
		}

		// Alias for bit_scan_reverse
		template<typename T, typename = std::enable_if_t<traits::has_bitops_v<T>>>
		uint32_t bsr(T&& x) noexcept {
			return bit_scan_reverse(std::forward<T>(x));
		}

	} // bop

} // wlib
