#include "../include/weirdlib_crypto.hpp"
#include <numeric>
#include "cpu_detection.hpp"

#ifdef __x86_64__
	#include <immintrin.h>
#endif

namespace wlib
{
namespace crypto
{
	template<uint32_t rev_polynomial>
	inline constexpr std::array<uint32_t, 256> __compileTimeGenerateCRC32Table() noexcept {
		std::array<uint32_t, 256> result{};

		for (uint32_t i = 0; i < 256; i++) {
			uint32_t checksum = i;

			for (size_t j = 0; j < 8; j++) {
				checksum = (checksum >> 1) ^ ((checksum & 0x1) ? rev_polynomial : 0);
			}
			result[i] = checksum;
		}

		return result;
	}

	inline constexpr std::array<uint32_t, 256> CRC32LookupTable  = __compileTimeGenerateCRC32Table<0xEDB88320>();	// ISO-HDLC (0x04C11DB7)
	inline constexpr std::array<uint32_t, 256> CRC32CLookupTable = __compileTimeGenerateCRC32Table<0x82F63B78>();	// Castagnolli polynomial (0x1EDC6F41)
	inline constexpr std::array<uint32_t, 256> CRC32DLookupTable = __compileTimeGenerateCRC32Table<0xD419CC15>();	// BASE91-D (0xA833982B)
	inline constexpr std::array<uint32_t, 256> CRC32AUTOSARLookupTable = __compileTimeGenerateCRC32Table<0xC8DF352F>();		// AUTOSAR (0xF4ACFB13)
	inline constexpr std::array<uint32_t, 256> CRC32JAMCRCLookupTable  = __compileTimeGenerateCRC32Table<0xEDB88320>();		// JAMCRC (0x04C11DB7)

	#if defined(__RDRND__) && defined(__GNUC__) && !defined(__clang__)
	uint16_t SecureRandomInteger_u16() {
		uint16_t output;
		asm volatile (
			".rnd_failed_int_u16:;"
			"rdrand ax;"
			"jnc .rnd_failed_int_u16;"
			: [out]"=a"(output)
			:
			:
		);

		return output;
	}

	uint32_t SecureRandomInteger_u32() {
		uint32_t output;
		asm volatile (
			".rnd_failed_int_u32:;"
			"rdrand eax;"
			"jnc .rnd_failed_int_u32;"
			: [out]"=a"(output)
			:
			:
		);

		return output;
	}

	uint64_t SecureRandomInteger_u64() {
		uint64_t output;
		asm volatile (
			".rnd_failed_int_u64:;"
			"rdrand rax;"
			"jnc .rnd_failed_int_u64;"
			: [out]"=a"(output)
			:
			:
		);

		return output;
	}
	#endif

	#if defined(__RDSEED__) && defined(__GNUC__) && !defined(__clang__)
	uint16_t SecureRandomSeed_u16() {
		uint16_t output;
		asm volatile (
			".rnd_failed_seed_u16:;"
			"rdseed ax;"
			"jnc .rnd_failed_seed_u16;"
			: [out]"=a"(output)
			:
			:
		);

		return output;
	}

	uint32_t SecureRandomSeed_u32() {
		uint32_t output;
		asm volatile (
			".rnd_failed_seed_u32:;"
			"rdseed eax;"
			"jnc .rnd_failed_seed_u32;"
			: [out]"=a"(output)
			:
			:
		);

		return output;
	}

	uint64_t SecureRandomSeed_u64() {
		uint64_t output;
		asm volatile (
			".rnd_failed_seed_u64:;"
			"rdseed rax;"
			"jnc .rnd_failed_seed_u64;"
			: [out]"=a"(output)
			:
			:
		);

		return output;
	}

	#endif

	template<uint32_t initVal = 0xFFFFFFFF, uint32_t xorout = 0xFFFFFFFF>
	uint32_t CRC32_base(const uint8_t* first, const uint8_t* last, const std::array<uint32_t, 256>& lookupTable) {
		#if defined(__x86_64__) && defined(WLIB_ENABLE_PREFETCH)
			// This might kill Ivy Bridge and Jaguar performance
			// For some reason prefetches are extremely slow on those
			_mm_prefetch(first, _MM_HINT_T0);
			_mm_prefetch(first+64, _MM_HINT_T0);
			_mm_prefetch(first+128, _MM_HINT_T0);
			_mm_prefetch(first+192, _MM_HINT_T0);
		#endif

		return uint32_t(xorout) ^ std::accumulate(first, last, uint32_t(initVal),
			[&lookupTable](uint32_t sum, uint32_t val) {
				return lookupTable[(sum ^ val) & 0xFF] ^ (sum >> 8);
			});
	}

	uint32_t CRC32(uint8_t* first, uint8_t* last) {
		return CRC32_base<0xFFFFFFFF>(first, last, CRC32LookupTable);
	}

	uint32_t CRC32C(uint8_t* first, uint8_t* last) {
		return CRC32_base<0xFFFFFFFF>(first, last, CRC32CLookupTable);
	}

	uint32_t CRC32D(uint8_t* first, uint8_t* last) {
		return CRC32_base<0xFFFFFFFF>(first, last, CRC32DLookupTable);
	}

	uint32_t CRC32AUTOSAR(uint8_t* first, uint8_t* last) {
		return CRC32_base<0xFFFFFFFF>(first, last, CRC32AUTOSARLookupTable);
	}

	uint32_t CRC32JAMCRC(uint8_t* first, uint8_t* last) {
		return CRC32_base<0xFFFFFFFF, 0x00000000>(first, last, CRC32JAMCRCLookupTable);
	}

	} // namespace crypto
} // namespace wlib
