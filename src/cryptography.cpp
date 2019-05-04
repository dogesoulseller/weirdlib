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
	template<uint16_t rev_polynomial>
	inline constexpr std::array<uint16_t, 256> __compileTimeGenerateCRC16Table() noexcept {
		std::array<uint16_t, 256> result{};

		for (uint16_t i = 0; i < 256; i++) {
			uint16_t checksum = i;

			for (size_t j = 0; j < 8; j++) {
				checksum = (checksum >> 1) ^ ((checksum & 0x1) ? rev_polynomial : 0);
			}
			result[i] = checksum;
		}

		return result;
	}

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

	template<uint64_t rev_polynomial>
	inline constexpr std::array<uint64_t, 256> __compileTimeGenerateCRC64Table() noexcept {
		std::array<uint64_t, 256> result{};

		for (uint64_t i = 0; i < 256; i++) {
			uint64_t checksum = i;

			for (size_t j = 0; j < 8; j++) {
				checksum = (checksum >> 1) ^ ((checksum & 0x1) ? rev_polynomial : 0);
			}
			result[i] = checksum;
		}

		return result;
	}

	inline constexpr std::array<uint16_t, 256> CRC16CCITTLookupTable = __compileTimeGenerateCRC16Table<0x8408>();	// CCITT (0x1021)
	inline constexpr std::array<uint16_t, 256> CRC16ARCLookupTable   = __compileTimeGenerateCRC16Table<0xA001>();	// ARC (0x8005)
	inline constexpr std::array<uint16_t, 256> CRC16DNPLookupTable   = __compileTimeGenerateCRC16Table<0xA6BC>();	// DNP (0x3D65)

	inline constexpr std::array<uint32_t, 256> CRC32LookupTable  = __compileTimeGenerateCRC32Table<0xEDB88320>();	// ISO-HDLC (0x04C11DB7)
	inline constexpr std::array<uint32_t, 256> CRC32CLookupTable = __compileTimeGenerateCRC32Table<0x82F63B78>();	// Castagnolli polynomial (0x1EDC6F41)
	inline constexpr std::array<uint32_t, 256> CRC32DLookupTable = __compileTimeGenerateCRC32Table<0xD419CC15>();	// BASE91-D (0xA833982B)
	inline constexpr std::array<uint32_t, 256> CRC32AUTOSARLookupTable = __compileTimeGenerateCRC32Table<0xC8DF352F>();		// AUTOSAR (0xF4ACFB13)

	inline constexpr std::array<uint64_t, 256> CRC64XZLookupTable = __compileTimeGenerateCRC64Table<0xC96C5795D7870F42>();	// GO-ECMA
	inline constexpr std::array<uint64_t, 256> CRC64ISOLookupTable = __compileTimeGenerateCRC64Table<0xD800000000000000>();	// GO-ISO


	#if defined(__RDRND__)
	uint16_t SecureRandomInteger_u16() {
		uint16_t output;
		#if defined(__GNUC__) && !defined(__clang__)
		asm volatile (
			".rnd_failed_int_u16:;"
			"rdrand ax;"
			"jnc .rnd_failed_int_u16;"
			: [out]"=a"(output)
			:
			:
		);
		#else
			int result;
			do {
				result = _rdrand16_step(&output);
			} while (result == 0);
		#endif

		return output;
	}

	uint32_t SecureRandomInteger_u32() {
		uint32_t output;
		#if defined(__GNUC__) && !defined(__clang__)
		asm volatile (
			".rnd_failed_int_u32:;"
			"rdrand eax;"
			"jnc .rnd_failed_int_u32;"
			: [out]"=a"(output)
			:
			:
		);
		#else
			int result;
			do {
				result = _rdrand32_step(&output);
			} while (result == 0);
		#endif

		return output;
	}

	uint64_t SecureRandomInteger_u64() {
		uint64_t output;
		#if defined(__GNUC__) && !defined(__clang__)
			asm volatile (
				".rnd_failed_int_u64:;"
				"rdrand rax;"
				"jnc .rnd_failed_int_u64;"
				: [out]"=a"(output)
				:
				:
			);
		#else
			int result;
			do {
				result = _rdrand64_step(&output);
			} while (result == 0);
		#endif

		return output;
	}
	#endif

	#if defined(__RDSEED__)
	uint16_t SecureRandomSeed_u16() {
		uint16_t output;

		#if defined(__GNUC__) && !defined(__clang__)
			asm volatile (
				".rnd_failed_seed_u16:;"
				"rdseed ax;"
				"jnc .rnd_failed_seed_u16;"
				: [out]"=a"(output)
				:
				:
			);
		#else
			int result;
			do {
				result = _rdseed16_step(&output);
			} while (result == 0);
		#endif

		return output;
	}

	uint32_t SecureRandomSeed_u32() {
		uint32_t output;
		#if defined(__GNUC__) && !defined(__clang__)
			asm volatile (
				".rnd_failed_seed_u32:;"
				"rdseed eax;"
				"jnc .rnd_failed_seed_u32;"
				: [out]"=a"(output)
				:
				:
			);
		#else
			int result;
			do {
				result = _rdseed32_step(&output);
			} while (result == 0);
		#endif

		return output;
	}

	uint64_t SecureRandomSeed_u64() {
		uint64_t output;

		#if defined(__GNUC__) && !defined(__clang__)
			asm volatile (
				".rnd_failed_seed_u64:;"
				"rdseed rax;"
				"jnc .rnd_failed_seed_u64;"
				: [out]"=a"(output)
				:
				:
			);
		#else
			int result;
			do {
				result = _rdseed64_step(&output);
			} while (result == 0);
		#endif

		return output;
	}

	#endif



	template<uint16_t initVal = 0xFFFF, uint16_t xorout = 0xFFFF>
	uint16_t CRC16_base(const uint8_t* first, const uint8_t* last, const std::array<uint16_t, 256>& lookupTable) {
		#if defined(__x86_64__) && defined(WLIB_ENABLE_PREFETCH)
			// This might kill Ivy Bridge and Jaguar performance
			// For some reason prefetches are extremely slow on those
			_mm_prefetch(first, _MM_HINT_T0);
			_mm_prefetch(first+64, _MM_HINT_T0);
			_mm_prefetch(first+128, _MM_HINT_T0);
			_mm_prefetch(first+192, _MM_HINT_T0);
		#endif

		return uint16_t(xorout) ^ std::accumulate(first, last, uint16_t(initVal),
			[&lookupTable](uint16_t sum, uint16_t val) {
				return lookupTable[(sum ^ val) & 0xFF] ^ (sum >> 8);
			});
	}

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

	template<uint64_t initVal = 0xFFFFFFFFFFFFFFFF, uint64_t xorout = 0xFFFFFFFFFFFFFFFF>
	uint64_t CRC64_base(const uint8_t* first, const uint8_t* last, const std::array<uint64_t, 256>& lookupTable) {
		#if defined(__x86_64__) && defined(WLIB_ENABLE_PREFETCH)
			// This might kill Ivy Bridge and Jaguar performance
			// For some reason prefetches are extremely slow on those
			_mm_prefetch(first, _MM_HINT_T0);
			_mm_prefetch(first+64, _MM_HINT_T0);
			_mm_prefetch(first+128, _MM_HINT_T0);
			_mm_prefetch(first+192, _MM_HINT_T0);
		#endif

		return uint64_t(xorout) ^ std::accumulate(first, last, uint64_t(initVal),
			[&lookupTable](uint64_t sum, uint64_t val) {
				return lookupTable[(sum ^ val) & 0xFF] ^ (sum >> 8);
			});
	}




	uint16_t CRC16ARC(uint8_t* first, uint8_t* last) {
		return CRC16_base<0x0000, 0x0000>(first, last, CRC16ARCLookupTable);
	}

	uint16_t CRC16DNP(uint8_t* first, uint8_t* last) {
		return CRC16_base<0x0000, 0xFFFF>(first, last, CRC16DNPLookupTable);
	}

	uint16_t CRC16MAXIM(uint8_t* first, uint8_t* last) {
		return CRC16_base<0x0000, 0xFFFF>(first, last, CRC16ARCLookupTable);
	}

	uint16_t CRC16USB(uint8_t* first, uint8_t* last) {
		return CRC16_base<0xFFFF, 0xFFFF>(first, last, CRC16ARCLookupTable);
	}

	uint16_t CRC16X_25(uint8_t* first, uint8_t* last) {
		return CRC16_base<0xFFFF, 0xFFFF>(first, last, CRC16CCITTLookupTable);
	}




	uint32_t CRC32(uint8_t* first, uint8_t* last) {
		return CRC32_base(first, last, CRC32LookupTable);
	}

	uint32_t CRC32C(uint8_t* first, uint8_t* last) {
		return CRC32_base(first, last, CRC32CLookupTable);
	}

	uint32_t CRC32D(uint8_t* first, uint8_t* last) {
		return CRC32_base(first, last, CRC32DLookupTable);
	}

	uint32_t CRC32AUTOSAR(uint8_t* first, uint8_t* last) {
		return CRC32_base(first, last, CRC32AUTOSARLookupTable);
	}

	uint32_t CRC32JAMCRC(uint8_t* first, uint8_t* last) {
		return CRC32_base<0xFFFFFFFF, 0x00000000>(first, last, CRC32LookupTable);	// JAMCRC uses the same polynomial as the ISO standard
	}




	uint64_t CRC64XZ(uint8_t* first, uint8_t* last) {
		return CRC64_base(first, last, CRC64XZLookupTable);
	}

	uint64_t CRC64ISO(uint8_t* first, uint8_t* last) {
		return CRC64_base(first, last, CRC64ISOLookupTable);
	}

	} // namespace crypto
} // namespace wlib
