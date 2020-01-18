#ifdef WEIRDLIB_ENABLE_CRYPTOGRAPHY
#include "../../include/weirdlib_crypto.hpp"
#include "../../include/weirdlib_traits.hpp"
#include "../../include/cpu_detection.hpp"

#include "crc_tables.hpp"

#include <numeric>

namespace wlib::crypto
{
	template<uint16_t initVal = 0xFFFF, uint16_t xorout = 0xFFFF>
	static uint16_t CRC16_base(const uint8_t* first, const uint8_t* last, const std::array<uint16_t, 256>& lookupTable) {
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
	static uint32_t CRC32_base(const uint8_t* first, const uint8_t* last, const std::array<uint32_t, 256>& lookupTable) {
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
	static uint64_t CRC64_base(const uint8_t* first, const uint8_t* last, const std::array<uint64_t, 256>& lookupTable) {
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


	uint16_t CRC16ARC(const uint8_t* first, const uint8_t* last) {
		return CRC16_base<0x0000, 0x0000>(first, last, CRC16ARCLookupTable);
	}

	uint16_t CRC16DNP(const uint8_t* first, const uint8_t* last) {
		return CRC16_base<0x0000, 0xFFFF>(first, last, CRC16DNPLookupTable);
	}

	uint16_t CRC16MAXIM(const uint8_t* first, const uint8_t* last) {
		return CRC16_base<0x0000, 0xFFFF>(first, last, CRC16ARCLookupTable);
	}

	uint16_t CRC16USB(const uint8_t* first, const uint8_t* last) {
		return CRC16_base<0xFFFF, 0xFFFF>(first, last, CRC16ARCLookupTable);
	}

	uint16_t CRC16X_25(const uint8_t* first, const uint8_t* last) {
		return CRC16_base<0xFFFF, 0xFFFF>(first, last, CRC16CCITTLookupTable);
	}


	uint32_t CRC32(const uint8_t* first, const uint8_t* last) {
		return CRC32_base(first, last, CRC32LookupTable);
	}

	uint32_t CRC32C(const uint8_t* first, const uint8_t* last) {
		return CRC32_base(first, last, CRC32CLookupTable);
	}

	uint32_t CRC32D(const uint8_t* first, const uint8_t* last) {
		return CRC32_base(first, last, CRC32DLookupTable);
	}

	uint32_t CRC32AUTOSAR(const uint8_t* first, const uint8_t* last) {
		return CRC32_base(first, last, CRC32AUTOSARLookupTable);
	}

	uint32_t CRC32JAMCRC(const uint8_t* first, const uint8_t* last) {
		return CRC32_base<0xFFFFFFFF, 0x00000000>(first, last, CRC32LookupTable);	// JAMCRC uses the same polynomial as the ISO standard
	}


	uint64_t CRC64XZ(const uint8_t* first, const uint8_t* last) {
		return CRC64_base(first, last, CRC64XZLookupTable);
	}

	uint64_t CRC64ISO(const uint8_t* first, const uint8_t* last) {
		return CRC64_base(first, last, CRC64ISOLookupTable);
	}
}
#endif // WEIRDLIB_ENABLE_CRYPTOGRAPHY
