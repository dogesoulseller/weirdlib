#pragma once
#include <array>
#include <cstdint>

template<uint16_t rev_polynomial>
static constexpr std::array<uint16_t, 256> __compileTimeGenerateCRC16Table() noexcept {
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
static constexpr std::array<uint32_t, 256> __compileTimeGenerateCRC32Table() noexcept {
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
static constexpr std::array<uint64_t, 256> __compileTimeGenerateCRC64Table() noexcept {
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

static constexpr std::array<uint16_t, 256> CRC16CCITTLookupTable = __compileTimeGenerateCRC16Table<0x8408>();	// CCITT (0x1021)
static constexpr std::array<uint16_t, 256> CRC16ARCLookupTable   = __compileTimeGenerateCRC16Table<0xA001>();	// ARC (0x8005)
static constexpr std::array<uint16_t, 256> CRC16DNPLookupTable   = __compileTimeGenerateCRC16Table<0xA6BC>();	// DNP (0x3D65)

static constexpr std::array<uint32_t, 256> CRC32LookupTable  = __compileTimeGenerateCRC32Table<0xEDB88320>();	// ISO-HDLC (0x04C11DB7)
static constexpr std::array<uint32_t, 256> CRC32CLookupTable = __compileTimeGenerateCRC32Table<0x82F63B78>();	// Castagnolli polynomial (0x1EDC6F41)
static constexpr std::array<uint32_t, 256> CRC32DLookupTable = __compileTimeGenerateCRC32Table<0xD419CC15>();	// BASE91-D (0xA833982B)
static constexpr std::array<uint32_t, 256> CRC32AUTOSARLookupTable = __compileTimeGenerateCRC32Table<0xC8DF352F>();		// AUTOSAR (0xF4ACFB13)

static constexpr std::array<uint64_t, 256> CRC64XZLookupTable = __compileTimeGenerateCRC64Table<0xC96C5795D7870F42>();	// GO-ECMA
static constexpr std::array<uint64_t, 256> CRC64ISOLookupTable = __compileTimeGenerateCRC64Table<0xD800000000000000>();	// GO-ISO
