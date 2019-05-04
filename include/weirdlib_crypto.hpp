#pragma once
#include <cstdint>
#include <array>

namespace wlib
{

/// Functions that deal with cryptography and hashing
/// SecureRandomInteger / Seed functions are only valid when the appropriate compiler defines are set
namespace crypto
{
	#if defined(__RDRND__)

	/// Generate random 16-bit integer
	/// @return hardware-generated 16-bit integer
	uint16_t SecureRandomInteger_u16();

	/// Generate random 32-bit integer
	/// @return hardware-generated 32-bit integer
	uint32_t SecureRandomInteger_u32();

	/// Generate random 64-bit integer
	/// @return hardware-generated 64-bit integer
	uint64_t SecureRandomInteger_u64();
	#endif

	#if defined(__RDSEED__)

	/// Generate random 16-bit seed
	/// @return hardware-generated 16-bit integer meant for use as a seed
	uint16_t SecureRandomSeed_u16();

	/// Generate random 32-bit seed
	/// @return hardware-generated 32-bit integer meant for use as a seed
	uint32_t SecureRandomSeed_u32();

	/// Generate random 64-bit seed
	/// @return hardware-generated 64-bit integer meant for use as a seed
	uint64_t SecureRandomSeed_u64();
	#endif


	/// Generate a CRC16/ARC hash
	/// @param first pointer (inclusive) to start of byte array
	/// @param last pointer (exclusive) to end of byte array
	/// @return CRC16/ARC value
	uint16_t CRC16ARC(uint8_t* first, uint8_t* last);

	/// Generate a CRC16/DNP hash
	/// @param first pointer (inclusive) to start of byte array
	/// @param last pointer (exclusive) to end of byte array
	/// @return CRC16/DNP value
	uint16_t CRC16DNP(uint8_t* first, uint8_t* last);

	/// Generate a CRC16/MAXIM hash
	/// @param first pointer (inclusive) to start of byte array
	/// @param last pointer (exclusive) to end of byte array
	/// @return CRC16/MAXIM value
	uint16_t CRC16MAXIM(uint8_t* first, uint8_t* last);

	/// Generate a CRC16/USB hash
	/// @param first pointer (inclusive) to start of byte array
	/// @param last pointer (exclusive) to end of byte array
	/// @return CRC16/USB value
	uint16_t CRC16USB(uint8_t* first, uint8_t* last);

	/// Generate a CRC16/X-25 hash
	/// @param first pointer (inclusive) to start of byte array
	/// @param last pointer (exclusive) to end of byte array
	/// @return CRC16/X-25 value
	uint16_t CRC16X_25(uint8_t* first, uint8_t* last);

	/// Generate a CRC32 hash (ISO-HDLC)
	/// @param first pointer (inclusive) to start of byte array
	/// @param last pointer (exclusive) to end of byte array
	/// @return CRC32 value
	uint32_t CRC32(uint8_t* first, uint8_t* last);

	/// Generate a CRC32C hash (Castagnolli)
	/// @param first pointer (inclusive) to start of byte array
	/// @param last pointer (exclusive) to end of byte array
	/// @return CRC32C value
	uint32_t CRC32C(uint8_t* first, uint8_t* last);

	/// Generate a CRC32D hash (BASE91-D)
	/// @param first pointer (inclusive) to start of byte array
	/// @param last pointer (exclusive) to end of byte array
	/// @return CRC32D value
	uint32_t CRC32D(uint8_t* first, uint8_t* last);

	/// Generate a CRC32/AUTOSAR hash
	/// @param first pointer (inclusive) to start of byte array
	/// @param last pointer (exclusive) to end of byte array
	/// @return CRC32/AUTOSAR value
	uint32_t CRC32AUTOSAR(uint8_t* first, uint8_t* last);

	/// Generate a CRC32/JAMCRC hash
	/// @param first pointer (inclusive) to start of byte array
	/// @param last pointer (exclusive) to end of byte array
	/// @return CRC32/JAMCRC value
	uint32_t CRC32JAMCRC(uint8_t* first, uint8_t* last);

	/// Generate a CRC64/XZ hash
	/// @param first pointer (inclusive) to start of byte array
	/// @param last pointer (exclusive) to end of byte array
	/// @return CRC64/JAMCRC value
	uint64_t CRC64XZ(uint8_t* first, uint8_t* last);

	/// Generate a CRC64/GO-ISO hash
	/// @param first pointer (inclusive) to start of byte array
	/// @param last pointer (exclusive) to end of byte array
	/// @return CRC64/GO-ISO value
	uint64_t CRC64ISO(uint8_t* first, uint8_t* last);

} // namespace crypto

} // namespace wlib
