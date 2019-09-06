#pragma once
#include <cstdint>
#include <array>
#include <vector>
#include <string>
#include <sstream>

namespace wlib
{

/// Functions that deal with cryptography and hashing
/// SecureRandomInteger / Seed functions are only valid when the appropriate compiler defines are set
namespace crypto
{
	static constexpr std::array<uint32_t, 5> SHA1_DIGEST_BASE = {0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476, 0xc3d2e1f0};
	static constexpr size_t SHA1_BLOCK_INTS = 16;	// number of 32bit integers per SHA1 block
	static constexpr size_t SHA1_BLOCK_BYTES = SHA1_BLOCK_INTS * 4;

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

	/// SHA-1 generator <br>
	/// This version does not use Intel SHA Extensions yet<br>
	/// WARNING: SHA1 is no longer considered safe for cryptography. Use the SHA3 family if possible, or at least SHA2
	class SHA1
	{
		public:
		SHA1();
		void reset();
		void update(const std::string& str);
		void update(std::istream& str);
		std::string finalize_to_string();

		private:

		uint32_t blk(const std::array<uint32_t, SHA1_BLOCK_INTS>& block, const size_t i);

		void R0(const std::array<uint32_t, SHA1_BLOCK_INTS>& block, const uint32_t v, uint32_t &w, const uint32_t x, const uint32_t y, uint32_t &z, const size_t i);
		void R1(std::array<uint32_t, SHA1_BLOCK_INTS>& block, const uint32_t v, uint32_t &w, const uint32_t x, const uint32_t y, uint32_t &z, const size_t i);
		void R2(std::array<uint32_t, SHA1_BLOCK_INTS>& block, const uint32_t v, uint32_t &w, const uint32_t x, const uint32_t y, uint32_t &z, const size_t i);
		void R3(std::array<uint32_t, SHA1_BLOCK_INTS>& block, const uint32_t v, uint32_t &w, const uint32_t x, const uint32_t y, uint32_t &z, const size_t i);
		void R4(std::array<uint32_t, SHA1_BLOCK_INTS>& block, const uint32_t v, uint32_t &w, const uint32_t x, const uint32_t y, uint32_t &z, const size_t i);

		void transform(std::array<uint32_t, SHA1_BLOCK_INTS>& block);

		std::array<uint32_t, SHA1_BLOCK_INTS> buffer_to_block();

		std::array<uint32_t, 5> digest;
		std::string buffer;
		size_t transform_count;
	};

} // namespace crypto

} // namespace wlib
