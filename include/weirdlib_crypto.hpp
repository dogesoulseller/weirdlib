#pragma once
#include <cstdint>

namespace wlib
{

/// Functions that deal with cryptography and hashing
namespace crypto
{
	#if defined(__RDRND__) && defined(__GNUC__) && !defined(__clang__)

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

	#if defined(__RDSEED__) && defined(__GNUC__) && !defined(__clang__)

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

} // namespace crypto

} // namespace wlib
