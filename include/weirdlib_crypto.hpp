#pragma once
#include <cstdint>

namespace wlib
{
	#if defined (__RDRND__) && defined (__GNUC__)
	/// Generate random 16-bit integer
	uint16_t SecureRandomInteger_u16();

	/// Generate random 32-bit integer
	uint32_t SecureRandomInteger_u32();

	/// Generate random 64-bit integer
	uint64_t SecureRandomInteger_u64();
	#endif

	#if defined (__RDSEED__) && defined (__GNUC__)
	/// Generate random 16-bit integer
	uint16_t SecureRandomSeed_u16();

	/// Generate random 16-bit integer
	uint32_t SecureRandomSeed_u32();

	/// Generate random 16-bit integer
	uint64_t SecureRandomSeed_u64();
	#endif

} // namespace wlib
