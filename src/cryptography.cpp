#include "../include/weirdlib_crypto.hpp"

namespace wlib
{
	#if defined (__RDRND__) && defined (__GNUC__)
	uint16_t SecureRandomInteger_u16() {
		uint16_t output;
		asm volatile (
			"securerandominteger_rnd_failed_u16:;"
			"rdrand ax;"
			"jnc securerandominteger_rnd_failed_u16;"
			: [out]"=a"(output)
			:
			:
		);

		return output;
	}

	uint32_t SecureRandomInteger_u32() {
		uint32_t output;
		asm volatile (
			"securerandominteger_rnd_failed_u32:;"
			"rdrand eax;"
			"jnc securerandominteger_rnd_failed_u32;"
			: [out]"=a"(output)
			:
			:
		);

		return output;
	}

	uint64_t SecureRandomInteger_u64() {
		uint64_t output;
		asm volatile (
			"securerandominteger_rnd_failed_u64:;"
			"rdrand rax;"
			"jnc securerandominteger_rnd_failed_u64;"
			: [out]"=a"(output)
			:
			:
		);

		return output;
	}
	#endif

	#if defined (__RDSEED__) && defined (__GNUC__)
	uint16_t SecureRandomSeed_u16() {
		uint16_t output;
		asm volatile (
			"securerandomseed_rnd_failed_u16:;"
			"rdseed ax;"
			"jnc securerandomseed_rnd_failed_u16;"
			: [out]"=a"(output)
			:
			:
		);

		return output;
	}

	uint32_t SecureRandomSeed_u32() {
		uint32_t output;
		asm volatile (
			"securerandomseed_rnd_failed_u32:;"
			"rdseed eax;"
			"jnc securerandomseed_rnd_failed_u32;"
			: [out]"=a"(output)
			:
			:
		);

		return output;
	}

	uint64_t SecureRandomSeed_u64() {
		uint64_t output;
		asm volatile (
			"securerandomseed_rnd_failed_u64:;"
			"rdseed rax;"
			"jnc securerandomseed_rnd_failed_u64;"
			: [out]"=a"(output)
			:
			:
		);

		return output;
	}

	#endif
} // namespace wlib
