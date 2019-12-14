#ifdef WEIRDLIB_ENABLE_CRYPTOGRAPHY
#include "../../include/weirdlib_crypto.hpp"
#include "../../include/weirdlib_traits.hpp"
#include "../../include/cpu_detection.hpp"

#include <numeric>
#include <cstdio>

namespace wlib::crypto
{
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
					result = _rdrand64_step(reinterpret_cast<unsigned long long*>(&output));
				} while (result == 0);
			#endif

			return output;
		}

	#elif defined(__unix__)

		uint16_t SecureRandomInteger_u16() {
			uint16_t output;

			auto ptr = fopen("/dev/random", "rb");
			if (ptr == nullptr) {
				throw std::runtime_error("No random device found on machine");
			}

			fread(&output, 2, 1, ptr);
			fclose(ptr);

			return output;
		}

		uint32_t SecureRandomInteger_u32() {
			uint32_t output;

			auto ptr = fopen("/dev/random", "rb");
			if (ptr == nullptr) {
				throw std::runtime_error("No random device found on machine");
			}

			fread(&output, 4, 1, ptr);
			fclose(ptr);

			return output;
		}

		uint64_t SecureRandomInteger_u64() {
			uint64_t output;

			auto ptr = fopen("/dev/random", "rb");
			if (ptr == nullptr) {
				throw std::runtime_error("No random device found on machine");
			}

			fread(&output, 8, 1, ptr);
			fclose(ptr);

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
					result = _rdseed64_step(reinterpret_cast<unsigned long long*>(&output));
				} while (result == 0);
			#endif

			return output;
		}

	#elif defined(__unix__)

		uint16_t SecureRandomSeed_u16() {
			uint16_t output;

			auto ptr = fopen("/dev/urandom", "rb");
			if (ptr == nullptr) {
				throw std::runtime_error("No random device found on machine");
			}

			fread(&output, 2, 1, ptr);
			fclose(ptr);

			return output;
		}

		uint32_t SecureRandomSeed_u32() {
			uint32_t output;

			auto ptr = fopen("/dev/urandom", "rb");
			if (ptr == nullptr) {
				throw std::runtime_error("No random device found on machine");
			}

			fread(&output, 4, 1, ptr);
			fclose(ptr);

			return output;
		}

		uint64_t SecureRandomSeed_u64() {
			uint64_t output;

			auto ptr = fopen("/dev/urandom", "rb");
			if (ptr == nullptr) {
				throw std::runtime_error("No random device found on machine");
			}

			fread(&output, 8, 1, ptr);
			fclose(ptr);

			return output;
		}

	#endif
} // namespace wlib::crypt

#endif
