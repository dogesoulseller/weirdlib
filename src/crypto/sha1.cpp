#ifdef WEIRDLIB_ENABLE_CRYPTOGRAPHY
#include "../../include/weirdlib_crypto.hpp"
#include "../../include/weirdlib_bitops.hpp"
#include "../../include/cpu_detection.hpp"
#include "../common.hpp"
#include <cstring>

#include <numeric>
#include <iomanip>

// Implementation modified/modernized from https://github.com/vog/sha1
// Original version is available as public license

namespace wlib::crypto
{
	constexpr uint32_t SHA1_R0_R1_CONSTANT = 0x5a827999;
	constexpr uint32_t SHA1_R2_CONSTANT = 0x6ed9eba1;
	constexpr uint32_t SHA1_R3_CONSTANT = 0x8f1bbcdc;
	constexpr uint32_t SHA1_R4_CONSTANT = 0xca62c1d6;

	SHA1::SHA1() {
		reset();
	}

	void SHA1::reset() noexcept {
		digest = SHA1_DIGEST_BASE;
		transform_count = 0;
		buffer.clear();
	}

	std::array<uint32_t, SHA1_BLOCK_INTS> SHA1::buffer_to_block() noexcept {
		std::array<uint32_t, SHA1_BLOCK_INTS> block;

		for (size_t i = 0; i < block.size(); i++) {
			block[i] = (buffer[4*i+3] & 0xff)
				| (buffer[4*i+2] & 0xff)<<8
				| (buffer[4*i+1] & 0xff)<<16
				| (buffer[4*i+0] & 0xff)<<24;
		}

		return block;
	}

	void SHA1::update(std::istream& s) {
		while (true) {
			std::array<char, SHA1_BLOCK_BYTES> sbuf;
			s.read(sbuf.data(), SHA1_BLOCK_BYTES - buffer.size());
			buffer.append(sbuf.data(), s.gcount());
			if (buffer.size() != SHA1_BLOCK_BYTES) {
				return;
			}

			auto block = buffer_to_block();
			transform(block);
			buffer.clear();
		}
	}

	void SHA1::update(const std::string& str) {
		std::istringstream is(str);
		update(is);
	}

	std::string SHA1::finalize_to_string() {
		uint64_t totalBits = (transform_count * SHA1_BLOCK_BYTES + buffer.size()) * 8;

		// Pad
		buffer += char(0x80);
		size_t originSize = buffer.size();
		while (buffer.size() < SHA1_BLOCK_BYTES) {
			buffer += char(0x00);
		}

		auto block = buffer_to_block();

		if (originSize > SHA1_BLOCK_BYTES - 8) {
			transform(block);
			for (size_t i = 0; i < SHA1_BLOCK_INTS - 2; i++) {
				block[i] = 0;
			}
		}

		block[SHA1_BLOCK_INTS - 1] = uint32_t(totalBits);
		block[SHA1_BLOCK_INTS - 2] = uint32_t(totalBits >> 32);
		transform(block);

		std::ostringstream result;
		result << std::hex << std::setfill('0') << std::setw(8);

		for (size_t i = 0; i < digest.size(); i++) {
			result << digest[i];
		}

		return result.str();
	}

	uint32_t SHA1::blk(const std::array<uint32_t, SHA1_BLOCK_INTS>& block, const size_t i) noexcept {
		return wlib::bop::rotate_left(block[(i+13)&15] ^ block[(i+8)&15] ^ block[(i+2)&15] ^ block[i], 1);
	}

	void SHA1::R0(const std::array<uint32_t, SHA1_BLOCK_INTS>& block, const uint32_t v, uint32_t &w, const uint32_t x, const uint32_t y, uint32_t &z, const size_t i) noexcept {
		z += ((w&(x^y))^y) + block[i] + SHA1_R0_R1_CONSTANT + wlib::bop::rotate_left(v, 5);
		w = wlib::bop::rotate_left(w, 30);
	}

	void SHA1::R1(std::array<uint32_t, SHA1_BLOCK_INTS>& block, const uint32_t v, uint32_t &w, const uint32_t x, const uint32_t y, uint32_t &z, const size_t i) noexcept {
		block[i] = blk(block, i);
		z += ((w&(x^y))^y) + block[i] + SHA1_R0_R1_CONSTANT + wlib::bop::rotate_left(v, 5);
		w = wlib::bop::rotate_left(w, 30);
	}

	void SHA1::R2(std::array<uint32_t, SHA1_BLOCK_INTS>& block, const uint32_t v, uint32_t &w, const uint32_t x, const uint32_t y, uint32_t &z, const size_t i) noexcept {
		block[i] = blk(block, i);
		z += (w^x^y) + block[i] + SHA1_R2_CONSTANT + wlib::bop::rotate_left(v, 5);
		w = wlib::bop::rotate_left(w, 30);
	}

	void SHA1::R3(std::array<uint32_t, SHA1_BLOCK_INTS>& block, const uint32_t v, uint32_t &w, const uint32_t x, const uint32_t y, uint32_t &z, const size_t i) noexcept {
		block[i] = blk(block, i);
		z += (((w|x)&y)|(w&x)) + block[i] + SHA1_R3_CONSTANT + wlib::bop::rotate_left(v, 5);
		w = wlib::bop::rotate_left(w, 30);
	}

	void SHA1::R4(std::array<uint32_t, SHA1_BLOCK_INTS>& block, const uint32_t v, uint32_t &w, const uint32_t x, const uint32_t y, uint32_t &z, const size_t i) noexcept {
		block[i] = blk(block, i);
		z += (w^x^y) + block[i] + SHA1_R4_CONSTANT + wlib::bop::rotate_left(v, 5);
		w = wlib::bop::rotate_left(w, 30);
	}

	void SHA1::transform(std::array<uint32_t, SHA1_BLOCK_INTS>& block) noexcept {
		uint32_t a = digest[0];
		uint32_t b = digest[1];
		uint32_t c = digest[2];
		uint32_t d = digest[3];
		uint32_t e = digest[4];

		R0(block, a, b, c, d, e,  0);
		R0(block, e, a, b, c, d,  1);
		R0(block, d, e, a, b, c,  2);
		R0(block, c, d, e, a, b,  3);
		R0(block, b, c, d, e, a,  4);
		R0(block, a, b, c, d, e,  5);
		R0(block, e, a, b, c, d,  6);
		R0(block, d, e, a, b, c,  7);
		R0(block, c, d, e, a, b,  8);
		R0(block, b, c, d, e, a,  9);
		R0(block, a, b, c, d, e, 10);
		R0(block, e, a, b, c, d, 11);
		R0(block, d, e, a, b, c, 12);
		R0(block, c, d, e, a, b, 13);
		R0(block, b, c, d, e, a, 14);
		R0(block, a, b, c, d, e, 15);
		R1(block, e, a, b, c, d,  0);
		R1(block, d, e, a, b, c,  1);
		R1(block, c, d, e, a, b,  2);
		R1(block, b, c, d, e, a,  3);
		R2(block, a, b, c, d, e,  4);
		R2(block, e, a, b, c, d,  5);
		R2(block, d, e, a, b, c,  6);
		R2(block, c, d, e, a, b,  7);
		R2(block, b, c, d, e, a,  8);
		R2(block, a, b, c, d, e,  9);
		R2(block, e, a, b, c, d, 10);
		R2(block, d, e, a, b, c, 11);
		R2(block, c, d, e, a, b, 12);
		R2(block, b, c, d, e, a, 13);
		R2(block, a, b, c, d, e, 14);
		R2(block, e, a, b, c, d, 15);
		R2(block, d, e, a, b, c,  0);
		R2(block, c, d, e, a, b,  1);
		R2(block, b, c, d, e, a,  2);
		R2(block, a, b, c, d, e,  3);
		R2(block, e, a, b, c, d,  4);
		R2(block, d, e, a, b, c,  5);
		R2(block, c, d, e, a, b,  6);
		R2(block, b, c, d, e, a,  7);
		R3(block, a, b, c, d, e,  8);
		R3(block, e, a, b, c, d,  9);
		R3(block, d, e, a, b, c, 10);
		R3(block, c, d, e, a, b, 11);
		R3(block, b, c, d, e, a, 12);
		R3(block, a, b, c, d, e, 13);
		R3(block, e, a, b, c, d, 14);
		R3(block, d, e, a, b, c, 15);
		R3(block, c, d, e, a, b,  0);
		R3(block, b, c, d, e, a,  1);
		R3(block, a, b, c, d, e,  2);
		R3(block, e, a, b, c, d,  3);
		R3(block, d, e, a, b, c,  4);
		R3(block, c, d, e, a, b,  5);
		R3(block, b, c, d, e, a,  6);
		R3(block, a, b, c, d, e,  7);
		R3(block, e, a, b, c, d,  8);
		R3(block, d, e, a, b, c,  9);
		R3(block, c, d, e, a, b, 10);
		R3(block, b, c, d, e, a, 11);
		R4(block, a, b, c, d, e, 12);
		R4(block, e, a, b, c, d, 13);
		R4(block, d, e, a, b, c, 14);
		R4(block, c, d, e, a, b, 15);
		R4(block, b, c, d, e, a,  0);
		R4(block, a, b, c, d, e,  1);
		R4(block, e, a, b, c, d,  2);
		R4(block, d, e, a, b, c,  3);
		R4(block, c, d, e, a, b,  4);
		R4(block, b, c, d, e, a,  5);
		R4(block, a, b, c, d, e,  6);
		R4(block, e, a, b, c, d,  7);
		R4(block, d, e, a, b, c,  8);
		R4(block, c, d, e, a, b,  9);
		R4(block, b, c, d, e, a, 10);
		R4(block, a, b, c, d, e, 11);
		R4(block, e, a, b, c, d, 12);
		R4(block, d, e, a, b, c, 13);
		R4(block, c, d, e, a, b, 14);
		R4(block, b, c, d, e, a, 15);

		digest[0] += a;
		digest[1] += b;
		digest[2] += c;
		digest[3] += d;
		digest[4] += e;

		++transform_count;
	}

} // namespace wlib::crypto
#endif
