#include <gtest/gtest.h>
#include "../include/weirdlib.hpp"

TEST(BitOps, bit_test) {
	const uint8_t x = 0b11101111;
	const uint8_t y = 0b11111111;

	EXPECT_FALSE(wlib::bop::test(x, 4));
	EXPECT_TRUE(wlib::bop::test(y, 4));
}

TEST(BitOps, bit_set) {
	const uint8_t x = 0b11101111;
	const uint8_t y = 0b11111111;

	EXPECT_EQ(wlib::bop::set(x, 4), 0b11111111);
	EXPECT_EQ(wlib::bop::set(y, 4), 0b11111111);
}

TEST(BitOps, bit_reset) {
	const uint8_t x = 0b11111111;
	const uint8_t y = 0b11101111;

	EXPECT_EQ(wlib::bop::reset(x, 4), 0b11101111);
	EXPECT_EQ(wlib::bop::reset(y, 4), 0b11101111);
}

TEST(BitOps, bit_toggle) {
	const uint8_t x = 0b11101111;
	const uint8_t y = 0b11111111;

	EXPECT_EQ(wlib::bop::toggle(x, 4), 0b11111111);
	EXPECT_EQ(wlib::bop::toggle(y, 4), 0b11101111);
}



TEST(BitOps, bit_set_ip) {
	uint8_t x = 0b11101111;
	uint8_t y = 0b11111111;

	wlib::bop::set_ip(x, 4);
	wlib::bop::set_ip(y, 4);

	EXPECT_EQ(x, 0b11111111);
	EXPECT_EQ(y, 0b11111111);
}

TEST(BitOps, bit_reset_ip) {
	uint8_t x = 0b11111111;
	uint8_t y = 0b11101111;

	wlib::bop::reset_ip(x, 4);
	wlib::bop::reset_ip(y, 4);

	EXPECT_EQ(x, 0b11101111);
	EXPECT_EQ(y, 0b11101111);
}

TEST(BitOps, bit_toggle_ip) {
	uint8_t x = 0b11101111;
	uint8_t y = 0b11111111;

	wlib::bop::toggle_ip(x, 4);
	wlib::bop::toggle_ip(y, 4);

	EXPECT_EQ(x, 0b11111111);
	EXPECT_EQ(y, 0b11101111);
}



TEST(BitOps, ctz_8bit) {
	const uint8_t x = 0xF0;
	const uint8_t y = 0x0F;
	const uint8_t z = 0xFF;
	const uint8_t w = 0x00;
	const uint8_t s = 0x80;

	EXPECT_EQ(wlib::bop::ctz(x), 4);
	EXPECT_EQ(wlib::bop::count_trailing_zeros(x), 4);

	EXPECT_EQ(wlib::bop::ctz(y), 0);
	EXPECT_EQ(wlib::bop::count_trailing_zeros(y), 0);

	EXPECT_EQ(wlib::bop::ctz(z), 0);
	EXPECT_EQ(wlib::bop::count_trailing_zeros(z), 0);

	EXPECT_EQ(wlib::bop::ctz(w), sizeof(uint8_t) * 8);
	EXPECT_EQ(wlib::bop::count_trailing_zeros(w), sizeof(uint8_t) * 8);

	EXPECT_EQ(wlib::bop::ctz(s), 7);
	EXPECT_EQ(wlib::bop::count_trailing_zeros(s), 7);
}

TEST(BitOps, ctz_16bit) {
	const uint16_t x = 0x0FF0;
	const uint16_t y = 0xFFF0;
	const uint16_t z = 0xFFFF;
	const uint16_t w = 0x0000;
	const uint16_t s = 0x0F00;

	EXPECT_EQ(wlib::bop::ctz(x), 4);
	EXPECT_EQ(wlib::bop::count_trailing_zeros(x), 4);

	EXPECT_EQ(wlib::bop::ctz(y), 4);
	EXPECT_EQ(wlib::bop::count_trailing_zeros(y), 4);

	EXPECT_EQ(wlib::bop::ctz(z), 0);
	EXPECT_EQ(wlib::bop::count_trailing_zeros(z), 0);

	EXPECT_EQ(wlib::bop::ctz(w), sizeof(uint16_t) * 8);
	EXPECT_EQ(wlib::bop::count_trailing_zeros(w), sizeof(uint16_t) * 8);

	EXPECT_EQ(wlib::bop::ctz(s), 8);
	EXPECT_EQ(wlib::bop::count_trailing_zeros(s), 8);
}

TEST(BitOps, ctz_32bit) {
	const uint32_t x = 0x0FFFFFF0;
	const uint32_t y = 0xFFFFFFF0;
	const uint32_t z = 0xFFFFFFFF;
	const uint32_t w = 0x00000000;
	const uint32_t s = 0x0FFFF000;

	EXPECT_EQ(wlib::bop::ctz(x), 4);
	EXPECT_EQ(wlib::bop::count_trailing_zeros(x), 4);

	EXPECT_EQ(wlib::bop::ctz(y), 4);
	EXPECT_EQ(wlib::bop::count_trailing_zeros(y), 4);

	EXPECT_EQ(wlib::bop::ctz(z), 0);
	EXPECT_EQ(wlib::bop::count_trailing_zeros(z), 0);

	EXPECT_EQ(wlib::bop::ctz(s), 12);
	EXPECT_EQ(wlib::bop::count_trailing_zeros(s), 12);

	EXPECT_EQ(wlib::bop::ctz(w), sizeof(uint32_t) * 8);
	EXPECT_EQ(wlib::bop::count_trailing_zeros(w), sizeof(uint32_t) * 8);
}

TEST(BitOps, ctz_64bit) {
	const uint64_t x = 0x0FFFFFFFFFFFFFF0;
	const uint64_t y = 0xFFFFFFFFFFFFFFF0;
	const uint64_t z = 0xFFFFFFFFFFFFFFFF;
	const uint64_t w = 0x0000000000000000;
	const uint64_t s = 0xFFFFFFFFFF000000;

	EXPECT_EQ(wlib::bop::ctz(x), 4);
	EXPECT_EQ(wlib::bop::count_trailing_zeros(x), 4);

	EXPECT_EQ(wlib::bop::ctz(y), 4);
	EXPECT_EQ(wlib::bop::count_trailing_zeros(y), 4);

	EXPECT_EQ(wlib::bop::ctz(z), 0);
	EXPECT_EQ(wlib::bop::count_trailing_zeros(z), 0);

	EXPECT_EQ(wlib::bop::ctz(s), 24);
	EXPECT_EQ(wlib::bop::count_trailing_zeros(s), 24);

	EXPECT_EQ(wlib::bop::ctz(w), sizeof(uint64_t) * 8);
	EXPECT_EQ(wlib::bop::count_trailing_zeros(w), sizeof(uint64_t) * 8);
}



TEST(BitOps, clz_8bit) {
	const uint8_t x = 0xF0;
	const uint8_t y = 0x0F;
	const uint8_t z = 0xFF;
	const uint8_t w = 0x00;

	EXPECT_EQ(wlib::bop::clz(x), 0);
	EXPECT_EQ(wlib::bop::count_leading_zeros(x), 0);

	EXPECT_EQ(wlib::bop::clz(y), 4);
	EXPECT_EQ(wlib::bop::count_leading_zeros(y), 4);

	EXPECT_EQ(wlib::bop::clz(z), 0);
	EXPECT_EQ(wlib::bop::count_leading_zeros(z), 0);

	EXPECT_EQ(wlib::bop::ctz(w), sizeof(uint8_t) * 8);
	EXPECT_EQ(wlib::bop::count_trailing_zeros(w), sizeof(uint8_t) * 8);
}

TEST(BitOps, clz_16bit) {
	const uint16_t x = 0x0FF0;
	const uint16_t y = 0x0FFF;
	const uint16_t z = 0xFFFF;
	const uint16_t w = 0x0000;
	const uint16_t s = 0x00FF;

	EXPECT_EQ(wlib::bop::clz(x), 4);
	EXPECT_EQ(wlib::bop::count_leading_zeros(x), 4);

	EXPECT_EQ(wlib::bop::clz(y), 4);
	EXPECT_EQ(wlib::bop::count_leading_zeros(y), 4);

	EXPECT_EQ(wlib::bop::clz(z), 0);
	EXPECT_EQ(wlib::bop::count_leading_zeros(z), 0);

	EXPECT_EQ(wlib::bop::clz(s), 8);
	EXPECT_EQ(wlib::bop::count_leading_zeros(s), 8);

	EXPECT_EQ(wlib::bop::ctz(w), sizeof(uint16_t) * 8);
	EXPECT_EQ(wlib::bop::count_trailing_zeros(w), sizeof(uint16_t) * 8);
}

TEST(BitOps, clz_32bit) {
	const uint32_t x = 0x0FFFFFF0;
	const uint32_t y = 0x0FFFFFFF;
	const uint32_t z = 0xFFFFFFFF;
	const uint32_t w = 0x00000000;
	const uint32_t s = 0x000FFFFF;

	EXPECT_EQ(wlib::bop::clz(x), 4);
	EXPECT_EQ(wlib::bop::count_leading_zeros(x), 4);

	EXPECT_EQ(wlib::bop::clz(y), 4);
	EXPECT_EQ(wlib::bop::count_leading_zeros(y), 4);

	EXPECT_EQ(wlib::bop::clz(z), 0);
	EXPECT_EQ(wlib::bop::count_leading_zeros(z), 0);

	EXPECT_EQ(wlib::bop::clz(s), 12);
	EXPECT_EQ(wlib::bop::count_leading_zeros(s), 12);

	EXPECT_EQ(wlib::bop::ctz(w), sizeof(uint32_t) * 8);
	EXPECT_EQ(wlib::bop::count_trailing_zeros(w), sizeof(uint32_t) * 8);
}

TEST(BitOps, clz_64bit) {
	const uint64_t x = 0x0FFFFFFFFFFFFFF0;
	const uint64_t y = 0x0FFFFFFFFFFFFFFF;
	const uint64_t z = 0xFFFFFFFFFFFFFFFF;
	const uint64_t w = 0x0000000000000000;
	const uint64_t s = 0x00000FFFFFFFFFFF;

	EXPECT_EQ(wlib::bop::clz(x), 4);
	EXPECT_EQ(wlib::bop::count_leading_zeros(x), 4);

	EXPECT_EQ(wlib::bop::clz(y), 4);
	EXPECT_EQ(wlib::bop::count_leading_zeros(y), 4);

	EXPECT_EQ(wlib::bop::clz(z), 0);
	EXPECT_EQ(wlib::bop::count_leading_zeros(z), 0);

	EXPECT_EQ(wlib::bop::clz(s), 20);
	EXPECT_EQ(wlib::bop::count_leading_zeros(s), 20);

	EXPECT_EQ(wlib::bop::ctz(w), sizeof(uint64_t) * 8);
	EXPECT_EQ(wlib::bop::count_trailing_zeros(w), sizeof(uint64_t) * 8);
}



TEST(BitOps, bit_scan_forward_8bit) {
	const uint8_t x = 0x0F;
	const uint8_t y = 0xF0;
	const uint8_t z = 0xFF;
	const uint8_t w = 0x0000;

	EXPECT_EQ(wlib::bop::bsf(x), 0);
	EXPECT_EQ(wlib::bop::bit_scan_forward(x), 0);

	EXPECT_EQ(wlib::bop::bsf(y), 4);
	EXPECT_EQ(wlib::bop::bit_scan_forward(y), 4);

	EXPECT_EQ(wlib::bop::bsf(z), 0);
	EXPECT_EQ(wlib::bop::bit_scan_forward(z), 0);

	EXPECT_EQ(wlib::bop::bsf(w), 0);
	EXPECT_EQ(wlib::bop::bit_scan_forward(w), 0);
}

TEST(BitOps, bit_scan_forward_16bit) {
	const uint16_t x = 0x0FF0;
	const uint16_t y = 0xFFF0;
	const uint16_t z = 0xFFFF;
	const uint16_t w = 0x0000;
	const uint16_t s = 0x0F00;

	EXPECT_EQ(wlib::bop::bsf(x), 4);
	EXPECT_EQ(wlib::bop::bit_scan_forward(x), 4);

	EXPECT_EQ(wlib::bop::bsf(y), 4);
	EXPECT_EQ(wlib::bop::bit_scan_forward(y), 4);

	EXPECT_EQ(wlib::bop::bsf(z), 0);
	EXPECT_EQ(wlib::bop::bit_scan_forward(z), 0);

	EXPECT_EQ(wlib::bop::bsf(s), 8);
	EXPECT_EQ(wlib::bop::bit_scan_forward(s), 8);

	EXPECT_EQ(wlib::bop::bsf(w), 0);
	EXPECT_EQ(wlib::bop::bit_scan_forward(w), 0);
}

TEST(BitOps, bit_scan_forward_32bit) {
	const uint32_t x = 0x0FFFFFF0;
	const uint32_t y = 0xFFFFFFF0;
	const uint32_t z = 0xFFFFFFFF;
	const uint32_t w = 0x00000000;
	const uint32_t s = 0x0FFFF000;

	EXPECT_EQ(wlib::bop::bsf(x), 4);
	EXPECT_EQ(wlib::bop::bit_scan_forward(x), 4);

	EXPECT_EQ(wlib::bop::bsf(y), 4);
	EXPECT_EQ(wlib::bop::bit_scan_forward(y), 4);

	EXPECT_EQ(wlib::bop::bsf(z), 0);
	EXPECT_EQ(wlib::bop::bit_scan_forward(z), 0);

	EXPECT_EQ(wlib::bop::bsf(s), 12);
	EXPECT_EQ(wlib::bop::bit_scan_forward(s), 12);

	EXPECT_EQ(wlib::bop::bsf(w), 0);
	EXPECT_EQ(wlib::bop::bit_scan_forward(w), 0);
}

TEST(BitOps, bit_scan_forward_64bit) {
	const uint64_t x = 0x0FFFFFFFFFFFFFF0;
	const uint64_t y = 0xFFFFFFFFFFFFFFF0;
	const uint64_t z = 0xFFFFFFFFFFFFFFFF;
	const uint64_t w = 0x0000000000000000;
	const uint64_t s = 0xFFFFFFFFFF000000;

	EXPECT_EQ(wlib::bop::bsf(x), 4);
	EXPECT_EQ(wlib::bop::bit_scan_forward(x), 4);

	EXPECT_EQ(wlib::bop::bsf(y), 4);
	EXPECT_EQ(wlib::bop::bit_scan_forward(y), 4);

	EXPECT_EQ(wlib::bop::bsf(z), 0);
	EXPECT_EQ(wlib::bop::bit_scan_forward(z), 0);

	EXPECT_EQ(wlib::bop::bsf(s), 24);
	EXPECT_EQ(wlib::bop::bit_scan_forward(s), 24);

	EXPECT_EQ(wlib::bop::bsf(w), 0);
	EXPECT_EQ(wlib::bop::bit_scan_forward(w), 0);
}



TEST(BitOps, bit_scan_reverse_8bit) {
	const uint8_t x = 0xF0;
	const uint8_t y = 0x0F;
	const uint8_t z = 0xFF;
	const uint8_t w = 0x00;

	EXPECT_EQ(wlib::bop::bsr(x), 7);
	EXPECT_EQ(wlib::bop::bit_scan_reverse(x), 7);

	EXPECT_EQ(wlib::bop::bsr(y), 3);
	EXPECT_EQ(wlib::bop::bit_scan_reverse(y), 3);

	EXPECT_EQ(wlib::bop::bsr(z), 7);
	EXPECT_EQ(wlib::bop::bit_scan_reverse(z), 7);

	EXPECT_EQ(wlib::bop::bsr(w), 0);
	EXPECT_EQ(wlib::bop::bit_scan_reverse(w), 0);
}

TEST(BitOps, bit_scan_reverse_16bit) {
	const uint16_t x = 0x0FF0;
	const uint16_t y = 0x0FFF;
	const uint16_t z = 0xFFFF;
	const uint16_t w = 0x0000;

	EXPECT_EQ(wlib::bop::bsr(x), 11);
	EXPECT_EQ(wlib::bop::bit_scan_reverse(x), 11);

	EXPECT_EQ(wlib::bop::bsr(y), 11);
	EXPECT_EQ(wlib::bop::bit_scan_reverse(y), 11);

	EXPECT_EQ(wlib::bop::bsr(z), 15);
	EXPECT_EQ(wlib::bop::bit_scan_reverse(z), 15);

	EXPECT_EQ(wlib::bop::bsr(w), 0);
	EXPECT_EQ(wlib::bop::bit_scan_reverse(w), 0);
}

TEST(BitOps, bit_scan_reverse_32bit) {
	const uint32_t x = 0x0FFFFFF0;
	const uint32_t y = 0x0FFFFFFF;
	const uint32_t z = 0xFFFFFFFF;
	const uint32_t w = 0x00000000;

	EXPECT_EQ(wlib::bop::bsr(x), 27);
	EXPECT_EQ(wlib::bop::bit_scan_reverse(x), 27);

	EXPECT_EQ(wlib::bop::bsr(y), 27);
	EXPECT_EQ(wlib::bop::bit_scan_reverse(y), 27);

	EXPECT_EQ(wlib::bop::bsr(z), 31);
	EXPECT_EQ(wlib::bop::bit_scan_reverse(z), 31);

	EXPECT_EQ(wlib::bop::bsr(w), 0);
	EXPECT_EQ(wlib::bop::bit_scan_reverse(w), 0);
}

TEST(BitOps, bit_scan_reverse_64bit) {
	const uint64_t x = 0x0FFFFFFFFFFFFFF0;
	const uint64_t y = 0x0FFFFFFFFFFFFFFF;
	const uint64_t z = 0xFFFFFFFFFFFFFFFF;
	const uint64_t w = 0x0000000000000000;

	EXPECT_EQ(wlib::bop::bsr(x), 59);
	EXPECT_EQ(wlib::bop::bit_scan_reverse(x), 59);

	EXPECT_EQ(wlib::bop::bsr(y), 59);
	EXPECT_EQ(wlib::bop::bit_scan_reverse(y), 59);

	EXPECT_EQ(wlib::bop::bsr(z), 63);
	EXPECT_EQ(wlib::bop::bit_scan_reverse(z), 63);

	EXPECT_EQ(wlib::bop::bsr(w), 0);
	EXPECT_EQ(wlib::bop::bit_scan_reverse(w), 0);
}



TEST(BitOps, popcnt_8bit) {
	const uint8_t x = 0xFF;
	const uint8_t y = 0x0F;
	const uint8_t z = 0x00;

	EXPECT_EQ(wlib::bop::popcnt(x), 8);
	EXPECT_EQ(wlib::bop::population_count(x), 8);

	EXPECT_EQ(wlib::bop::popcnt(y), 4);
	EXPECT_EQ(wlib::bop::population_count(y), 4);

	EXPECT_EQ(wlib::bop::popcnt(z), 0);
	EXPECT_EQ(wlib::bop::population_count(z), 0);
}

TEST(BitOps, popcnt_16bit) {
	const uint16_t x = 0xFFFF;
	const uint16_t y = 0x0F0F;
	const uint16_t z = 0x0000;

	EXPECT_EQ(wlib::bop::popcnt(x), 16);
	EXPECT_EQ(wlib::bop::population_count(x), 16);

	EXPECT_EQ(wlib::bop::popcnt(y), 8);
	EXPECT_EQ(wlib::bop::population_count(y), 8);

	EXPECT_EQ(wlib::bop::popcnt(z), 0);
	EXPECT_EQ(wlib::bop::population_count(z), 0);
}

TEST(BitOps, popcnt_32bit) {
	const uint32_t x = 0xFFFFFFFF;
	const uint32_t y = 0x0FFFF0FF;
	const uint32_t z = 0x00000000;

	EXPECT_EQ(wlib::bop::popcnt(x), 32);
	EXPECT_EQ(wlib::bop::population_count(x), 32);

	EXPECT_EQ(wlib::bop::popcnt(y), 24);
	EXPECT_EQ(wlib::bop::population_count(y), 24);

	EXPECT_EQ(wlib::bop::popcnt(z), 0);
	EXPECT_EQ(wlib::bop::population_count(z), 0);
}

TEST(BitOps, popcnt_64bit) {
	const uint64_t x = 0xFFFFFFFFFFFFFFFF;
	const uint64_t y = 0x0FFFFFF0000FF0FF;
	const uint64_t z = 0x0000000000000000;

	EXPECT_EQ(wlib::bop::popcnt(x), 64);
	EXPECT_EQ(wlib::bop::population_count(x), 64);

	EXPECT_EQ(wlib::bop::popcnt(y), 40);
	EXPECT_EQ(wlib::bop::population_count(y), 40);

	EXPECT_EQ(wlib::bop::popcnt(z), 0);
	EXPECT_EQ(wlib::bop::population_count(z), 0);
}



TEST(BitOps, rotate_left_8bit) {
	const uint8_t x = 0xFF;
	const uint8_t y = 0x48;
	const uint8_t z = 0x00;

	EXPECT_EQ(wlib::bop::rotate_left(x, 4), 0xFF);
	EXPECT_EQ(wlib::bop::rotate_left(y, 4), 0x84);
	EXPECT_EQ(wlib::bop::rotate_left(z, 4), 0x00);
}

TEST(BitOps, rotate_left_16bit) {
	const uint16_t x = 0xFFFF;
	const uint16_t y = 0x0F0F;
	const uint16_t z = 0x0000;

	EXPECT_EQ(wlib::bop::rotate_left(x, 4), 0xFFFF);
	EXPECT_EQ(wlib::bop::rotate_left(y, 4), 0xF0F0);
	EXPECT_EQ(wlib::bop::rotate_left(z, 4), 0x0000);
}

TEST(BitOps, rotate_left_32bit) {
	const uint32_t x = 0xFFFFFFFF;
	const uint32_t y = 0x0FFFF0FF;
	const uint32_t z = 0x00000000;

	EXPECT_EQ(wlib::bop::rotate_left(x, 4), 0xFFFFFFFF);
	EXPECT_EQ(wlib::bop::rotate_left(y, 4), 0xFFFF0FF0);
	EXPECT_EQ(wlib::bop::rotate_left(z, 4), 0x00000000);
}

TEST(BitOps, rotate_left_64bit) {
	const uint64_t x = 0xFFFFFFFFFFFFFFFF;
	const uint64_t y = 0x0FFFFFFFF0FFFFFF;
	const uint64_t z = 0x0000000000000000;

	EXPECT_EQ(wlib::bop::rotate_left(x, 4), 0xFFFFFFFFFFFFFFFF);
	EXPECT_EQ(wlib::bop::rotate_left(y, 4), 0xFFFFFFFF0FFFFFF0);
	EXPECT_EQ(wlib::bop::rotate_left(z, 4), 0x0000000000000000);
}



TEST(BitOps, rotate_right_8bit) {
	const uint8_t x = 0xFF;
	const uint8_t y = 0x48;
	const uint8_t z = 0x00;

	EXPECT_EQ(wlib::bop::rotate_right(x, 4), 0xFF);
	EXPECT_EQ(wlib::bop::rotate_right(y, 4), 0x84);
	EXPECT_EQ(wlib::bop::rotate_right(z, 4), 0x00);
}

TEST(BitOps, rotate_right_16bit) {
	const uint16_t x = 0xFFFF;
	const uint16_t y = 0x0F0F;
	const uint16_t z = 0x0000;

	EXPECT_EQ(wlib::bop::rotate_right(x, 4), 0xFFFF);
	EXPECT_EQ(wlib::bop::rotate_right(y, 4), 0xF0F0);
	EXPECT_EQ(wlib::bop::rotate_right(z, 4), 0x0000);
}

TEST(BitOps, rotate_right_32bit) {
	const uint32_t x = 0xFFFFFFFF;
	const uint32_t y = 0x0FFFF0FF;
	const uint32_t z = 0x00000000;

	EXPECT_EQ(wlib::bop::rotate_right(x, 4), 0xFFFFFFFF);
	EXPECT_EQ(wlib::bop::rotate_right(y, 4), 0xF0FFFF0F);
	EXPECT_EQ(wlib::bop::rotate_right(z, 4), 0x00000000);
}

TEST(BitOps, rotate_right_64bit) {
	const uint64_t x = 0xFFFFFFFFFFFFFFFF;
	const uint64_t y = 0x0FFFFFFFF0FFFFFF;
	const uint64_t z = 0x0000000000000000;

	EXPECT_EQ(wlib::bop::rotate_right(x, 4), 0xFFFFFFFFFFFFFFFF);
	EXPECT_EQ(wlib::bop::rotate_right(y, 4), 0xF0FFFFFFFF0FFFFF);
	EXPECT_EQ(wlib::bop::rotate_right(z, 4), 0x0000000000000000);
}
