#include <gtest/gtest.h>
#include "../include/weirdlib.hpp"


TEST(BitOps, Test) {
	const uint8_t x = 0b11101111;
	const uint8_t y = 0b11111111;

	EXPECT_FALSE(wlib::bop::test(x, 4));
	EXPECT_TRUE(wlib::bop::test(y, 4));
}

TEST(BitOps, Set) {
	const uint8_t x = 0b11101111;
	const uint8_t y = 0b11111111;

	EXPECT_EQ(wlib::bop::set(x, 4), 0b11111111);
	EXPECT_EQ(wlib::bop::set(y, 4), 0b11111111);
}

TEST(BitOps, Reset) {
	const uint8_t x = 0b11111111;
	const uint8_t y = 0b11101111;

	EXPECT_EQ(wlib::bop::reset(x, 4), 0b11101111);
	EXPECT_EQ(wlib::bop::reset(y, 4), 0b11101111);
}

TEST(BitOps, Toggle) {
	const uint8_t x = 0b11101111;
	const uint8_t y = 0b11111111;

	EXPECT_EQ(wlib::bop::toggle(x, 4), 0b11111111);
	EXPECT_EQ(wlib::bop::toggle(y, 4), 0b11101111);
}


TEST(BitOps, SetIP) {
	uint8_t x = 0b11101111;
	uint8_t y = 0b11111111;

	wlib::bop::set_ip(x, 4);
	wlib::bop::set_ip(y, 4);

	EXPECT_EQ(x, 0b11111111);
	EXPECT_EQ(y, 0b11111111);
}

TEST(BitOps, ResetIP) {
	uint8_t x = 0b11111111;
	uint8_t y = 0b11101111;

	wlib::bop::reset_ip(x, 4);
	wlib::bop::reset_ip(y, 4);

	EXPECT_EQ(x, 0b11101111);
	EXPECT_EQ(y, 0b11101111);
}

TEST(BitOps, ToggleIP) {
	uint8_t x = 0b11101111;
	uint8_t y = 0b11111111;

	wlib::bop::toggle_ip(x, 4);
	wlib::bop::toggle_ip(y, 4);

	EXPECT_EQ(x, 0b11111111);
	EXPECT_EQ(y, 0b11101111);
}


TEST(BitOps, CTZ) {
	const uint8_t x_8bit = 0xF0;
	const uint8_t y_8bit = 0x0F;
	const uint8_t z_8bit = 0xFF;
	const uint8_t w_8bit = 0x00;
	const uint8_t s_8bit = 0x80;

	EXPECT_EQ(wlib::bop::ctz(x_8bit), 4);
	EXPECT_EQ(wlib::bop::count_trailing_zeros(x_8bit), 4);

	EXPECT_EQ(wlib::bop::ctz(y_8bit), 0);
	EXPECT_EQ(wlib::bop::count_trailing_zeros(y_8bit), 0);

	EXPECT_EQ(wlib::bop::ctz(z_8bit), 0);
	EXPECT_EQ(wlib::bop::count_trailing_zeros(z_8bit), 0);

	EXPECT_EQ(wlib::bop::ctz(w_8bit), sizeof(uint8_t) * 8);
	EXPECT_EQ(wlib::bop::count_trailing_zeros(w_8bit), sizeof(uint8_t) * 8);

	EXPECT_EQ(wlib::bop::ctz(s_8bit), 7);
	EXPECT_EQ(wlib::bop::count_trailing_zeros(s_8bit), 7);


	const uint16_t x_16bit = 0x0FF0;
	const uint16_t y_16bit = 0xFFF0;
	const uint16_t z_16bit = 0xFFFF;
	const uint16_t w_16bit = 0x0000;
	const uint16_t s_16bit = 0x0F00;

	EXPECT_EQ(wlib::bop::ctz(x_16bit), 4);
	EXPECT_EQ(wlib::bop::count_trailing_zeros(x_16bit), 4);

	EXPECT_EQ(wlib::bop::ctz(y_16bit), 4);
	EXPECT_EQ(wlib::bop::count_trailing_zeros(y_16bit), 4);

	EXPECT_EQ(wlib::bop::ctz(z_16bit), 0);
	EXPECT_EQ(wlib::bop::count_trailing_zeros(z_16bit), 0);

	EXPECT_EQ(wlib::bop::ctz(w_16bit), sizeof(uint16_t) * 8);
	EXPECT_EQ(wlib::bop::count_trailing_zeros(w_16bit), sizeof(uint16_t) * 8);

	EXPECT_EQ(wlib::bop::ctz(s_16bit), 8);
	EXPECT_EQ(wlib::bop::count_trailing_zeros(s_16bit), 8);


	const uint32_t x_32bit = 0x0FFFFFF0;
	const uint32_t y_32bit = 0xFFFFFFF0;
	const uint32_t z_32bit = 0xFFFFFFFF;
	const uint32_t w_32bit = 0x00000000;
	const uint32_t s_32bit = 0x0FFFF000;

	EXPECT_EQ(wlib::bop::ctz(x_32bit), 4);
	EXPECT_EQ(wlib::bop::count_trailing_zeros(x_32bit), 4);

	EXPECT_EQ(wlib::bop::ctz(y_32bit), 4);
	EXPECT_EQ(wlib::bop::count_trailing_zeros(y_32bit), 4);

	EXPECT_EQ(wlib::bop::ctz(z_32bit), 0);
	EXPECT_EQ(wlib::bop::count_trailing_zeros(z_32bit), 0);

	EXPECT_EQ(wlib::bop::ctz(s_32bit), 12);
	EXPECT_EQ(wlib::bop::count_trailing_zeros(s_32bit), 12);

	EXPECT_EQ(wlib::bop::ctz(w_32bit), sizeof(uint32_t) * 8);
	EXPECT_EQ(wlib::bop::count_trailing_zeros(w_32bit), sizeof(uint32_t) * 8);


	const uint64_t x_64bit = 0x0FFFFFFFFFFFFFF0;
	const uint64_t y_64bit = 0xFFFFFFFFFFFFFFF0;
	const uint64_t z_64bit = 0xFFFFFFFFFFFFFFFF;
	const uint64_t w_64bit = 0x0000000000000000;
	const uint64_t s_64bit = 0xFFFFFFFFFF000000;

	EXPECT_EQ(wlib::bop::ctz(x_64bit), 4);
	EXPECT_EQ(wlib::bop::count_trailing_zeros(x_64bit), 4);

	EXPECT_EQ(wlib::bop::ctz(y_64bit), 4);
	EXPECT_EQ(wlib::bop::count_trailing_zeros(y_64bit), 4);

	EXPECT_EQ(wlib::bop::ctz(z_64bit), 0);
	EXPECT_EQ(wlib::bop::count_trailing_zeros(z_64bit), 0);

	EXPECT_EQ(wlib::bop::ctz(s_64bit), 24);
	EXPECT_EQ(wlib::bop::count_trailing_zeros(s_64bit), 24);

	EXPECT_EQ(wlib::bop::ctz(w_64bit), sizeof(uint64_t) * 8);
	EXPECT_EQ(wlib::bop::count_trailing_zeros(w_64bit), sizeof(uint64_t) * 8);
}

TEST(BitOps, CLZ) {
	const uint8_t x_8bit = 0xF0;
	const uint8_t y_8bit = 0x0F;
	const uint8_t z_8bit = 0xFF;
	const uint8_t w_8bit = 0x00;

	EXPECT_EQ(wlib::bop::clz(x_8bit), 0);
	EXPECT_EQ(wlib::bop::count_leading_zeros(x_8bit), 0);

	EXPECT_EQ(wlib::bop::clz(y_8bit), 4);
	EXPECT_EQ(wlib::bop::count_leading_zeros(y_8bit), 4);

	EXPECT_EQ(wlib::bop::clz(z_8bit), 0);
	EXPECT_EQ(wlib::bop::count_leading_zeros(z_8bit), 0);

	EXPECT_EQ(wlib::bop::ctz(w_8bit), sizeof(uint8_t) * 8);
	EXPECT_EQ(wlib::bop::count_trailing_zeros(w_8bit), sizeof(uint8_t) * 8);


	const uint16_t x_16bit = 0x0FF0;
	const uint16_t y_16bit = 0x0FFF;
	const uint16_t z_16bit = 0xFFFF;
	const uint16_t w_16bit = 0x0000;
	const uint16_t s_16bit = 0x00FF;

	EXPECT_EQ(wlib::bop::clz(x_16bit), 4);
	EXPECT_EQ(wlib::bop::count_leading_zeros(x_16bit), 4);

	EXPECT_EQ(wlib::bop::clz(y_16bit), 4);
	EXPECT_EQ(wlib::bop::count_leading_zeros(y_16bit), 4);

	EXPECT_EQ(wlib::bop::clz(z_16bit), 0);
	EXPECT_EQ(wlib::bop::count_leading_zeros(z_16bit), 0);

	EXPECT_EQ(wlib::bop::clz(s_16bit), 8);
	EXPECT_EQ(wlib::bop::count_leading_zeros(s_16bit), 8);

	EXPECT_EQ(wlib::bop::ctz(w_16bit), sizeof(uint16_t) * 8);
	EXPECT_EQ(wlib::bop::count_trailing_zeros(w_16bit), sizeof(uint16_t) * 8);


	const uint32_t x_32bit = 0x0FFFFFF0;
	const uint32_t y_32bit = 0x0FFFFFFF;
	const uint32_t z_32bit = 0xFFFFFFFF;
	const uint32_t w_32bit = 0x00000000;
	const uint32_t s_32bit = 0x000FFFFF;

	EXPECT_EQ(wlib::bop::clz(x_32bit), 4);
	EXPECT_EQ(wlib::bop::count_leading_zeros(x_32bit), 4);

	EXPECT_EQ(wlib::bop::clz(y_32bit), 4);
	EXPECT_EQ(wlib::bop::count_leading_zeros(y_32bit), 4);

	EXPECT_EQ(wlib::bop::clz(z_32bit), 0);
	EXPECT_EQ(wlib::bop::count_leading_zeros(z_32bit), 0);

	EXPECT_EQ(wlib::bop::clz(s_32bit), 12);
	EXPECT_EQ(wlib::bop::count_leading_zeros(s_32bit), 12);

	EXPECT_EQ(wlib::bop::ctz(w_32bit), sizeof(uint32_t) * 8);
	EXPECT_EQ(wlib::bop::count_trailing_zeros(w_32bit), sizeof(uint32_t) * 8);


	const uint64_t x_64bit = 0x0FFFFFFFFFFFFFF0;
	const uint64_t y_64bit = 0x0FFFFFFFFFFFFFFF;
	const uint64_t z_64bit = 0xFFFFFFFFFFFFFFFF;
	const uint64_t w_64bit = 0x0000000000000000;
	const uint64_t s_64bit = 0x00000FFFFFFFFFFF;

	EXPECT_EQ(wlib::bop::clz(x_64bit), 4);
	EXPECT_EQ(wlib::bop::count_leading_zeros(x_64bit), 4);

	EXPECT_EQ(wlib::bop::clz(y_64bit), 4);
	EXPECT_EQ(wlib::bop::count_leading_zeros(y_64bit), 4);

	EXPECT_EQ(wlib::bop::clz(z_64bit), 0);
	EXPECT_EQ(wlib::bop::count_leading_zeros(z_64bit), 0);

	EXPECT_EQ(wlib::bop::clz(s_64bit), 20);
	EXPECT_EQ(wlib::bop::count_leading_zeros(s_64bit), 20);

	EXPECT_EQ(wlib::bop::ctz(w_64bit), sizeof(uint64_t) * 8);
	EXPECT_EQ(wlib::bop::count_trailing_zeros(w_64bit), sizeof(uint64_t) * 8);
}


TEST(BitOps, BSF) {
	const uint8_t x_8bit = 0x0F;
	const uint8_t y_8bit = 0xF0;
	const uint8_t z_8bit = 0xFF;
	const uint8_t w_8bit = 0x0000;

	EXPECT_EQ(wlib::bop::bsf(x_8bit), 0);
	EXPECT_EQ(wlib::bop::bit_scan_forward(x_8bit), 0);

	EXPECT_EQ(wlib::bop::bsf(y_8bit), 4);
	EXPECT_EQ(wlib::bop::bit_scan_forward(y_8bit), 4);

	EXPECT_EQ(wlib::bop::bsf(z_8bit), 0);
	EXPECT_EQ(wlib::bop::bit_scan_forward(z_8bit), 0);

	EXPECT_EQ(wlib::bop::bsf(w_8bit), 0);
	EXPECT_EQ(wlib::bop::bit_scan_forward(w_8bit), 0);


	const uint16_t x_16bit = 0x0FF0;
	const uint16_t y_16bit = 0xFFF0;
	const uint16_t z_16bit = 0xFFFF;
	const uint16_t w_16bit = 0x0000;
	const uint16_t s_16bit = 0x0F00;

	EXPECT_EQ(wlib::bop::bsf(x_16bit), 4);
	EXPECT_EQ(wlib::bop::bit_scan_forward(x_16bit), 4);

	EXPECT_EQ(wlib::bop::bsf(y_16bit), 4);
	EXPECT_EQ(wlib::bop::bit_scan_forward(y_16bit), 4);

	EXPECT_EQ(wlib::bop::bsf(z_16bit), 0);
	EXPECT_EQ(wlib::bop::bit_scan_forward(z_16bit), 0);

	EXPECT_EQ(wlib::bop::bsf(s_16bit), 8);
	EXPECT_EQ(wlib::bop::bit_scan_forward(s_16bit), 8);

	EXPECT_EQ(wlib::bop::bsf(w_16bit), 0);
	EXPECT_EQ(wlib::bop::bit_scan_forward(w_16bit), 0);


	const uint32_t x_32bit = 0x0FFFFFF0;
	const uint32_t y_32bit = 0xFFFFFFF0;
	const uint32_t z_32bit = 0xFFFFFFFF;
	const uint32_t w_32bit = 0x00000000;
	const uint32_t s_32bit = 0x0FFFF000;

	EXPECT_EQ(wlib::bop::bsf(x_32bit), 4);
	EXPECT_EQ(wlib::bop::bit_scan_forward(x_32bit), 4);

	EXPECT_EQ(wlib::bop::bsf(y_32bit), 4);
	EXPECT_EQ(wlib::bop::bit_scan_forward(y_32bit), 4);

	EXPECT_EQ(wlib::bop::bsf(z_32bit), 0);
	EXPECT_EQ(wlib::bop::bit_scan_forward(z_32bit), 0);

	EXPECT_EQ(wlib::bop::bsf(s_32bit), 12);
	EXPECT_EQ(wlib::bop::bit_scan_forward(s_32bit), 12);

	EXPECT_EQ(wlib::bop::bsf(w_32bit), 0);
	EXPECT_EQ(wlib::bop::bit_scan_forward(w_32bit), 0);


	const uint64_t x_64bit = 0x0FFFFFFFFFFFFFF0;
	const uint64_t y_64bit = 0xFFFFFFFFFFFFFFF0;
	const uint64_t z_64bit = 0xFFFFFFFFFFFFFFFF;
	const uint64_t w_64bit = 0x0000000000000000;
	const uint64_t s_64bit = 0xFFFFFFFFFF000000;

	EXPECT_EQ(wlib::bop::bsf(x_64bit), 4);
	EXPECT_EQ(wlib::bop::bit_scan_forward(x_64bit), 4);

	EXPECT_EQ(wlib::bop::bsf(y_64bit), 4);
	EXPECT_EQ(wlib::bop::bit_scan_forward(y_64bit), 4);

	EXPECT_EQ(wlib::bop::bsf(z_64bit), 0);
	EXPECT_EQ(wlib::bop::bit_scan_forward(z_64bit), 0);

	EXPECT_EQ(wlib::bop::bsf(s_64bit), 24);
	EXPECT_EQ(wlib::bop::bit_scan_forward(s_64bit), 24);

	EXPECT_EQ(wlib::bop::bsf(w_64bit), 0);
	EXPECT_EQ(wlib::bop::bit_scan_forward(w_64bit), 0);
}

TEST(BitOps, BSR) {
	const uint8_t x_8bit = 0xF0;
	const uint8_t y_8bit = 0x0F;
	const uint8_t z_8bit = 0xFF;
	const uint8_t w_8bit = 0x00;

	EXPECT_EQ(wlib::bop::bsr(x_8bit), 7);
	EXPECT_EQ(wlib::bop::bit_scan_reverse(x_8bit), 7);

	EXPECT_EQ(wlib::bop::bsr(y_8bit), 3);
	EXPECT_EQ(wlib::bop::bit_scan_reverse(y_8bit), 3);

	EXPECT_EQ(wlib::bop::bsr(z_8bit), 7);
	EXPECT_EQ(wlib::bop::bit_scan_reverse(z_8bit), 7);

	EXPECT_EQ(wlib::bop::bsr(w_8bit), 0);
	EXPECT_EQ(wlib::bop::bit_scan_reverse(w_8bit), 0);


	const uint16_t x_16bit = 0x0FF0;
	const uint16_t y_16bit = 0x0FFF;
	const uint16_t z_16bit = 0xFFFF;
	const uint16_t w_16bit = 0x0000;

	EXPECT_EQ(wlib::bop::bsr(x_16bit), 11);
	EXPECT_EQ(wlib::bop::bit_scan_reverse(x_16bit), 11);

	EXPECT_EQ(wlib::bop::bsr(y_16bit), 11);
	EXPECT_EQ(wlib::bop::bit_scan_reverse(y_16bit), 11);

	EXPECT_EQ(wlib::bop::bsr(z_16bit), 15);
	EXPECT_EQ(wlib::bop::bit_scan_reverse(z_16bit), 15);

	EXPECT_EQ(wlib::bop::bsr(w_16bit), 0);
	EXPECT_EQ(wlib::bop::bit_scan_reverse(w_16bit), 0);


	const uint32_t x_32bit = 0x0FFFFFF0;
	const uint32_t y_32bit = 0x0FFFFFFF;
	const uint32_t z_32bit = 0xFFFFFFFF;
	const uint32_t w_32bit = 0x00000000;

	EXPECT_EQ(wlib::bop::bsr(x_32bit), 27);
	EXPECT_EQ(wlib::bop::bit_scan_reverse(x_32bit), 27);

	EXPECT_EQ(wlib::bop::bsr(y_32bit), 27);
	EXPECT_EQ(wlib::bop::bit_scan_reverse(y_32bit), 27);

	EXPECT_EQ(wlib::bop::bsr(z_32bit), 31);
	EXPECT_EQ(wlib::bop::bit_scan_reverse(z_32bit), 31);

	EXPECT_EQ(wlib::bop::bsr(w_32bit), 0);
	EXPECT_EQ(wlib::bop::bit_scan_reverse(w_32bit), 0);


	const uint64_t x_64bit = 0x0FFFFFFFFFFFFFF0;
	const uint64_t y_64bit = 0x0FFFFFFFFFFFFFFF;
	const uint64_t z_64bit = 0xFFFFFFFFFFFFFFFF;
	const uint64_t w_64bit = 0x0000000000000000;

	EXPECT_EQ(wlib::bop::bsr(x_64bit), 59);
	EXPECT_EQ(wlib::bop::bit_scan_reverse(x_64bit), 59);

	EXPECT_EQ(wlib::bop::bsr(y_64bit), 59);
	EXPECT_EQ(wlib::bop::bit_scan_reverse(y_64bit), 59);

	EXPECT_EQ(wlib::bop::bsr(z_64bit), 63);
	EXPECT_EQ(wlib::bop::bit_scan_reverse(z_64bit), 63);

	EXPECT_EQ(wlib::bop::bsr(w_64bit), 0);
	EXPECT_EQ(wlib::bop::bit_scan_reverse(w_64bit), 0);
}

TEST(BitOps, PopulationCount) {
	const uint8_t x_8bit = 0xFF;
	const uint8_t y_8bit = 0x0F;
	const uint8_t z_8bit = 0x00;

	EXPECT_EQ(wlib::bop::popcnt(x_8bit), 8);
	EXPECT_EQ(wlib::bop::population_count(x_8bit), 8);

	EXPECT_EQ(wlib::bop::popcnt(y_8bit), 4);
	EXPECT_EQ(wlib::bop::population_count(y_8bit), 4);

	EXPECT_EQ(wlib::bop::popcnt(z_8bit), 0);
	EXPECT_EQ(wlib::bop::population_count(z_8bit), 0);


	const uint16_t x_16bit = 0xFFFF;
	const uint16_t y_16bit = 0x0F0F;
	const uint16_t z_16bit = 0x0000;

	EXPECT_EQ(wlib::bop::popcnt(x_16bit), 16);
	EXPECT_EQ(wlib::bop::population_count(x_16bit), 16);

	EXPECT_EQ(wlib::bop::popcnt(y_16bit), 8);
	EXPECT_EQ(wlib::bop::population_count(y_16bit), 8);

	EXPECT_EQ(wlib::bop::popcnt(z_16bit), 0);
	EXPECT_EQ(wlib::bop::population_count(z_16bit), 0);


	const uint32_t x_32bit = 0xFFFFFFFF;
	const uint32_t y_32bit = 0x0FFFF0FF;
	const uint32_t z_32bit = 0x00000000;

	EXPECT_EQ(wlib::bop::popcnt(x_32bit), 32);
	EXPECT_EQ(wlib::bop::population_count(x_32bit), 32);

	EXPECT_EQ(wlib::bop::popcnt(y_32bit), 24);
	EXPECT_EQ(wlib::bop::population_count(y_32bit), 24);

	EXPECT_EQ(wlib::bop::popcnt(z_32bit), 0);
	EXPECT_EQ(wlib::bop::population_count(z_32bit), 0);


	const uint64_t x_64bit = 0xFFFFFFFFFFFFFFFF;
	const uint64_t y_64bit = 0x0FFFFFF0000FF0FF;
	const uint64_t z_64bit = 0x0000000000000000;

	EXPECT_EQ(wlib::bop::popcnt(x_64bit), 64);
	EXPECT_EQ(wlib::bop::population_count(x_64bit), 64);

	EXPECT_EQ(wlib::bop::popcnt(y_64bit), 40);
	EXPECT_EQ(wlib::bop::population_count(y_64bit), 40);

	EXPECT_EQ(wlib::bop::popcnt(z_64bit), 0);
	EXPECT_EQ(wlib::bop::population_count(z_64bit), 0);
}


TEST(BitOps, Rotate) {
	// Rotate left

	const uint8_t x_l8bit = 0xFF;
	const uint8_t y_l8bit = 0x48;
	const uint8_t z_l8bit = 0x00;

	EXPECT_EQ(wlib::bop::rotate_left(x_l8bit, 4), 0xFF);
	EXPECT_EQ(wlib::bop::rotate_left(y_l8bit, 4), 0x84);
	EXPECT_EQ(wlib::bop::rotate_left(z_l8bit, 4), 0x00);


	const uint16_t x_l16bit = 0xFFFF;
	const uint16_t y_l16bit = 0x0F0F;
	const uint16_t z_l16bit = 0x0000;

	EXPECT_EQ(wlib::bop::rotate_left(x_l16bit, 4), 0xFFFF);
	EXPECT_EQ(wlib::bop::rotate_left(y_l16bit, 4), 0xF0F0);
	EXPECT_EQ(wlib::bop::rotate_left(z_l16bit, 4), 0x0000);


	const uint32_t x_l32bit = 0xFFFFFFFF;
	const uint32_t y_l32bit = 0x0FFFF0FF;
	const uint32_t z_l32bit = 0x00000000;

	EXPECT_EQ(wlib::bop::rotate_left(x_l32bit, 4), 0xFFFFFFFF);
	EXPECT_EQ(wlib::bop::rotate_left(y_l32bit, 4), 0xFFFF0FF0);
	EXPECT_EQ(wlib::bop::rotate_left(z_l32bit, 4), 0x00000000);


	const uint64_t x_l64bit = 0xFFFFFFFFFFFFFFFF;
	const uint64_t y_l64bit = 0x0FFFFFFFF0FFFFFF;
	const uint64_t z_l64bit = 0x0000000000000000;

	EXPECT_EQ(wlib::bop::rotate_left(x_l64bit, 4), 0xFFFFFFFFFFFFFFFF);
	EXPECT_EQ(wlib::bop::rotate_left(y_l64bit, 4), 0xFFFFFFFF0FFFFFF0);
	EXPECT_EQ(wlib::bop::rotate_left(z_l64bit, 4), 0x0000000000000000);


	// Rotate right

	const uint8_t x_r8bit = 0xFF;
	const uint8_t y_r8bit = 0x48;
	const uint8_t z_r8bit = 0x00;

	EXPECT_EQ(wlib::bop::rotate_right(x_r8bit, 4), 0xFF);
	EXPECT_EQ(wlib::bop::rotate_right(y_r8bit, 4), 0x84);
	EXPECT_EQ(wlib::bop::rotate_right(z_r8bit, 4), 0x00);


	const uint16_t x_r16bit = 0xFFFF;
	const uint16_t y_r16bit = 0x0F0F;
	const uint16_t z_r16bit = 0x0000;

	EXPECT_EQ(wlib::bop::rotate_right(x_r16bit, 4), 0xFFFF);
	EXPECT_EQ(wlib::bop::rotate_right(y_r16bit, 4), 0xF0F0);
	EXPECT_EQ(wlib::bop::rotate_right(z_r16bit, 4), 0x0000);


	const uint32_t x_r32bit = 0xFFFFFFFF;
	const uint32_t y_r32bit = 0x0FFFF0FF;
	const uint32_t z_r32bit = 0x00000000;

	EXPECT_EQ(wlib::bop::rotate_right(x_r32bit, 4), 0xFFFFFFFF);
	EXPECT_EQ(wlib::bop::rotate_right(y_r32bit, 4), 0xF0FFFF0F);
	EXPECT_EQ(wlib::bop::rotate_right(z_r32bit, 4), 0x00000000);


	const uint64_t x_r64bit = 0xFFFFFFFFFFFFFFFF;
	const uint64_t y_r64bit = 0x0FFFFFFFF0FFFFFF;
	const uint64_t z_r64bit = 0x0000000000000000;

	EXPECT_EQ(wlib::bop::rotate_right(x_r64bit, 4), 0xFFFFFFFFFFFFFFFF);
	EXPECT_EQ(wlib::bop::rotate_right(y_r64bit, 4), 0xF0FFFFFFFF0FFFFF);
	EXPECT_EQ(wlib::bop::rotate_right(z_r64bit, 4), 0x0000000000000000);
}
