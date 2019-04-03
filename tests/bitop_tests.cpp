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

TEST(BitOps, ctz) {
	const uint32_t x = 0x0FFFFFF0;
	const uint32_t y = 0xFFFFFFF0;
	const uint32_t z = 0xFFFFFFFF;
	const uint32_t w = 0x00000000;

	EXPECT_EQ(wlib::bop::ctz(x), 4);
	EXPECT_EQ(wlib::bop::count_trailing_zeros(x), 4);

	EXPECT_EQ(wlib::bop::ctz(y), 4);
	EXPECT_EQ(wlib::bop::count_trailing_zeros(y), 4);

	EXPECT_EQ(wlib::bop::ctz(z), 0);
	EXPECT_EQ(wlib::bop::count_trailing_zeros(z), 0);

	EXPECT_EQ(wlib::bop::ctz(w), sizeof(uint32_t) * 8);
	EXPECT_EQ(wlib::bop::count_trailing_zeros(w), sizeof(uint32_t) * 8);
}

TEST(BitOps, clz) {
	const uint32_t x = 0x0FFFFFF0;
	const uint32_t y = 0x0FFFFFFF;
	const uint32_t z = 0xFFFFFFFF;
	const uint32_t w = 0x00000000;

	EXPECT_EQ(wlib::bop::clz(x), 4);
	EXPECT_EQ(wlib::bop::count_leading_zeros(x), 4);

	EXPECT_EQ(wlib::bop::clz(y), 4);
	EXPECT_EQ(wlib::bop::count_leading_zeros(y), 4);

	EXPECT_EQ(wlib::bop::clz(z), 0);
	EXPECT_EQ(wlib::bop::count_leading_zeros(z), 0);

	EXPECT_EQ(wlib::bop::ctz(w), sizeof(uint32_t) * 8);
	EXPECT_EQ(wlib::bop::count_trailing_zeros(w), sizeof(uint32_t) * 8);
}

TEST(BitOps, bit_scan_forward) {
	const uint32_t x = 0x0FFFFFF0;
	const uint32_t y = 0xFFFFFFF0;
	const uint32_t z = 0xFFFFFFFF;
	const uint32_t w = 0x00000000;

	EXPECT_EQ(wlib::bop::bsf(x), 4);
	EXPECT_EQ(wlib::bop::bit_scan_forward(x), 4);

	EXPECT_EQ(wlib::bop::bsf(y), 4);
	EXPECT_EQ(wlib::bop::bit_scan_forward(y), 4);

	EXPECT_EQ(wlib::bop::bsf(z), 0);
	EXPECT_EQ(wlib::bop::bit_scan_forward(z), 0);

	EXPECT_EQ(wlib::bop::bsf(w), 0);
	EXPECT_EQ(wlib::bop::bit_scan_forward(w), 0);
}

TEST(BitOps, bit_scan_reverse) {
	const uint32_t x = 0x0FFFFFF0;
	const uint32_t y = 0x0FFFFFFF;
	const uint32_t z = 0xFFFFFFFF;
	const uint32_t w = 0x00000000;

	EXPECT_EQ(wlib::bop::bsr(x), 28);
	EXPECT_EQ(wlib::bop::bit_scan_reverse(x), 28);

	EXPECT_EQ(wlib::bop::bsr(y), 28);
	EXPECT_EQ(wlib::bop::bit_scan_reverse(y), 28);

	EXPECT_EQ(wlib::bop::bsr(z), 0);
	EXPECT_EQ(wlib::bop::bit_scan_reverse(z), 0);

	EXPECT_EQ(wlib::bop::bsr(w), 0);
	EXPECT_EQ(wlib::bop::bit_scan_reverse(w), 0);
}

TEST(BitOps, popcnt) {
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

TEST(BitOps, rotate_left) {
	const uint32_t x = 0xFFFFFFFF;
	const uint32_t y = 0x0FFFF0FF;
	const uint32_t z = 0x00000000;

	EXPECT_EQ(wlib::bop::rotate_left(x, 4), 0xFFFFFFFF);
	EXPECT_EQ(wlib::bop::rotate_left(y, 4), 0xFFFF0FF0);
	EXPECT_EQ(wlib::bop::rotate_left(z, 4), 0x00000000);
}

TEST(BitOps, rotate_right) {
	const uint32_t x = 0xFFFFFFFF;
	const uint32_t y = 0x0FFFF0FF;
	const uint32_t z = 0x00000000;

	EXPECT_EQ(wlib::bop::rotate_right(x, 4), 0xFFFFFFFF);
	EXPECT_EQ(wlib::bop::rotate_right(y, 4), 0xF0FFFF0F);
	EXPECT_EQ(wlib::bop::rotate_right(z, 4), 0x00000000);
}
