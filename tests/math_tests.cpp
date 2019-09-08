#include <gtest/gtest.h>
#include "../include/weirdlib_math.hpp"

TEST(Math, NextMultiple) {
	EXPECT_EQ(wlib::math::next_multiple(89, 64), 128);
	EXPECT_EQ(wlib::math::next_multiple(64, 64), 64);
	EXPECT_EQ(wlib::math::next_multiple(0, 0), 0);
}

TEST(Math, PrevMultiple) {
	EXPECT_EQ(wlib::math::previous_multiple(89, 64), 64);
	EXPECT_EQ(wlib::math::previous_multiple(64, 64), 64);
	EXPECT_EQ(wlib::math::previous_multiple(0, 0), 0);
}

TEST(Math, NearestMultiple) {
	EXPECT_EQ(wlib::math::nearest_multiple(70, 64), 64);
	EXPECT_EQ(wlib::math::nearest_multiple(64, 64), 64);
	EXPECT_EQ(wlib::math::nearest_multiple(110, 64), 128);
	EXPECT_EQ(wlib::math::nearest_multiple(0, 0), 0);
}

TEST(Math, FloatEquality) {
	EXPECT_TRUE(wlib::math::float_eq(0.1 + 0.2, 0.3));
	EXPECT_FALSE(wlib::math::float_eq(0.00000001, 0.0000001));	// 0.000_000_01 not eq 0.000_000_1
	EXPECT_FALSE(wlib::math::float_eq(0.0000000000001, 0.000000000001)); // 0.000_000_000_000_1 not eq 0.000_000_000_001
	EXPECT_TRUE(wlib::math::float_eq(0.0000000000001, 0.0000000000001)); // 0.000_000_000_000_1 eq 0.000_000_000_000_1
}
