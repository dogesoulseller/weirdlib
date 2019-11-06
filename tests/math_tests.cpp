#include <gtest/gtest.h>
#include "../include/weirdlib_math.hpp"
#include <random>

std::uniform_int_distribution random_dist(4, 65);

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

TEST(Math, VariadicMinMax) {
	std::mt19937 rng;

	EXPECT_EQ(wlib::math::max(13, 14, 15, 16, 32, 3, 13, 5, random_dist(rng), 66, 4, 5, 3), 66);
	EXPECT_EQ(wlib::math::min(13, 14, 15, random_dist(rng), 32, 3, 13, 5, 66, 5, 13, 22, 3), 3);
}

TEST(Math, VariadicAverage) {
	// Compute average
	EXPECT_EQ(wlib::math::average(13, 13, 13, 13, 13, 14), 13);
	EXPECT_FLOAT_EQ(wlib::math::average(13.0f, 13.0f, 13.0f, 13.0f, 13.0f, 14.0f), 13.1666666666666f);
	EXPECT_FLOAT_EQ(wlib::math::average(13.0, 13.0, 13.0, 13.0, 13.0, 14.0), 13.1666666666666);
	EXPECT_FLOAT_EQ(wlib::math::average(13.0, 13.0, 13.0, 13, 13.0, 14), 13.1666666666666);

	// Compute average with remainder
	auto [avg, rem] = wlib::math::average_and_remainder(13, 13, 13, 13, 13, 14);
	EXPECT_EQ(avg, 13);
	EXPECT_EQ(rem, 1);
}