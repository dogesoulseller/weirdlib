#include <gtest/gtest.h>
#include "../include/weirdlib_anxiety.hpp"

TEST(Anxiety, Sqrt) {
	EXPECT_NO_FATAL_FAILURE(wlib::anxiety::StressSquareRoot(std::chrono::milliseconds(100), 2));
	EXPECT_NO_FATAL_FAILURE(wlib::anxiety::StressSquareRoot(10, 1));
}

TEST(Anxiety, RecipSqrt) {
	EXPECT_NO_FATAL_FAILURE(wlib::anxiety::StressInverseSquareRoot(std::chrono::milliseconds(100), 2));
	EXPECT_NO_FATAL_FAILURE(wlib::anxiety::StressInverseSquareRoot(10, 1));
}

TEST(Anxiety, FMA) {
	EXPECT_NO_FATAL_FAILURE(wlib::anxiety::StressFMA(std::chrono::milliseconds(100), 2));
	EXPECT_NO_FATAL_FAILURE(wlib::anxiety::StressFMA(10, 1));
}
