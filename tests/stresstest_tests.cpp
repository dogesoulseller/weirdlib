#ifdef WEIRDLIB_ENABLE_ANXIETY
#include <gtest/gtest.h>
#include "../include/weirdlib_anxiety.hpp"

TEST(Anxiety, Sqrt) {
	EXPECT_NO_FATAL_FAILURE(wlib::anxiety::StressSquareRoot(std::chrono::milliseconds(5), 2));
	EXPECT_NO_FATAL_FAILURE(wlib::anxiety::StressSquareRoot(5, 1));
}

TEST(Anxiety, RecipSqrt) {
	EXPECT_NO_FATAL_FAILURE(wlib::anxiety::StressInverseSquareRoot(std::chrono::milliseconds(5), 2));
	EXPECT_NO_FATAL_FAILURE(wlib::anxiety::StressInverseSquareRoot(5, 1));
}

TEST(Anxiety, FMA) {
	EXPECT_NO_FATAL_FAILURE(wlib::anxiety::StressFMA(std::chrono::milliseconds(5), 2));
	EXPECT_NO_FATAL_FAILURE(wlib::anxiety::StressFMA(5, 1));
}
#endif
