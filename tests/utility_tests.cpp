#include <gtest/gtest.h>
#include "../include/weirdlib_utility.hpp"

#include <cstddef>
#include <string>
#include <random>

TEST(Utility, EqualToOneOf) {
	std::string testString = "abcdefghijklmnoprstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";

	EXPECT_FALSE(wlib::util::EqualToOneOf('q', testString.cbegin(), testString.cend()));
	EXPECT_TRUE(wlib::util::EqualToOneOf('a', testString.cbegin(), testString.cend()));
	EXPECT_TRUE(wlib::util::EqualToOneOf('Z', testString.cbegin(), testString.cend()));
	EXPECT_TRUE(wlib::util::EqualToOneOf('A', testString.cbegin(), testString.cend()));
}

TEST(Utility, Denormalize) {
	std::random_device dev;
	std::uniform_real_distribution<long double> distLdbl(static_cast<long double>(0.0), static_cast<long double>(1.0));
	std::uniform_real_distribution<double> distDbl(0.0, 1.0);
	std::uniform_real_distribution<float> distFlt(0.0f, 1.0f);

	std::mt19937_64 mtrng(dev());

	std::vector<float> datavecflt(64*57);
	std::generate(datavecflt.begin()+2, datavecflt.end(), [&](){return distFlt(mtrng);});
	datavecflt[0] = 0.0f;
	datavecflt[1] = 1.0f;

	std::vector<double> datavecdbl(64*57);
	std::generate(datavecdbl.begin()+2, datavecdbl.end(), [&](){return distDbl(mtrng);});
	datavecdbl[0] = 0.0;
	datavecdbl[1] = 1.0;

	std::vector<long double> datavecldbl(64*57);
	std::generate(datavecldbl.begin()+2, datavecldbl.end(), [&](){return distLdbl(mtrng);});
	datavecldbl[0] = static_cast<long double>(0.0);
	datavecldbl[1] = static_cast<long double>(1.0);

	auto datavecflt_ref = datavecflt;
	auto datavecdbl_ref = datavecdbl;
	auto datavecldbl_ref = datavecldbl;

	wlib::util::DenormalizeData(datavecflt.data(), datavecflt.size(), 255.0f);
	wlib::util::DenormalizeData(datavecdbl.data(), datavecdbl.size(), 255.0);
	wlib::util::DenormalizeData(datavecldbl.data(), datavecldbl.size(), static_cast<long double>(255.0));

	// Check extreme values
	EXPECT_FLOAT_EQ(datavecflt[0], 0.0f);
	EXPECT_FLOAT_EQ(datavecflt[1], 255.0f);

	EXPECT_DOUBLE_EQ(datavecdbl[0], 0.0);
	EXPECT_DOUBLE_EQ(datavecdbl[1], 255.0);

	EXPECT_DOUBLE_EQ(datavecldbl[0], 0.0);
	EXPECT_DOUBLE_EQ(datavecldbl[1], 255.0);

	// Check all values
	for (size_t i = 0; i < datavecflt.size(); i++) {
		EXPECT_FLOAT_EQ(datavecflt_ref[i] * 255.0f, datavecflt[i]);
		EXPECT_DOUBLE_EQ(datavecdbl_ref[i] * 255.0, datavecdbl[i]);
		EXPECT_DOUBLE_EQ(datavecldbl_ref[i] * static_cast<long double>(255.0), datavecldbl[i]);
	}
}