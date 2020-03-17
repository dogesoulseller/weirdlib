#include <gtest/gtest.h>
#include "../include/weirdlib_utility.hpp"

#include <string>

TEST(Utility, EqualToOneOf) {
	std::string testString = "abcdefghijklmnoprstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";

	EXPECT_FALSE(wlib::util::EqualToOneOf('q', testString.cbegin(), testString.cend()));
	EXPECT_TRUE(wlib::util::EqualToOneOf('a', testString.cbegin(), testString.cend()));
	EXPECT_TRUE(wlib::util::EqualToOneOf('Z', testString.cbegin(), testString.cend()));
	EXPECT_TRUE(wlib::util::EqualToOneOf('A', testString.cbegin(), testString.cend()));
}