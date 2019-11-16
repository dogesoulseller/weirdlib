#include <gtest/gtest.h>
#include <cstring>
#include <array>
#include <cstdint>
#include <filesystem>

#include "../include/weirdlib.hpp"
#include "../include/weirdlib_parsers.hpp"

constexpr const char* wlibTestDir = WLIBTEST_TESTING_DIRECTORY;

TEST(Parse, Comfyg) {
	std::filesystem::path filePath = std::filesystem::path(wlibTestDir) / "parser_files" / "test_config.comf";

	wlib::parse::Comfyg cfg(filePath);
	auto testFloatPlain = cfg.GetVal<double>("ExampleFloatPlain");
	auto testFloatSci = cfg.GetVal<double>("ExampleFloatSci");
	auto testIntegerPlain = cfg.GetVal<int64_t>("ExampleIntegerPlain");
	auto testIntegerSep = cfg.GetVal<int64_t>("ExampleIntegerSep");
	auto testBoolT = cfg.GetVal<bool>("ExampleBoolT");
	auto testBoolF = cfg.GetVal<bool>("ExampleBoolF");
	auto testString = cfg.GetVal<std::string>("ExampleString");

	// No implicit conversions
	EXPECT_ANY_THROW(cfg.GetVal<bool>("ExampleFloatPlain"));
	EXPECT_ANY_THROW(cfg.GetVal<float>("ExampleFloatNonexistent"));
	EXPECT_ANY_THROW(cfg.GetVal<float>(""));

	EXPECT_FLOAT_EQ(testFloatPlain, 903100.1328);
	EXPECT_FLOAT_EQ(testFloatSci, 903100.1328);
	EXPECT_EQ(testIntegerPlain, -903100);
	EXPECT_EQ(testIntegerSep, -903100);
	EXPECT_TRUE(testBoolT);
	EXPECT_FALSE(testBoolF);
	EXPECT_EQ(testString, "Lorem ipsum\n Newlines");
}