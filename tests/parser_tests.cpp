#ifdef WEIRDLIB_ENABLE_FILE_PARSERS
#include <gtest/gtest.h>
#include <cstring>
#include <array>
#include <cstdint>
#include <filesystem>

#include "../include/weirdlib.hpp"
#include "../include/weirdlib_parsers.hpp"

using namespace std::string_literals;

constexpr const char* wlibTestDir = WLIBTEST_TESTING_DIRECTORY;

TEST(Parse, Comfyg) {
	std::filesystem::path filePath = std::filesystem::path(wlibTestDir) / "parser_files" / "test_config.comf";

	ASSERT_TRUE(std::filesystem::exists(filePath));

	wlib::parse::Comfyg cfg(filePath);
	auto testFloatPlain = cfg.GetVal<double>("ExampleFloatPlain");
	auto testFloatSci = cfg.GetVal<double>("ExampleFloatSci");
	auto testIntegerPlain = cfg.GetVal<int64_t>("ExampleIntegerPlain");
	auto testIntegerSep = cfg.GetVal<int64_t>("ExampleIntegerSep");
	auto testBoolT = cfg.GetVal<bool>("ExampleBoolT");
	auto testBoolF = cfg.GetVal<bool>("ExampleBoolF");
	auto testString = cfg.GetVal<std::string>("ExampleString");

	auto testFloatLowP = cfg.GetVal<float>("ExampleFloatPlain");
	auto testFloatHiP  = cfg.GetVal<long double>("ExampleFloatPlain");

	auto testInteger32 = cfg.GetVal<int32_t>("ExampleIntegerPlain");

	// No implicit conversions
	EXPECT_THROW(cfg.GetVal<bool>("ExampleFloatPlain"), wlib::parse::comfyg_value_get_error);
	EXPECT_THROW(cfg.GetVal<float>("ExampleFloatNonexistent"), std::out_of_range);
	EXPECT_THROW(cfg.GetVal<float>(""), std::out_of_range);

	EXPECT_FLOAT_EQ(testFloatPlain, 903100.1328);
	EXPECT_FLOAT_EQ(testFloatSci, 903100.1328);
	EXPECT_EQ(testIntegerPlain, -903100);
	EXPECT_EQ(testIntegerSep, -903100);
	EXPECT_TRUE(testBoolT);
	EXPECT_FALSE(testBoolF);
	EXPECT_EQ(testString, "Lorem ipsum\n Newlines");

	EXPECT_FLOAT_EQ(testFloatLowP, testFloatPlain);
	EXPECT_FLOAT_EQ(testFloatHiP, testFloatPlain);

	EXPECT_EQ(testInteger32, testIntegerPlain);
}

TEST(Parse, CuesheetSimple) {
	std::filesystem::path filePath = std::filesystem::path(wlibTestDir) / "parser_files" / "test_audiocue_single.cue";

	ASSERT_TRUE(std::filesystem::exists(filePath));

	wlib::parse::Cuesheet cue(filePath);

	EXPECT_EQ(cue.GetContents().size(), 1);

	auto cueContents = cue.GetContents()[0];

	EXPECT_EQ(cueContents.artist, "Faithless"s);
	EXPECT_EQ(cueContents.title, "Live in Berlin"s);
	EXPECT_EQ(cueContents.path, "Faithless - Live in Berlin.mp3"s);

	EXPECT_EQ(cueContents.tracks.size(), 8);
	EXPECT_EQ(cueContents.tracks[0].title, "Reverence"s);
	EXPECT_EQ(cueContents.tracks[0].startTimestamp, "00:00:00"s);
	EXPECT_EQ(cueContents.tracks[3].title, "Insomnia"s);
	EXPECT_EQ(cueContents.tracks[3].startTimestamp, "17:04:00"s);
	EXPECT_EQ(cueContents.tracks[7].title, "God Is a DJ"s);
	EXPECT_EQ(cueContents.tracks[7].startTimestamp, "42:35:00"s);
}

TEST(Parse, CuesheetUnicode) {
	std::filesystem::path filePath = std::filesystem::path(wlibTestDir) / "parser_files" / "test_audiocue_utf8.cue";

	ASSERT_TRUE(std::filesystem::exists(filePath));

	wlib::parse::Cuesheet cue(filePath);

	EXPECT_EQ(cue.GetContents().size(), 1);

	auto cueContents = cue.GetContents()[0];

	EXPECT_EQ(cueContents.artist, "Amateras Records"s);
	EXPECT_EQ(cueContents.title, "Resonate Anthems"s);

	EXPECT_EQ(cueContents.tracks.size(), 11);
	EXPECT_EQ(cueContents.tracks[0].title, "Dead or Alive"s);
	EXPECT_EQ(cueContents.tracks[0].startTimestamp, "00:00:00"s);
	EXPECT_EQ(cueContents.tracks[0].artist, "音召缶"s);
	EXPECT_EQ(cueContents.tracks[4].title, "DOUBLE CHERRY BLOSSOM [Extended Mix]"s);
	EXPECT_EQ(cueContents.tracks[4].startTimestamp, "20:07:33"s);
	EXPECT_EQ(cueContents.tracks[4].artist, "Alstroemeria Records"s);
	EXPECT_EQ(cueContents.tracks[10].title, "Dear My Stage [Halozy SMJ Eurotbeat Remix]"s);
	EXPECT_EQ(cueContents.tracks[10].startTimestamp, "52:24:40"s);
	EXPECT_EQ(cueContents.tracks[10].artist, "miko"s);
}

TEST(Parse, CuesheetMultifile) {
	std::filesystem::path filePath = std::filesystem::path(wlibTestDir) / "parser_files" / "test_audiocue_multifile.cue";
	ASSERT_TRUE(std::filesystem::exists(filePath));

	wlib::parse::Cuesheet cue(filePath);

	EXPECT_EQ(cue.GetContents().size(), 15);

	EXPECT_EQ(cue.GetContents()[0].tracks.size(), 1);
	EXPECT_EQ(cue.GetContents()[6].tracks.size(), 1);
	EXPECT_EQ(cue.GetContents()[14].tracks.size(), 1);

	EXPECT_EQ(cue.GetContents()[0].tracks[0].title, "Genesis"s);
	EXPECT_EQ(cue.GetContents()[0].tracks[0].artist, "Pendulum"s);
	EXPECT_EQ(cue.GetContents()[0].tracks[0].startTimestamp, "00:00:00"s);

	EXPECT_EQ(cue.GetContents()[6].tracks[0].startTimestamp, "00:00:00"s);
	EXPECT_EQ(cue.GetContents()[6].tracks[0].artist, "Pendulum"s);
	EXPECT_EQ(cue.GetContents()[6].tracks[0].title, "Immunize (ft. Liam H)"s);

	EXPECT_EQ(cue.GetContents()[14].tracks[0].startTimestamp, "00:00:00"s);
	EXPECT_EQ(cue.GetContents()[14].tracks[0].artist, "Pendulum"s);
	EXPECT_EQ(cue.GetContents()[14].tracks[0].title, "Encoder"s);

}
#endif
