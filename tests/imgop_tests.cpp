#include <gtest/gtest.h>
#include "../include/weirdlib.hpp"
#include <filesystem>
#include <future>

constexpr const char* wlibTestDir = WLIBTEST_TESTING_DIRECTORY;

constexpr uint64_t imageWidth = 1200;
constexpr uint64_t imageHeight = 1500;

TEST(ImgOps, LoadRawData) {
	const std::string imgPath = std::string(wlibTestDir) + std::string("testphoto.rawpix");
	ASSERT_TRUE(std::filesystem::exists(imgPath));

	wlib::image::Image testImg(imgPath, true, imageWidth, imageHeight, wlib::image::F_RGBA);

	EXPECT_EQ(testImg.GetWidth(), imageWidth);
	EXPECT_EQ(testImg.GetHeight(), imageHeight);
	EXPECT_EQ(testImg.GetFormat(), wlib::image::F_RGBA);
	EXPECT_FLOAT_EQ(testImg.GetPixels()[3], 255.0f);
}

TEST(ImgOps, ConvertToSoA) {
	const std::string imgPath = std::string(wlibTestDir) + std::string("testphoto.rawpix");
	ASSERT_TRUE(std::filesystem::exists(imgPath));

	wlib::image::Image testImg(imgPath, true, imageWidth, imageHeight, wlib::image::F_RGBA);
	wlib::image::ImageSoA testSoA(testImg);

	EXPECT_EQ(testSoA.channels.size(), 4);
	EXPECT_FLOAT_EQ(testImg.GetPixels()[0], testSoA.channels[0][0]);
	EXPECT_FLOAT_EQ(testImg.GetPixels()[1], testSoA.channels[1][0]);
	EXPECT_FLOAT_EQ(testImg.GetPixels()[2], testSoA.channels[2][0]);
	EXPECT_FLOAT_EQ(testImg.GetPixels()[3], testSoA.channels[3][0]);
}

TEST(ImgOps, ConvertSoAToImage) {
	const std::string imgPath = std::string(wlibTestDir) + std::string("testphoto.rawpix");
	ASSERT_TRUE(std::filesystem::exists(imgPath));
	wlib::image::Image testImg(imgPath, true, imageWidth, imageHeight, wlib::image::F_RGBA);
	wlib::image::ImageSoA testSoA(testImg);
	wlib::image::Image testImgFromSoA = testSoA.ConvertToImage();

	EXPECT_FLOAT_EQ(testImg.GetPixels()[0], testImgFromSoA.GetPixels()[0]);
	EXPECT_FLOAT_EQ(testImg.GetPixels()[1], testImgFromSoA.GetPixels()[1]);
	EXPECT_FLOAT_EQ(testImg.GetPixels()[2], testImgFromSoA.GetPixels()[2]);
	EXPECT_FLOAT_EQ(testImg.GetPixels()[3], testImgFromSoA.GetPixels()[3]);
}

TEST(ImgOps, LoadFromFormatted) {
	const std::string imgPath = std::string(wlibTestDir) + std::string("testphoto.rawpix");
	const std::string imgPathF = std::string(wlibTestDir) + std::string("testphoto.bmp");
	ASSERT_TRUE(std::filesystem::exists(imgPath));
	ASSERT_TRUE(std::filesystem::exists(imgPathF));
	wlib::image::Image testImg(imgPath, true, imageWidth, imageHeight, wlib::image::F_RGBA);
	wlib::image::Image testImgF(imgPathF);

	EXPECT_EQ(testImg.GetWidth(), testImgF.GetWidth());
	EXPECT_EQ(testImg.GetHeight(), testImgF.GetHeight());
	EXPECT_EQ(testImg.GetFormat(), wlib::image::F_RGBA);
	EXPECT_FLOAT_EQ(testImg.GetPixels()[0], testImgF.GetPixels()[0]);
	EXPECT_FLOAT_EQ(testImg.GetPixels()[1], testImgF.GetPixels()[1]);
	EXPECT_FLOAT_EQ(testImg.GetPixels()[2], testImgF.GetPixels()[2]);
	EXPECT_FLOAT_EQ(testImg.GetPixels()[3], testImgF.GetPixels()[3]);
}
