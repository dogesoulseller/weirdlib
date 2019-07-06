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

TEST(ImgOps, ConvertRGBA_Grayscale) {
	const std::string imgPath = std::string(wlibTestDir) + std::string("testphoto.rawpix");
	ASSERT_TRUE(std::filesystem::exists(imgPath));

	wlib::image::Image testImg(imgPath, true, imageWidth, imageHeight, wlib::image::F_RGBA);
	wlib::image::ImageSoA testSoAA(testImg);
	wlib::image::ImageSoA testSoANA(testImg);

	wlib::image::ImageSoA testSoA_Average(testImg);
	wlib::image::ImageSoA testSoA_Lightness(testImg);

	testSoAA = wlib::image::ConvertToGrayscale(testSoAA, true);
	EXPECT_EQ(testSoAA.channels.size(), 2);

	testSoANA = wlib::image::ConvertToGrayscale(testSoANA, false);
	EXPECT_EQ(testSoANA.channels.size(), 1);

	testSoA_Average = wlib::image::ConvertToGrayscale(testSoA_Average, false, wlib::image::GrayscaleMethod::Average);
	testSoA_Lightness = wlib::image::ConvertToGrayscale(testSoA_Lightness, false, wlib::image::GrayscaleMethod::Lightness);
}

TEST(ImgOps, ColorOrderConversion) {
	const std::string imgPath = std::string(wlibTestDir) + std::string("testphoto.rawpix");
	ASSERT_TRUE(std::filesystem::exists(imgPath));

	wlib::image::Image testImg(imgPath, true, imageWidth, imageHeight, wlib::image::F_RGBA);
	wlib::image::ImageSoA testSoAA(testImg);
	wlib::image::ImageSoA testSoANA(testImg);

	wlib::image::ConvertToBGRA(testSoAA);
	wlib::image::ConvertToBGR(testSoANA);

	EXPECT_EQ(testSoAA.channels[0][0], testImg.GetPixels()[2]);
	EXPECT_EQ(testSoAA.channels[1][0], testImg.GetPixels()[1]);
	EXPECT_EQ(testSoAA.channels[2][0], testImg.GetPixels()[0]);
	EXPECT_EQ(testSoAA.channels[3][0], testImg.GetPixels()[3]);

	EXPECT_EQ(testSoANA.channels[0][0], testImg.GetPixels()[2]);
	EXPECT_EQ(testSoANA.channels[1][0], testImg.GetPixels()[1]);
	EXPECT_EQ(testSoANA.channels[2][0], testImg.GetPixels()[0]);
	EXPECT_EQ(testSoANA.channels.size(), 3);

	// Check alpha append
	wlib::image::ConvertToBGRA(testSoANA);

	EXPECT_EQ(testSoANA.channels[0][0], testImg.GetPixels()[2]);
	EXPECT_EQ(testSoANA.channels[1][0], testImg.GetPixels()[1]);
	EXPECT_EQ(testSoANA.channels[2][0], testImg.GetPixels()[0]);
	EXPECT_EQ(testSoANA.channels[3][0], 255.0f);
}
