#ifdef WEIRDLIB_ENABLE_IMAGE_OPERATIONS
#include <gtest/gtest.h>
#include "../include/weirdlib.hpp"
#include <filesystem>
#include <future>
#include <random>
#include <algorithm>
#include <vector>
#include <fstream>
#include <thread>

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
	const std::string imgPathF = std::string(wlibTestDir) + std::string("testphoto.png");
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
	wlib::image::ImageSoA testSoA_LumBT601(testImg);

	testSoAA = wlib::image::ConvertToGrayscale(testSoAA, true);
	EXPECT_EQ(testSoAA.channels.size(), 2);

	testSoANA = wlib::image::ConvertToGrayscale(testSoANA, false);
	EXPECT_EQ(testSoANA.channels.size(), 1);

	testSoA_Average = wlib::image::ConvertToGrayscale(testSoA_Average, false, wlib::image::GrayscaleMethod::Average);
	testSoA_Lightness = wlib::image::ConvertToGrayscale(testSoA_Lightness, false, wlib::image::GrayscaleMethod::Lightness);
	testSoA_LumBT601 = wlib::image::ConvertToGrayscale(testSoA_LumBT601, false, wlib::image::GrayscaleMethod::LuminosityBT601);
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

constexpr size_t PixelSampleCount = 3000;

TEST(ImgOps, NegateValues) {
	const std::string imgPath = std::string(wlibTestDir) + std::string("testphoto.rawpix");
	ASSERT_TRUE(std::filesystem::exists(imgPath));
	wlib::image::Image testImg(imgPath, true, imageWidth, imageHeight, wlib::image::F_RGBA);
	wlib::image::ImageSoA testSoA_Alpha(testImg);
	wlib::image::ImageSoA testSoA_NoAlpha(testImg);
	wlib::image::ImageSoA testSoABase(testImg);

	EXPECT_NO_FATAL_FAILURE(wlib::image::NegateValues(testSoA_Alpha, true));
	EXPECT_NO_FATAL_FAILURE(wlib::image::NegateValues(testSoA_NoAlpha, false));

	std::random_device dev;
	std::mt19937_64 rng(dev());
	std::uniform_int_distribution<uint64_t> dist(0, testSoABase.width * testSoABase.height - 1);

	std::array<uint64_t, PixelSampleCount> Samples;
	std::generate(Samples.begin(), Samples.end(), [&dist, &rng](){return dist(rng);});

	for (size_t i = 0; i < Samples.size(); i++) {
		for (size_t c = 0; c < testSoA_NoAlpha.channels.size() - 1; c++) {
			EXPECT_FLOAT_EQ(testSoA_NoAlpha.channels[c][Samples[i]], 255.0f - testSoABase.channels[c][Samples[i]]);
		}

		for (size_t c = 0; c < testSoA_Alpha.channels.size(); c++) {
			EXPECT_FLOAT_EQ(testSoA_Alpha.channels[c][Samples[i]], 255.0f - testSoABase.channels[c][Samples[i]]);
		}
	}
}

TEST(ImgOps, FloatToUint8) {
	const std::string imgPath = std::string(wlibTestDir) + std::string("testphoto.rawpix");
	ASSERT_TRUE(std::filesystem::exists(imgPath));

	std::ifstream rawfile(imgPath, std::ios::binary | std::ios::ate);
	size_t fileSize = rawfile.tellg();
	rawfile.seekg(0);

	std::vector<uint8_t> pixelsRef(fileSize);
	rawfile.read(reinterpret_cast<char*>(pixelsRef.data()), fileSize);

	wlib::image::Image testImg(imgPath, true, imageWidth, imageHeight, wlib::image::F_RGBA);
	auto pixels = testImg.GetPixelsAsInt();

	EXPECT_TRUE(std::equal(pixelsRef.cbegin(), pixelsRef.cend(), pixels.cbegin()));
}

TEST(ImgOps, ColorOrderConversions) {
	const std::string imgPath = std::string(wlibTestDir) + std::string("testphoto.rawpix");
	ASSERT_TRUE(std::filesystem::exists(imgPath));
	wlib::image::Image testImg(imgPath, true, imageWidth, imageHeight, wlib::image::F_RGBA);

	// Testing from RGBA
	std::thread tRGBA([&](){
		wlib::image::Image testImg1(testImg);
		wlib::image::Image testImg2(testImg);
		wlib::image::Image testImg3(testImg);

		// RGBA to BGRA
		wlib::image::ConvertToBGRA(testImg1);
		EXPECT_FLOAT_EQ(testImg1.GetPixels()[0], testImg.GetPixels()[2]);
		EXPECT_FLOAT_EQ(testImg1.GetPixels()[1], testImg.GetPixels()[1]);
		EXPECT_FLOAT_EQ(testImg1.GetPixels()[2], testImg.GetPixels()[0]);
		EXPECT_FLOAT_EQ(testImg1.GetPixels()[3], testImg.GetPixels()[3]);

		// RGBA to RGB (drop alpha)
		wlib::image::ConvertToRGB(testImg2);
		EXPECT_FLOAT_EQ(testImg2.GetPixels()[0], testImg.GetPixels()[0]);
		EXPECT_FLOAT_EQ(testImg2.GetPixels()[1], testImg.GetPixels()[1]);
		EXPECT_FLOAT_EQ(testImg2.GetPixels()[2], testImg.GetPixels()[2]);
		EXPECT_FLOAT_EQ(testImg2.GetPixels()[3], testImg.GetPixels()[4]);

		// RGBA to BGR (drop alpha and flip)
		wlib::image::ConvertToBGR(testImg3);
		EXPECT_FLOAT_EQ(testImg3.GetPixels()[0], testImg.GetPixels()[2]);
		EXPECT_FLOAT_EQ(testImg3.GetPixels()[1], testImg.GetPixels()[1]);
		EXPECT_FLOAT_EQ(testImg3.GetPixels()[2], testImg.GetPixels()[0]);
		EXPECT_FLOAT_EQ(testImg3.GetPixels()[3], testImg.GetPixels()[6]);
	});

	// Testing from BGRA
	std::thread tBGRA([&](){
		wlib::image::Image testImgFromBGRA(testImg);
		wlib::image::ConvertToBGRA(testImgFromBGRA);

		wlib::image::Image testImgFromBGRA1(testImgFromBGRA);
		wlib::image::Image testImgFromBGRA2(testImgFromBGRA);
		wlib::image::Image testImgFromBGRA3(testImgFromBGRA);

		// BGRA to RGBA
		wlib::image::ConvertToRGBA(testImgFromBGRA1);
		EXPECT_FLOAT_EQ(testImgFromBGRA1.GetPixels()[0], testImgFromBGRA.GetPixels()[2]);
		EXPECT_FLOAT_EQ(testImgFromBGRA1.GetPixels()[1], testImgFromBGRA.GetPixels()[1]);
		EXPECT_FLOAT_EQ(testImgFromBGRA1.GetPixels()[2], testImgFromBGRA.GetPixels()[0]);
		EXPECT_FLOAT_EQ(testImgFromBGRA1.GetPixels()[3], testImgFromBGRA.GetPixels()[3]);

		// BGRA to BGR (drop alpha)
		wlib::image::ConvertToBGR(testImgFromBGRA2);
		EXPECT_FLOAT_EQ(testImgFromBGRA2.GetPixels()[0], testImgFromBGRA.GetPixels()[0]);
		EXPECT_FLOAT_EQ(testImgFromBGRA2.GetPixels()[1], testImgFromBGRA.GetPixels()[1]);
		EXPECT_FLOAT_EQ(testImgFromBGRA2.GetPixels()[2], testImgFromBGRA.GetPixels()[2]);
		EXPECT_FLOAT_EQ(testImgFromBGRA2.GetPixels()[3], testImgFromBGRA.GetPixels()[4]);

		// BGRA to RGB (drop alpha and flip)
		wlib::image::ConvertToRGB(testImgFromBGRA3);
		EXPECT_FLOAT_EQ(testImgFromBGRA3.GetPixels()[0], testImgFromBGRA.GetPixels()[2]);
		EXPECT_FLOAT_EQ(testImgFromBGRA3.GetPixels()[1], testImgFromBGRA.GetPixels()[1]);
		EXPECT_FLOAT_EQ(testImgFromBGRA3.GetPixels()[2], testImgFromBGRA.GetPixels()[0]);
		EXPECT_FLOAT_EQ(testImgFromBGRA3.GetPixels()[3], testImgFromBGRA.GetPixels()[6]);
	});

	// Testing from RGB
	std::thread tRGB([&](){
		wlib::image::Image testImgBaseRGB(testImg);
		wlib::image::ConvertToRGB(testImgBaseRGB);
		wlib::image::Image testImgBaseRGB1(testImgBaseRGB);
		wlib::image::Image testImgBaseRGB2(testImgBaseRGB);
		wlib::image::Image testImgBaseRGB3(testImgBaseRGB);

		// RGB to BGR
		wlib::image::ConvertToBGR(testImgBaseRGB1);
		EXPECT_FLOAT_EQ(testImgBaseRGB1.GetPixels()[0], testImgBaseRGB.GetPixels()[2]);
		EXPECT_FLOAT_EQ(testImgBaseRGB1.GetPixels()[1], testImgBaseRGB.GetPixels()[1]);
		EXPECT_FLOAT_EQ(testImgBaseRGB1.GetPixels()[2], testImgBaseRGB.GetPixels()[0]);

		// RGB to RGBA (append alpha)
		wlib::image::ConvertToRGBA(testImgBaseRGB2);
		EXPECT_FLOAT_EQ(testImgBaseRGB2.GetPixels()[0], testImgBaseRGB.GetPixels()[0]);
		EXPECT_FLOAT_EQ(testImgBaseRGB2.GetPixels()[1], testImgBaseRGB.GetPixels()[1]);
		EXPECT_FLOAT_EQ(testImgBaseRGB2.GetPixels()[2], testImgBaseRGB.GetPixels()[2]);
		EXPECT_FLOAT_EQ(testImgBaseRGB2.GetPixels()[3], 255.0f);

		// RGB to BGRA (append alpha and flip)
		wlib::image::ConvertToBGRA(testImgBaseRGB3);
		EXPECT_FLOAT_EQ(testImgBaseRGB3.GetPixels()[0], testImgBaseRGB.GetPixels()[2]);
		EXPECT_FLOAT_EQ(testImgBaseRGB3.GetPixels()[1], testImgBaseRGB.GetPixels()[1]);
		EXPECT_FLOAT_EQ(testImgBaseRGB3.GetPixels()[2], testImgBaseRGB.GetPixels()[0]);
		EXPECT_FLOAT_EQ(testImgBaseRGB3.GetPixels()[3], 255.0f);
	});

	// Testing from BGR
	wlib::image::Image testImgBaseBGR(testImg);
	wlib::image::ConvertToBGR(testImgBaseBGR);
	wlib::image::Image testImgBaseBGR1(testImgBaseBGR);
	wlib::image::Image testImgBaseBGR2(testImgBaseBGR);
	wlib::image::Image testImgBaseBGR3(testImgBaseBGR);

	// BGR to RGB
	wlib::image::ConvertToRGB(testImgBaseBGR1);
	EXPECT_FLOAT_EQ(testImgBaseBGR1.GetPixels()[0], testImgBaseBGR.GetPixels()[2]);
	EXPECT_FLOAT_EQ(testImgBaseBGR1.GetPixels()[1], testImgBaseBGR.GetPixels()[1]);
	EXPECT_FLOAT_EQ(testImgBaseBGR1.GetPixels()[2], testImgBaseBGR.GetPixels()[0]);

	// BGR to BGRA (append alpha)
	wlib::image::ConvertToBGRA(testImgBaseBGR2);
	EXPECT_FLOAT_EQ(testImgBaseBGR2.GetPixels()[0], testImgBaseBGR.GetPixels()[0]);
	EXPECT_FLOAT_EQ(testImgBaseBGR2.GetPixels()[1], testImgBaseBGR.GetPixels()[1]);
	EXPECT_FLOAT_EQ(testImgBaseBGR2.GetPixels()[2], testImgBaseBGR.GetPixels()[2]);
	EXPECT_FLOAT_EQ(testImgBaseBGR2.GetPixels()[3], 255.0f);

	// BGR to RGBA (append alpha and flip)
	wlib::image::ConvertToRGBA(testImgBaseBGR3);
	EXPECT_FLOAT_EQ(testImgBaseBGR3.GetPixels()[0], testImgBaseBGR.GetPixels()[2]);
	EXPECT_FLOAT_EQ(testImgBaseBGR3.GetPixels()[1], testImgBaseBGR.GetPixels()[1]);
	EXPECT_FLOAT_EQ(testImgBaseBGR3.GetPixels()[2], testImgBaseBGR.GetPixels()[0]);
	EXPECT_FLOAT_EQ(testImgBaseBGR3.GetPixels()[3], 255.0f);


	tBGRA.join();
	tRGBA.join();
	tRGB.join();
}

#endif
