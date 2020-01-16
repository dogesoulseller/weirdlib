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
#include <limits>

using namespace std::string_literals;

constexpr const char* wlibTestDir = WLIBTEST_TESTING_DIRECTORY;

constexpr uint64_t imageWidth = 64;
constexpr uint64_t imageHeight = 64;

TEST(ImgOps, LoadRawData) {
	const std::string imgPath = std::string(wlibTestDir) + "imgload_files/base.rawpix"s;
	ASSERT_TRUE(std::filesystem::exists(imgPath));

	wlib::image::Image testImg(imgPath, true, imageWidth, imageHeight, wlib::image::F_RGBA);

	EXPECT_EQ(testImg.GetWidth(), imageWidth);
	EXPECT_EQ(testImg.GetHeight(), imageHeight);
	EXPECT_EQ(testImg.GetFormat(), wlib::image::F_RGBA);
	EXPECT_FLOAT_EQ(testImg.GetPixels()[3], 255.0f);
}

TEST(ImgOps, ConvertToSoA) {
	const std::string imgPath = std::string(wlibTestDir) + "imgload_files/base.rawpix"s;
	ASSERT_TRUE(std::filesystem::exists(imgPath));

	wlib::image::Image testImg(imgPath, true, imageWidth, imageHeight, wlib::image::F_RGBA);
	auto testSoA = wlib::image::MakeSoAFromAoS(testImg);

	EXPECT_EQ(testSoA.channels.size(), 4);
	EXPECT_FLOAT_EQ(testImg.GetPixels()[0], testSoA.channels[0][0]);
	EXPECT_FLOAT_EQ(testImg.GetPixels()[1], testSoA.channels[1][0]);
	EXPECT_FLOAT_EQ(testImg.GetPixels()[2], testSoA.channels[2][0]);
	EXPECT_FLOAT_EQ(testImg.GetPixels()[3], testSoA.channels[3][0]);
}

TEST(ImgOps, ConvertSoAToImage) {
	const std::string imgPath = std::string(wlibTestDir) + "imgload_files/base.rawpix"s;
	ASSERT_TRUE(std::filesystem::exists(imgPath));
	wlib::image::Image testImg(imgPath, true, imageWidth, imageHeight, wlib::image::F_RGBA);
	auto testSoA = wlib::image::MakeSoAFromAoS(testImg);
	wlib::image::Image testImgFromSoA = wlib::image::MakeAoSFromSoA(testSoA);

	EXPECT_FLOAT_EQ(testImg.GetPixels()[0], testImgFromSoA.GetPixels()[0]);
	EXPECT_FLOAT_EQ(testImg.GetPixels()[1], testImgFromSoA.GetPixels()[1]);
	EXPECT_FLOAT_EQ(testImg.GetPixels()[2], testImgFromSoA.GetPixels()[2]);
	EXPECT_FLOAT_EQ(testImg.GetPixels()[3], testImgFromSoA.GetPixels()[3]);
}

TEST(ImgOps, LoadFromFormatted) {
	const std::string imgPath = std::string(wlibTestDir) + "imgload_files/base.rawpix"s;
	const std::string imgPathF = std::string(wlibTestDir) + "imgload_files/base.png"s;
	ASSERT_TRUE(std::filesystem::exists(imgPath));
	ASSERT_TRUE(std::filesystem::exists(imgPathF));
	wlib::image::Image testImg(imgPath, true, imageWidth, imageHeight, wlib::image::F_RGBA);
	wlib::image::Image testImgF(imgPathF);

	auto testImgSoA = wlib::image::MakeSoAFromAoS(testImgF);

	EXPECT_EQ(testImg.GetWidth(), testImgF.GetWidth());
	EXPECT_EQ(testImg.GetHeight(), testImgF.GetHeight());
	EXPECT_EQ(testImg.GetFormat(), wlib::image::F_RGBA);

	EXPECT_FLOAT_EQ(testImg.GetPixels()[0], testImgF.GetPixels()[0]);
	EXPECT_FLOAT_EQ(testImg.GetPixels()[1], testImgF.GetPixels()[1]);
	EXPECT_FLOAT_EQ(testImg.GetPixels()[2], testImgF.GetPixels()[2]);
	EXPECT_FLOAT_EQ(testImg.GetPixels()[3], testImgF.GetPixels()[3]);

	EXPECT_FLOAT_EQ(testImg.GetPixels()[0], testImgSoA.channels[0][0]);
	EXPECT_FLOAT_EQ(testImg.GetPixels()[1], testImgSoA.channels[1][0]);
	EXPECT_FLOAT_EQ(testImg.GetPixels()[2], testImgSoA.channels[2][0]);
	EXPECT_FLOAT_EQ(testImg.GetPixels()[3], testImgSoA.channels[3][0]);
}

TEST(ImgOps, ConvertRGBA_Grayscale) {
	const std::string imgPath = std::string(wlibTestDir) + "imgload_files/base.rawpix"s;
	ASSERT_TRUE(std::filesystem::exists(imgPath));

	wlib::image::Image testImg(imgPath, true, imageWidth, imageHeight, wlib::image::F_RGBA);
	auto testSoA = wlib::image::MakeSoAFromAoS(testImg);
	auto testSoAA = wlib::image::MakeSoAFromAoS(testImg);
	auto testSoANA = wlib::image::MakeSoAFromAoS(testImg);

	auto testSoA_Average = wlib::image::MakeSoAFromAoS(testImg);
	auto testSoA_Lightness = wlib::image::MakeSoAFromAoS(testImg);
	auto testSoA_LumBT601 = wlib::image::MakeSoAFromAoS(testImg);
	auto testSoA_LumBT2100 = wlib::image::MakeSoAFromAoS(testImg);

	testSoAA = wlib::image::ConvertToGrayscale(testSoAA, true);
	EXPECT_EQ(testSoAA.channels.size(), 2);

	testSoANA = wlib::image::ConvertToGrayscale(testSoANA, false);
	EXPECT_EQ(testSoANA.channels.size(), 1);

	testSoA_Average = wlib::image::ConvertToGrayscale(testSoA_Average, false, wlib::image::GrayscaleMethod::Average);
	testSoA_Lightness = wlib::image::ConvertToGrayscale(testSoA_Lightness, false, wlib::image::GrayscaleMethod::Lightness);
	testSoA_LumBT601 = wlib::image::ConvertToGrayscale(testSoA_LumBT601, false, wlib::image::GrayscaleMethod::LuminosityBT601);
	testSoA_LumBT601 = wlib::image::ConvertToGrayscale(testSoA_LumBT2100, false, wlib::image::GrayscaleMethod::LuminosityBT2100);
}

constexpr size_t PixelSampleCount = 50;

TEST(ImgOps, NegateValues) {
	const std::string imgPath = std::string(wlibTestDir) + "imgload_files/base.rawpix"s;
	ASSERT_TRUE(std::filesystem::exists(imgPath));
	wlib::image::Image testImg(imgPath, true, imageWidth, imageHeight, wlib::image::F_RGBA);
	auto testSoA_Alpha = wlib::image::MakeSoAFromAoS(testImg);
	auto testSoA_NoAlpha = wlib::image::MakeSoAFromAoS(testImg);
	auto testSoABase = wlib::image::MakeSoAFromAoS(testImg);

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
	const std::string imgPath = std::string(wlibTestDir) + "imgload_files/base.rawpix"s;
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

// TODO: Change to 64x64 images
TEST(ImgOps, ColorOrderConversions) {
	const std::string imgPath = std::string(wlibTestDir) + "imgload_files/base.rawpix"s;
	ASSERT_TRUE(std::filesystem::exists(imgPath));
	wlib::image::Image testImg(imgPath, true, imageWidth, imageHeight, wlib::image::F_RGBA);

	// Testing from RGBA
	std::thread tRGBA([&](){
		wlib::image::Image testImg1(testImg);
		wlib::image::Image testImg2(testImg);
		wlib::image::Image testImg3(testImg);

		// RGBA to BGRA
		testImg1.ConvertToBGRA();
		EXPECT_FLOAT_EQ(testImg1.GetPixels()[0], testImg.GetPixels()[2]);
		EXPECT_FLOAT_EQ(testImg1.GetPixels()[1], testImg.GetPixels()[1]);
		EXPECT_FLOAT_EQ(testImg1.GetPixels()[2], testImg.GetPixels()[0]);
		EXPECT_FLOAT_EQ(testImg1.GetPixels()[3], testImg.GetPixels()[3]);

		// RGBA to RGB (drop alpha)
		testImg2.ConvertToRGB();
		EXPECT_FLOAT_EQ(testImg2.GetPixels()[0], testImg.GetPixels()[0]);
		EXPECT_FLOAT_EQ(testImg2.GetPixels()[1], testImg.GetPixels()[1]);
		EXPECT_FLOAT_EQ(testImg2.GetPixels()[2], testImg.GetPixels()[2]);
		EXPECT_FLOAT_EQ(testImg2.GetPixels()[3], testImg.GetPixels()[4]);

		// RGBA to BGR (drop alpha and flip)
		testImg3.ConvertToBGR();
		EXPECT_FLOAT_EQ(testImg3.GetPixels()[0], testImg.GetPixels()[2]);
		EXPECT_FLOAT_EQ(testImg3.GetPixels()[1], testImg.GetPixels()[1]);
		EXPECT_FLOAT_EQ(testImg3.GetPixels()[2], testImg.GetPixels()[0]);
		EXPECT_FLOAT_EQ(testImg3.GetPixels()[3], testImg.GetPixels()[6]);
	});

	// Testing from BGRA
	std::thread tBGRA([&](){
		wlib::image::Image testImgFromBGRA(testImg);
		testImgFromBGRA.ConvertToBGRA();

		wlib::image::Image testImgFromBGRA1(testImgFromBGRA);
		wlib::image::Image testImgFromBGRA2(testImgFromBGRA);
		wlib::image::Image testImgFromBGRA3(testImgFromBGRA);

		// BGRA to RGBA
		testImgFromBGRA1.ConvertToRGBA();
		EXPECT_FLOAT_EQ(testImgFromBGRA1.GetPixels()[0], testImgFromBGRA.GetPixels()[2]);
		EXPECT_FLOAT_EQ(testImgFromBGRA1.GetPixels()[1], testImgFromBGRA.GetPixels()[1]);
		EXPECT_FLOAT_EQ(testImgFromBGRA1.GetPixels()[2], testImgFromBGRA.GetPixels()[0]);
		EXPECT_FLOAT_EQ(testImgFromBGRA1.GetPixels()[3], testImgFromBGRA.GetPixels()[3]);

		// BGRA to BGR (drop alpha)
		testImgFromBGRA2.ConvertToBGR();
		EXPECT_FLOAT_EQ(testImgFromBGRA2.GetPixels()[0], testImgFromBGRA.GetPixels()[0]);
		EXPECT_FLOAT_EQ(testImgFromBGRA2.GetPixels()[1], testImgFromBGRA.GetPixels()[1]);
		EXPECT_FLOAT_EQ(testImgFromBGRA2.GetPixels()[2], testImgFromBGRA.GetPixels()[2]);
		EXPECT_FLOAT_EQ(testImgFromBGRA2.GetPixels()[3], testImgFromBGRA.GetPixels()[4]);

		// BGRA to RGB (drop alpha and flip)
		testImgFromBGRA3.ConvertToRGB();
		EXPECT_FLOAT_EQ(testImgFromBGRA3.GetPixels()[0], testImgFromBGRA.GetPixels()[2]);
		EXPECT_FLOAT_EQ(testImgFromBGRA3.GetPixels()[1], testImgFromBGRA.GetPixels()[1]);
		EXPECT_FLOAT_EQ(testImgFromBGRA3.GetPixels()[2], testImgFromBGRA.GetPixels()[0]);
		EXPECT_FLOAT_EQ(testImgFromBGRA3.GetPixels()[3], testImgFromBGRA.GetPixels()[6]);
	});

	// Testing from RGB
	std::thread tRGB([&](){
		wlib::image::Image testImgBaseRGB(testImg);
		testImgBaseRGB.ConvertToRGB();
		wlib::image::Image testImgBaseRGB1(testImgBaseRGB);
		wlib::image::Image testImgBaseRGB2(testImgBaseRGB);
		wlib::image::Image testImgBaseRGB3(testImgBaseRGB);

		// RGB to BGR
		testImgBaseRGB1.ConvertToBGR();
		EXPECT_FLOAT_EQ(testImgBaseRGB1.GetPixels()[0], testImgBaseRGB.GetPixels()[2]);
		EXPECT_FLOAT_EQ(testImgBaseRGB1.GetPixels()[1], testImgBaseRGB.GetPixels()[1]);
		EXPECT_FLOAT_EQ(testImgBaseRGB1.GetPixels()[2], testImgBaseRGB.GetPixels()[0]);

		// RGB to RGBA (append alpha)
		testImgBaseRGB2.ConvertToRGBA();
		EXPECT_FLOAT_EQ(testImgBaseRGB2.GetPixels()[0], testImgBaseRGB.GetPixels()[0]);
		EXPECT_FLOAT_EQ(testImgBaseRGB2.GetPixels()[1], testImgBaseRGB.GetPixels()[1]);
		EXPECT_FLOAT_EQ(testImgBaseRGB2.GetPixels()[2], testImgBaseRGB.GetPixels()[2]);
		EXPECT_FLOAT_EQ(testImgBaseRGB2.GetPixels()[3], 255.0f);

		// RGB to BGRA (append alpha and flip)
		testImgBaseRGB3.ConvertToBGRA();
		EXPECT_FLOAT_EQ(testImgBaseRGB3.GetPixels()[0], testImgBaseRGB.GetPixels()[2]);
		EXPECT_FLOAT_EQ(testImgBaseRGB3.GetPixels()[1], testImgBaseRGB.GetPixels()[1]);
		EXPECT_FLOAT_EQ(testImgBaseRGB3.GetPixels()[2], testImgBaseRGB.GetPixels()[0]);
		EXPECT_FLOAT_EQ(testImgBaseRGB3.GetPixels()[3], 255.0f);
	});

	// Testing from BGR
	wlib::image::Image testImgBaseBGR(testImg);
	testImgBaseBGR.ConvertToBGR();
	wlib::image::Image testImgBaseBGR1(testImgBaseBGR);
	wlib::image::Image testImgBaseBGR2(testImgBaseBGR);
	wlib::image::Image testImgBaseBGR3(testImgBaseBGR);

	// BGR to RGB
	testImgBaseBGR1.ConvertToRGB();
	EXPECT_FLOAT_EQ(testImgBaseBGR1.GetPixels()[0], testImgBaseBGR.GetPixels()[2]);
	EXPECT_FLOAT_EQ(testImgBaseBGR1.GetPixels()[1], testImgBaseBGR.GetPixels()[1]);
	EXPECT_FLOAT_EQ(testImgBaseBGR1.GetPixels()[2], testImgBaseBGR.GetPixels()[0]);

	// BGR to BGRA (append alpha)
	testImgBaseBGR2.ConvertToBGRA();
	EXPECT_FLOAT_EQ(testImgBaseBGR2.GetPixels()[0], testImgBaseBGR.GetPixels()[0]);
	EXPECT_FLOAT_EQ(testImgBaseBGR2.GetPixels()[1], testImgBaseBGR.GetPixels()[1]);
	EXPECT_FLOAT_EQ(testImgBaseBGR2.GetPixels()[2], testImgBaseBGR.GetPixels()[2]);
	EXPECT_FLOAT_EQ(testImgBaseBGR2.GetPixels()[3], 255.0f);

	// BGR to RGBA (append alpha and flip)
	testImgBaseBGR3.ConvertToRGBA();
	EXPECT_FLOAT_EQ(testImgBaseBGR3.GetPixels()[0], testImgBaseBGR.GetPixels()[2]);
	EXPECT_FLOAT_EQ(testImgBaseBGR3.GetPixels()[1], testImgBaseBGR.GetPixels()[1]);
	EXPECT_FLOAT_EQ(testImgBaseBGR3.GetPixels()[2], testImgBaseBGR.GetPixels()[0]);
	EXPECT_FLOAT_EQ(testImgBaseBGR3.GetPixels()[3], 255.0f);


	tBGRA.join();
	tRGBA.join();
	tRGB.join();
}

TEST(ImgOps, PSNR) {
	const std::string imgPath = std::string(wlibTestDir) + "imgload_files/base.rawpix"s;
	const std::string imgPathJPG = std::string(wlibTestDir) + "imgload_files/base.jpg"s;

	ASSERT_TRUE(std::filesystem::exists(imgPath));
	wlib::image::Image testImg(imgPath, true, imageWidth, imageHeight, wlib::image::F_RGBA);
	wlib::image::Image testImgJPG(imgPathJPG);

	auto testImgSoA = wlib::image::MakeSoAFromAoS(testImg);
	auto testImgJPGSoA = wlib::image::MakeSoAFromAoS(testImgJPG);

	auto outData = wlib::image::CalculatePSNR<float>(testImgSoA, testImgSoA);

	// MSE of identical images should be 0
	EXPECT_FLOAT_EQ(outData.MSEPerChannel[0], 0.0f);
	EXPECT_FLOAT_EQ(outData.MSEPerChannel[1], 0.0f);
	EXPECT_FLOAT_EQ(outData.MSEPerChannel[2], 0.0f);

	// PSNR of identical images should be infinity
	EXPECT_FLOAT_EQ(outData.PSNRPerChannel[0], std::numeric_limits<float>::infinity());
	EXPECT_FLOAT_EQ(outData.PSNRPerChannel[1], std::numeric_limits<float>::infinity());
	EXPECT_FLOAT_EQ(outData.PSNRPerChannel[2], std::numeric_limits<float>::infinity());

	// Same, but with doubles
	auto outDataDbl = wlib::image::CalculatePSNR<double>(testImgSoA, testImgSoA);

	// MSE of identical images should be 0
	EXPECT_DOUBLE_EQ(outDataDbl.MSEPerChannel[0], 0.0);
	EXPECT_DOUBLE_EQ(outDataDbl.MSEPerChannel[1], 0.0);
	EXPECT_DOUBLE_EQ(outDataDbl.MSEPerChannel[2], 0.0);

	// PSNR of identical images should be infinity
	EXPECT_DOUBLE_EQ(outDataDbl.PSNRPerChannel[0], std::numeric_limits<double>::infinity());
	EXPECT_DOUBLE_EQ(outDataDbl.PSNRPerChannel[1], std::numeric_limits<double>::infinity());
	EXPECT_DOUBLE_EQ(outDataDbl.PSNRPerChannel[2], std::numeric_limits<double>::infinity());

	// Different images
	auto outDataDiff = wlib::image::CalculatePSNR(testImgSoA, testImgJPGSoA);

	EXPECT_LE(outDataDiff.MSEPerChannel[0], 2.07f);
	EXPECT_GE(outDataDiff.MSEPerChannel[0], 2.04f);

	EXPECT_LE(outDataDiff.PSNRPerChannel[0], 45.01f);
	EXPECT_GE(outDataDiff.PSNRPerChannel[0], 45.001f);

	// Different images with doubles
	auto outDataDiffDbl = wlib::image::CalculatePSNR<double>(testImgSoA, testImgJPGSoA);

	EXPECT_LE(outDataDiffDbl.MSEPerChannel[0], 2.07);
	EXPECT_GE(outDataDiffDbl.MSEPerChannel[0], 2.04);

	EXPECT_LE(outDataDiffDbl.PSNRPerChannel[0], 45.01);
	EXPECT_GE(outDataDiffDbl.PSNRPerChannel[0], 45.001);

	// TODO: Test exceptions on wrong size
	// TODO: Test on grayscale
}

#endif
