#ifdef WLIB_ENABLE_CUDA
#include <gtest/gtest.h>
#include "../include/cuda/weirdlib_cuda_image.hpp"
#include <filesystem>
#include <future>
#include <random>

constexpr const char* wlibTestDir = WLIBTEST_TESTING_DIRECTORY;

constexpr uint64_t imageWidth = 1200;
constexpr uint64_t imageHeight = 1500;

TEST(CUDAImage, Grayscale) {
	const std::string imgPath = std::string(wlibTestDir) + std::string("testphoto.rawpix");
	ASSERT_TRUE(std::filesystem::exists(imgPath));

	wlib::image::Image testImg(imgPath, true, imageWidth, imageHeight, wlib::image::F_RGBA);
	wlib::image::cu::ImageSoACUDA testSoAA(testImg);
	wlib::image::cu::ImageSoACUDA testSoANA(testImg);

	wlib::image::cu::ImageSoACUDA testSoA_Average(testImg);
	wlib::image::cu::ImageSoACUDA testSoA_Lightness(testImg);

	testSoAA = wlib::image::cu::ConvertToGrayscale(testSoAA, true);
	EXPECT_EQ(testSoAA.channels.size(), 2);

	testSoANA = wlib::image::cu::ConvertToGrayscale(testSoANA, false);
	EXPECT_EQ(testSoANA.channels.size(), 1);

	testSoA_Average = wlib::image::cu::ConvertToGrayscale(testSoA_Average, false, wlib::image::GrayscaleMethod::Average);
	testSoA_Lightness = wlib::image::cu::ConvertToGrayscale(testSoA_Lightness, false, wlib::image::GrayscaleMethod::Lightness);
}

constexpr size_t PixelSampleCount = 3000;

TEST(CUDAImage, NegateValues) {
	const std::string imgPath = std::string(wlibTestDir) + std::string("testphoto.rawpix");
	ASSERT_TRUE(std::filesystem::exists(imgPath));

	wlib::image::Image testImg(imgPath, true, imageWidth, imageHeight, wlib::image::F_RGBA);

	wlib::image::cu::ImageSoACUDA testImgDev(testImg);
	wlib::image::cu::ImageSoACUDA testImgDev_alpha(testImg);

	wlib::image::cu::NegateValues(testImgDev);
	wlib::image::cu::NegateValues(testImgDev_alpha, true);

	auto testImgHost = testImgDev.ConvertToImageSoA();
	auto testImgHost_alpha = testImgDev_alpha.ConvertToImageSoA();

	std::random_device dev;
	std::mt19937_64 rng(dev());
	std::uniform_int_distribution<uint64_t> dist(0, testImgHost.width * testImgHost.height - 1);

	std::array<uint64_t, PixelSampleCount> Samples;
	std::generate(Samples.begin(), Samples.end(), [&dist, &rng](){return dist(rng);});

	for (size_t i = 0; i < Samples.size(); i++) {
		EXPECT_FLOAT_EQ(testImgHost.channels[0][i], 255.0f - testImg.GetPixels()[i*4]);
		EXPECT_FLOAT_EQ(testImgHost.channels[1][i], 255.0f - testImg.GetPixels()[i*4+1]);
		EXPECT_FLOAT_EQ(testImgHost.channels[2][i], 255.0f - testImg.GetPixels()[i*4+2]);
		EXPECT_FLOAT_EQ(testImgHost.channels[3][i], testImg.GetPixels()[i*4+3]);

		EXPECT_FLOAT_EQ(testImgHost_alpha.channels[0][i], 255.0f - testImg.GetPixels()[i*4]);
		EXPECT_FLOAT_EQ(testImgHost_alpha.channels[1][i], 255.0f - testImg.GetPixels()[i*4+1]);
		EXPECT_FLOAT_EQ(testImgHost_alpha.channels[2][i], 255.0f - testImg.GetPixels()[i*4+2]);
		EXPECT_FLOAT_EQ(testImgHost_alpha.channels[3][i], 255.0f - testImg.GetPixels()[i*4+3]);
	}

}

#endif