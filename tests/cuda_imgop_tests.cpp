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
	wlib::image::cu::ImageSoACUDA testSoANA(testSoAA);

	wlib::image::cu::ImageSoACUDA testSoA_Average(testSoAA);
	wlib::image::cu::ImageSoACUDA testSoA_Lightness(testSoAA);
	wlib::image::cu::ImageSoACUDA testSoA_LumBT601(testSoAA);

	testSoAA = wlib::image::cu::ConvertToGrayscale(testSoAA, true);
	EXPECT_EQ(testSoAA.channels.size(), 2);

	testSoANA = wlib::image::cu::ConvertToGrayscale(testSoANA, false);
	EXPECT_EQ(testSoANA.channels.size(), 1);

	testSoA_Average = wlib::image::cu::ConvertToGrayscale(testSoA_Average, false, wlib::image::GrayscaleMethod::Average);
	testSoA_Lightness = wlib::image::cu::ConvertToGrayscale(testSoA_Lightness, false, wlib::image::GrayscaleMethod::Lightness);
	testSoA_LumBT601 = wlib::image::cu::ConvertToGrayscale(testSoA_LumBT601, false, wlib::image::GrayscaleMethod::LuminosityBT601);
}

constexpr size_t PixelSampleCount = 3000;

TEST(CUDAImage, NegateValues) {
	const std::string imgPath = std::string(wlibTestDir) + std::string("testphoto.rawpix");
	ASSERT_TRUE(std::filesystem::exists(imgPath));

	wlib::image::Image testImg(imgPath, true, imageWidth, imageHeight, wlib::image::F_RGBA);

	wlib::image::cu::ImageSoACUDA testImgDev(testImg);
	wlib::image::cu::ImageSoACUDA testImgDev_alpha(testImgDev);

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

TEST(CUDAImage, Histogram) {
	const std::string imgPath = std::string(wlibTestDir) + std::string("testphoto.rawpix");
	ASSERT_TRUE(std::filesystem::exists(imgPath));

	wlib::image::Image testImg(imgPath, true, imageWidth, imageHeight, wlib::image::F_RGBA);
	wlib::image::cu::ImageSoACUDA testRGBA(testImg);

	auto histogram = wlib::image::cu::GetHistogram(testRGBA);

	EXPECT_TRUE(histogram.Gray.empty());
	EXPECT_FALSE(histogram.Red.empty());
	EXPECT_FALSE(histogram.Green.empty());
	EXPECT_FALSE(histogram.Blue.empty());
	EXPECT_FALSE(histogram.Alpha.empty());
}

TEST(CUDAImage, DeviceSoA) {
	const std::string imgPath = std::string(wlibTestDir) + std::string("testphoto.rawpix");
	ASSERT_TRUE(std::filesystem::exists(imgPath));

	// From image file
	wlib::image::Image testImg_baseAoS(imgPath, true, imageWidth, imageHeight, wlib::image::F_RGBA);
	auto pixref = testImg_baseAoS.GetPixelsAsInt();

	wlib::image::cu::ImageSoACUDA testImg(testImg_baseAoS);
	auto pixconv = testImg.ConvertToImage().GetPixelsAsInt();

	EXPECT_TRUE(std::equal(pixconv.begin(), pixconv.end(), pixref.begin()));

	// Copy constructed
	wlib::image::cu::ImageSoACUDA testImgCopyConstructed(testImg);

	for (const auto& chan : testImg.channels) {
		EXPECT_NE(chan, nullptr);
	}

	auto copypix = testImgCopyConstructed.ConvertToImage().GetPixelsAsInt();

	EXPECT_TRUE(std::equal(copypix.begin(), copypix.end(), pixref.begin()));

	// Move constructed
	wlib::image::cu::ImageSoACUDA testImgMoveConstructed(std::move(testImg));

	for (const auto& chan : testImg.channels) {
		EXPECT_EQ(chan, nullptr);
	}

	auto movepix = testImgMoveConstructed.ConvertToImage().GetPixelsAsInt();

	EXPECT_TRUE(std::equal(movepix.begin(), movepix.end(), pixref.begin()));
}

TEST(CUDAImage, DeviceAoS) {
	const std::string imgPath = std::string(wlibTestDir) + std::string("testphoto.rawpix");
	ASSERT_TRUE(std::filesystem::exists(imgPath));

	// From image file
	wlib::image::Image testImg_base(imgPath, true, imageWidth, imageHeight, wlib::image::F_RGBA);
	wlib::image::cu::ImageCUDA testImg(imgPath, true, imageWidth, imageHeight, wlib::image::F_RGBA);

	auto hostpix = testImg_base.GetPixelsAsInt();
	auto devpix = testImg.GetPixelsAsInt();

	EXPECT_TRUE(std::equal(hostpix.begin(), hostpix.end(), devpix.begin()));

	// Copy constructed
	wlib::image::cu::ImageCUDA testImgCopyConstructed(testImg);

	EXPECT_NE(testImg.pixels, nullptr);
	auto copypix = testImgCopyConstructed.GetPixelsAsInt();

	EXPECT_TRUE(std::equal(hostpix.begin(), hostpix.end(), copypix.begin()));

	// Move constructed
	wlib::image::cu::ImageCUDA testImgMoveConstructed(std::move(testImg));

	EXPECT_EQ(testImg.pixels, nullptr);
	auto movepix = testImgMoveConstructed.GetPixelsAsInt();

	EXPECT_TRUE(std::equal(hostpix.begin(), hostpix.end(), movepix.begin()));
}

TEST(CUDAImage, RandomImage) {
	constexpr int genWidth =  1024;
	constexpr int genheight = 768;
	auto imgsoa0 = wlib::image::cu::GenerateRandomImageSoA(genWidth, genheight, wlib::image::ColorFormat::F_RGB);
	auto imgsoa1 = wlib::image::cu::GenerateRandomImageSoA(genWidth, genheight, wlib::image::ColorFormat::F_RGBA);
}

#endif
