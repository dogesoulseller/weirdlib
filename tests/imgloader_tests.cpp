#ifdef WEIRDLIB_ENABLE_IMAGE_OPERATIONS
#include <gtest/gtest.h>
#include "../src/img_loaders/image_format_loaders.hpp"
#include <filesystem>
#include <algorithm>

#include "../external/stb/stb_image.h"

#include "pnmdata.hpp"

constexpr const char* wlibTestDir = WLIBTEST_TESTING_DIRECTORY;
constexpr int expectedWidthPNM = 27;
constexpr int expectedHeightPNM = 12;
constexpr int expectedMaxValPNM = 255;

constexpr uint64_t imageWidth = 64;
constexpr uint64_t imageHeight = 64;

TEST(ImgOps_Loader, PNM_Binary) {
	std::string pathPBM = std::filesystem::path(wlibTestDir) / "fileop_files" / "pbm.pbm";
	std::string pathPGM = std::filesystem::path(wlibTestDir) / "fileop_files" / "pgm.pgm";
	std::string pathPPM = std::filesystem::path(wlibTestDir) / "fileop_files" / "ppm.ppm";

	auto infoPBM = wlib::image::LoadPNM(pathPBM);
	auto infoPGM = wlib::image::LoadPNM(pathPGM);
	auto infoPPM = wlib::image::LoadPNM(pathPPM);

	EXPECT_EQ(infoPBM.width, expectedWidthPNM);
	EXPECT_EQ(infoPGM.width, expectedWidthPNM);
	EXPECT_EQ(infoPPM.width, expectedWidthPNM);

	EXPECT_EQ(infoPBM.height, expectedHeightPNM);
	EXPECT_EQ(infoPGM.height, expectedHeightPNM);
	EXPECT_EQ(infoPPM.height, expectedHeightPNM);

	EXPECT_EQ(infoPGM.maxValue, expectedMaxValPNM);
	EXPECT_EQ(infoPPM.maxValue, expectedMaxValPNM);

	EXPECT_EQ(infoPBM.colorChannels, 1);
	EXPECT_EQ(infoPGM.colorChannels, 1);
	EXPECT_EQ(infoPPM.colorChannels, 3);

	for (size_t i = 0; i < ExpectedDataPBM.size(); i++) {
		if (ExpectedDataPBM[i] != infoPBM.pixels[i]) {
			std::cerr << "Mismatch at " << i << std::endl;
		}
	}

	EXPECT_TRUE(std::equal(infoPBM.pixels.begin(), infoPBM.pixels.end(), ExpectedDataPBM.begin()));
	EXPECT_TRUE(std::equal(infoPGM.pixels.begin(), infoPGM.pixels.end(), ExpectedDataPGM.begin()));
	EXPECT_TRUE(std::equal(infoPPM.pixels.begin(), infoPPM.pixels.end(), ExpectedDataPPM.begin()));
}

TEST(ImgOps_Loader, PNM_ASCII) {
	std::string pathPBM = std::filesystem::path(wlibTestDir) / "fileop_files" / "pbm_ascii.pbm";
	std::string pathPGM = std::filesystem::path(wlibTestDir) / "fileop_files" / "pgm_ascii.pgm";
	std::string pathPPM = std::filesystem::path(wlibTestDir) / "fileop_files" / "ppm_ascii.ppm";

	auto infoPBM = wlib::image::LoadPNM(pathPBM);
	auto infoPGM = wlib::image::LoadPNM(pathPGM);
	auto infoPPM = wlib::image::LoadPNM(pathPPM);

	EXPECT_EQ(infoPBM.width, expectedWidthPNM);
	EXPECT_EQ(infoPGM.width, expectedWidthPNM);
	EXPECT_EQ(infoPPM.width, expectedWidthPNM);

	EXPECT_EQ(infoPBM.height, expectedHeightPNM);
	EXPECT_EQ(infoPGM.height, expectedHeightPNM);
	EXPECT_EQ(infoPPM.height, expectedHeightPNM);

	EXPECT_EQ(infoPGM.maxValue, expectedMaxValPNM);
	EXPECT_EQ(infoPPM.maxValue, expectedMaxValPNM);

	EXPECT_EQ(infoPBM.colorChannels, 1);
	EXPECT_EQ(infoPGM.colorChannels, 1);
	EXPECT_EQ(infoPPM.colorChannels, 3);

	EXPECT_TRUE(std::equal(infoPBM.pixels.begin(), infoPBM.pixels.end(), ExpectedDataPBM.begin(), ExpectedDataPBM.end()));
	EXPECT_TRUE(std::equal(infoPGM.pixels.begin(), infoPGM.pixels.end(), ExpectedDataPGM.begin(), ExpectedDataPGM.end()));
	EXPECT_TRUE(std::equal(infoPPM.pixels.begin(), infoPPM.pixels.end(), ExpectedDataPPM.begin(), ExpectedDataPPM.end()));
}

TEST(ImgOps_Loader, PAM) {
	std::string pathPAM = std::filesystem::path(wlibTestDir) / "fileop_files" / "pam.pam";
	auto infoPAM = wlib::image::LoadPAM(pathPAM);

	EXPECT_EQ(infoPAM.width, expectedWidthPNM);
	EXPECT_EQ(infoPAM.height, expectedHeightPNM);
	EXPECT_EQ(infoPAM.maxValue, expectedMaxValPNM);
	EXPECT_EQ(infoPAM.colorChannels, 3);
}

const std::string pathBase = std::filesystem::path(wlibTestDir) / "imgload_files" / "base.bmp";

TEST(ImgOps_Loader, TGA) {
	ASSERT_TRUE(std::filesystem::exists(pathBase));

	const std::string pathTGA = std::filesystem::path(wlibTestDir) / "imgload_files" / "tga_color_nocompress.tga";
	const std::string pathTGAAlpha = std::filesystem::path(wlibTestDir) / "imgload_files" / "tga_colora_nocompress.tga";
	const std::string pathTGA555 = std::filesystem::path(wlibTestDir) / "imgload_files" / "tga_color555_nocompress.tga";

	ASSERT_TRUE(std::filesystem::exists(pathTGA));
	ASSERT_TRUE(std::filesystem::exists(pathTGAAlpha));
	ASSERT_TRUE(std::filesystem::exists(pathTGA555));

	auto infoTGA = wlib::image::LoadTGA(pathTGA);
	auto infoTGAAlpha = wlib::image::LoadTGA(pathTGAAlpha);
	auto infoTGA555 = wlib::image::LoadTGA(pathTGA555);

	EXPECT_EQ(infoTGA.bpp, 24);
	EXPECT_EQ(infoTGA.width, imageWidth);
	EXPECT_EQ(infoTGA.height, imageHeight);

	EXPECT_EQ(infoTGAAlpha.bpp, 32);
	EXPECT_EQ(infoTGAAlpha.width, imageWidth);
	EXPECT_EQ(infoTGAAlpha.height, imageHeight);

	EXPECT_EQ(infoTGA555.bpp, 16);
	EXPECT_EQ(infoTGA555.width, imageWidth);
	EXPECT_EQ(infoTGA555.height, imageHeight);

	int x, y, c;
	auto basePix = stbi_load(pathBase.c_str(), &x, &y, &c, 3);
	auto basePixAlpha = stbi_load(pathBase.c_str(), &x, &y, &c, 4);

	EXPECT_TRUE(std::equal(basePix, basePix+x*y*3, infoTGA.data.data()));
	EXPECT_TRUE(std::equal(basePixAlpha, basePixAlpha+x*y*4, infoTGAAlpha.data.data()));
}

TEST(ImgOps_Loader, TGA_RLE) {
	ASSERT_TRUE(std::filesystem::exists(pathBase));

	const std::string pathRLETGA = std::filesystem::path(wlibTestDir) / "imgload_files" / "tga_color_compress.tga";
	const std::string pathRLETGAAlpha = std::filesystem::path(wlibTestDir) / "imgload_files" / "tga_colora_compress.tga";
	const std::string pathRLETGA555 = std::filesystem::path(wlibTestDir) / "imgload_files" / "tga_color555_compress.tga";

	ASSERT_TRUE(std::filesystem::exists(pathRLETGA));
	ASSERT_TRUE(std::filesystem::exists(pathRLETGAAlpha));
	ASSERT_TRUE(std::filesystem::exists(pathRLETGA555));

	auto infoRLETGA = wlib::image::LoadTGA(pathRLETGA);
	auto infoRLETGAAlpha = wlib::image::LoadTGA(pathRLETGAAlpha);
	auto infoRLETGA555 = wlib::image::LoadTGA(pathRLETGA555);

	EXPECT_EQ(infoRLETGA.bpp, 24);
	EXPECT_EQ(infoRLETGA.width, imageWidth);
	EXPECT_EQ(infoRLETGA.height, imageHeight);

	EXPECT_EQ(infoRLETGAAlpha.bpp, 32);
	EXPECT_EQ(infoRLETGAAlpha.width, imageWidth);
	EXPECT_EQ(infoRLETGAAlpha.height, imageHeight);

	EXPECT_EQ(infoRLETGA555.bpp, 16);
	EXPECT_EQ(infoRLETGA555.width, imageWidth);
	EXPECT_EQ(infoRLETGA555.height, imageHeight);

	int x, y, c;
	auto basePix = stbi_load(pathBase.c_str(), &x, &y, &c, 3);
	auto basePixAlpha = stbi_load(pathBase.c_str(), &x, &y, &c, 4);

	EXPECT_TRUE(std::equal(basePix, basePix+x*y*3, infoRLETGA.data.data()));
	EXPECT_TRUE(std::equal(basePixAlpha, basePixAlpha+x*y*4, infoRLETGAAlpha.data.data()));
}

#endif
