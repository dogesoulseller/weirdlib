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

#endif
