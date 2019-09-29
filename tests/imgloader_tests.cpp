#ifdef WEIRDLIB_ENABLE_IMAGE_OPERATIONS
#include <gtest/gtest.h>
#include "../src/img_loaders/image_format_loaders.hpp"
#include <filesystem>
#include <algorithm>

#include "../external/stb/stb_image.h"

constexpr const char* wlibTestDir = WLIBTEST_TESTING_DIRECTORY;

TEST(ImgOps_Loader, BMP) {
	std::string path = std::filesystem::path(wlibTestDir) / "fileop_files" / "bmp.bmp";
	auto info = wlib::image::LoadBMP(path.c_str());

	int width, height;
	auto pix = stbi_load(path.c_str(), &width, &height, nullptr, 3);

	EXPECT_EQ(width, info.width);
	EXPECT_EQ(height, info.height);
	EXPECT_TRUE(std::equal(info.pixels.data(), info.pixels.data()+info.pixels.size(), pix));

	stbi_image_free(pix);
}

#endif
