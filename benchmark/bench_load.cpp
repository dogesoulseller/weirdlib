#ifdef WEIRDLIB_ENABLE_IMAGE_OPERATIONS
#include "bench_load.hpp"
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Weverything"
#include "../external/stb/stb_image.h"
#pragma clang diagnostic pop
#include <functional>

namespace bench
{
	std::vector<uint8_t>* loadImage(const std::filesystem::path& imagePath, int channels) {
		auto pathTmp = imagePath.c_str();

		int width; int height;

		auto pix = stbi_load(pathTmp, &width, &height, nullptr, channels);
		auto outputPixels = new std::vector<uint8_t>(pix, pix+width*height*channels);
		stbi_image_free(pix);

		return outputPixels;
	}

} // namespace bench

#endif
