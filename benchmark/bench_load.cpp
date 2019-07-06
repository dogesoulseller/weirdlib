#include "bench_load.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "../external/stb/stb_image.h"

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
