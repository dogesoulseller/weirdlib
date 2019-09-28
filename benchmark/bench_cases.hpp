#pragma once
#ifdef WEIRDLIB_ENABLE_IMAGE_OPERATIONS
#include <string>
#include <vector>

namespace bench
{
	std::string testStrings(const char* stringToTest, const size_t length);

	std::string testImages(const std::vector<uint8_t>* rawPixels, const int width, const int height);
} // namespace bench
#endif
