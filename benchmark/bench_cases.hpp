#pragma once
#include <string>
#include <vector>

namespace bench
{
	volatile std::string testStrings(const char* stringToTest, const size_t length);

	volatile std::string testImages(const std::vector<uint8_t>* rawPixels, const int width, const int height);
} // namespace bench
