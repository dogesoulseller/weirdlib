#pragma once
#include <cstddef>
#include <cstdint>

#include <tuple>
#include <vector>
#include <string>
#include <fstream>

namespace wlib::image
{
	struct ImageInfo
	{
		size_t width;
		size_t height;
		uint8_t colorChannels;
		std::vector<uint8_t> pixels;
	};

	ImageInfo LoadBMP(const std::string& path);

	ImageInfo LoadBMP(std::ifstream& in);

	ImageInfo LoadBMP(const uint8_t* in);
} // namespace wlib::image
