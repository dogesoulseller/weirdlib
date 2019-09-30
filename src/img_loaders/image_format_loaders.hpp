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

	struct ImageInfoPNM
	{
		size_t width;
		size_t height;
		uint8_t maxValue;
		uint8_t colorChannels;
		std::vector<uint8_t> pixels;
	};

	ImageInfoPNM LoadPNM(const std::string& path);
	ImageInfoPNM LoadPNM(std::ifstream& in);
	ImageInfoPNM LoadPNM(const uint8_t* in, size_t size);

	struct ImageInfoPAM
	{
		size_t width;
		size_t height;
		uint16_t maxValue;
		uint8_t colorChannels;
		std::vector<uint16_t> pixels;
	};

	ImageInfoPAM LoadPAM(const std::string& path);
	ImageInfoPAM LoadPAM(std::ifstream& in);
	ImageInfoPAM LoadPAM(const uint8_t* in, size_t size);
} // namespace wlib::image
