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

	/// Load BMP file <br>
	/// Supports 24bpp BGR without compression only
	/// @param path file path to location on disk
	/// @return @{link ImageInfo}
	ImageInfo LoadBMP(const std::string& path);

	/// Load BMP file <br>
	///	Supports 24bpp BGR without compression only
	/// @param in fstream (must be opened using binary mode, otherwise behavior is undefined)
	/// @return @{link ImageInfo}
	ImageInfo LoadBMP(std::ifstream& in);

	/// Load BMP file <br>
	///	Supports 24bpp BGR without compression only
	/// @param in pointer to start of memory with a BMP file's contents
	/// @param size size in bytes of data
	/// @return @{link ImageInfo}
	ImageInfo LoadBMP(const uint8_t* in);


	struct ImageInfoPNM
	{
		size_t width;
		size_t height;
		uint8_t maxValue;
		uint8_t colorChannels;
		std::vector<uint8_t> pixels;
	};

	/// Load PNM (Portable aNy Map) file
	/// @param path file path to location on disk
	/// @return @{link ImageInfoPNM}
	ImageInfoPNM LoadPNM(const std::string& path);

	/// Load PNM (Portable aNy Map) file
	/// @param in fstream (must be opened using binary mode, otherwise behavior is undefined)
	/// @return @{link ImageInfoPNM}
	ImageInfoPNM LoadPNM(std::ifstream& in);

	/// Load PNM (Portable aNy Map) file
	/// @param in pointer to start of memory with a PNM file's contents
	/// @param size size in bytes of data
	/// @return @{link ImageInfoPNM}
	ImageInfoPNM LoadPNM(const uint8_t* in, size_t size);


	struct ImageInfoPAM
	{
		size_t width;
		size_t height;
		uint16_t maxValue;
		uint8_t colorChannels;
		std::vector<uint16_t> pixels;
	};

	/// Load PAM (Portable Arbitrary Map) file
	/// @param path file path to location on disk
	/// @return @{link ImageInfoPAM}
	ImageInfoPAM LoadPAM(const std::string& path);

	/// Load PAM (Portable Arbitrary Map) file
	/// @param in fstream (must be opened using binary mode, otherwise behavior is undefined)
	/// @return @{link ImageInfoPAM}
	ImageInfoPAM LoadPAM(std::ifstream& in);

	/// Load PAM (Portable Arbitrary Map) file
	/// @param in pointer to start of memory with a PAM file's contents
	/// @param size size in bytes of data
	/// @return @{link ImageInfoPAM}
	ImageInfoPAM LoadPAM(const uint8_t* in, size_t size);


	struct ImageInfoPFM
	{
		size_t width;
		size_t height;
		bool isLittleEndian;
		std::vector<float> data;
	};

	/// Load PFM (Portable Float Map) file <br>
	/// Warning: This function is untested and might not work correctly 100% of the time
	/// @param path file path to location on disk
	/// @return @{link ImageInfoPFM}
	ImageInfoPFM LoadPFM(const std::string& path);

	/// Load PFM (Portable Float Map) file <br>
	/// Warning: This function is untested and might not work correctly 100% of the time
	/// @param in fstream (must be opened using binary mode, otherwise behavior is undefined)
	/// @return @{link ImageInfoPFM}
	ImageInfoPFM LoadPFM(std::ifstream& in);

	/// Load PFM (Portable Float Map) file <br>
	/// Warning: This function is untested and might not work correctly 100% of the time
	/// @param in pointer to start of memory with a PFM file's contents
	/// @param size size in bytes of data
	/// @return @{link ImageInfoPFM}
	ImageInfoPFM LoadPFM(const uint8_t* in, size_t size = 0);

} // namespace wlib::image
