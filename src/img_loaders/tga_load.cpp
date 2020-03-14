#ifdef WEIRDLIB_ENABLE_IMAGE_OPERATIONS
#include "image_format_loaders.hpp"
#include "img_loader_exceptions.hpp"
#include "../../include/weirdlib_bitops.hpp"

#include <utility>
#include <memory>
#include <cstring>
#include <algorithm>

using namespace std::string_literals;

namespace wlib::image
{
	enum TGAImageType
	{
		NoData = 0,
		ColorMap = 1,
		TrueColor = 2,
		Grayscale = 3,
		RLEColorMap = 9,
		RLETrueColor = 10,
		RLEGrayscale = 11
	};

	#pragma pack(push, 1)
	struct TGAColorMapInfo
	{
		uint16_t firstEntry;
		uint16_t mapLength;
		uint8_t bpp;
	};

	struct TGAImageInfo
	{
		uint16_t xOrigin;
		uint16_t yOrigin;
		uint16_t width;
		uint16_t height;
		uint8_t bpp;
		uint8_t descriptor;
	};
	#pragma pack(pop)

	static std::vector<uint8_t> LoadTrueColorTGA(uint16_t width, uint16_t height, uint8_t bpp, const uint8_t* in) {
		uint8_t bytesPerPixel = 0;
		std::vector<uint8_t> tmp;

		switch (bpp)
		{
		  case 15:
		  case 16:
			bytesPerPixel = 2;
			break;
		  case 24:
			bytesPerPixel = 3;
			break;
		  case 32:
			bytesPerPixel = 4;
			break;
		  default:
			break;
		}

		if (bytesPerPixel == 3 || bytesPerPixel == 4) {
			tmp.resize(width*height*bytesPerPixel);

			switch (bytesPerPixel)
			{
			  case 3:
				// BGR to RGB
				for (size_t i = 0; i < width*height; i++) {
					tmp[i*3] = in[i*3+2];
					tmp[i*3+1] = in[i*3+1];
					tmp[i*3+2] = in[i*3];
				}
				break;
			  case 4:
				// BGRA to RGBA
				for (size_t i = 0; i < width*height; i++) {
					tmp[i*4] = in[i*4+2];
					tmp[i*4+1] = in[i*4+1];
					tmp[i*4+2] = in[i*4];
					tmp[i*4+3] = in[i*4+3];
				}
				break;
			  default:
				break;
			}
		} else {
			// Values are scaled to the full range of 0 to 255
			if (bpp == 16) { //B5G5R5A1
				tmp.reserve(width*height*4);
				for (size_t i = 0; i < width * height * bytesPerPixel; i += 2) {
					uint8_t r = (in[i+1] >> 2) & 0x1F;
        			uint8_t g = ((in[i+1] << 3) & 0x1C) | ((in[i] >> 5) & 0x07);
        			uint8_t b = (in[i] & 0x1F);
      				uint8_t a = ((in[i+1] & 0x80) >> 7) == 1 ? 255 : 1;

					// Space out 5 bit values to 8 bit values
					r = (r << 3) | (r >> 2);
					g = (g << 3) | (g >> 2);
					b = (b << 3) | (b >> 2);

					tmp.push_back(r);
					tmp.push_back(g);
					tmp.push_back(b);
					tmp.push_back(a);
				}
			} else if (bpp == 15) { // B5G5R5
				tmp.reserve(width*height*3);
				for (size_t i = 0; i < width * height * bytesPerPixel; i += 2) {
					uint8_t r = (in[i+1] >> 2) & 0x1F;
        			uint8_t g = ((in[i+1] << 3) & 0x1C) | ((in[i] >> 5) & 0x07);
        			uint8_t b = (in[i] & 0x1F);

					// Space out 5 bit values to 8 bit values
					r = (r << 3) | (r >> 2);
					g = (g << 3) | (g >> 2);
					b = (b << 3) | (b >> 2);

					tmp.push_back(r);
					tmp.push_back(g);
					tmp.push_back(b);
				}
			}
		}

		return tmp;
	}

	static std::vector<uint8_t> LoadRLETrueColorTGA(uint16_t width, uint16_t height, uint8_t bpp, const uint8_t* in) {
		std::vector<uint8_t> tmp;

		if (bpp == 24) {
			tmp.resize(width*height*3);
			size_t totalOffset = 0;
			size_t outputOffset = 0;

			while (outputOffset < width*height*3) {
				if (bop::test(in[totalOffset], 7)) { // RLE Packet
					uint8_t repCount = bop::reset(in[totalOffset], 7)+1;

					const int r = in[totalOffset+3];
					const int g = in[totalOffset+2];
					const int b = in[totalOffset+1];

					for (size_t i = 0; i < repCount; i++) {
						tmp[outputOffset+i*3] = r;
						tmp[outputOffset+i*3+1] = g;
						tmp[outputOffset+i*3+2] = b;
					}

					totalOffset += 4;
					outputOffset += repCount*3;
				} else { // Raw packet
					uint8_t repCount = bop::reset(in[totalOffset], 7)+1;
					totalOffset++;

					for (size_t i = 0; i < repCount; i++) {
						tmp[outputOffset+i*3] = in[totalOffset+i*3+2];
						tmp[outputOffset+i*3+1] = in[totalOffset+i*3+1];
						tmp[outputOffset+i*3+2] = in[totalOffset+i*3];
					}

					totalOffset += repCount*3;
					outputOffset += repCount*3;
				}
			}
		} else if (bpp == 32) {
			tmp.resize(width*height*4);
			size_t totalOffset = 0;
			size_t outputOffset = 0;

			while (outputOffset < width*height*4) {
				if (bop::test(in[totalOffset], 7)) { // RLE Packet
					uint8_t repCount = bop::reset(in[totalOffset], 7)+1;

					const int a = in[totalOffset+4];
					const int r = in[totalOffset+3];
					const int g = in[totalOffset+2];
					const int b = in[totalOffset+1];

					for (size_t i = 0; i < repCount; i++) {
						tmp[outputOffset+i*4] = r;
						tmp[outputOffset+i*4+1] = g;
						tmp[outputOffset+i*4+2] = b;
						tmp[outputOffset+i*4+3] = a;
					}

					totalOffset += 5;
					outputOffset += repCount*4;
				} else { // Raw packet
					uint8_t repCount = bop::reset(in[totalOffset], 7)+1;
					totalOffset++;

					for (size_t i = 0; i < repCount; i++) {
						tmp[outputOffset+i*4] = in[totalOffset+i*4+2];
						tmp[outputOffset+i*4+1] = in[totalOffset+i*4+1];
						tmp[outputOffset+i*4+2] = in[totalOffset+i*4];
						tmp[outputOffset+i*4+3] = in[totalOffset+i*4+3];
					}

					totalOffset += repCount*4;
					outputOffset += repCount*4;
				}
			}
		} else if (bpp == 16) {
			tmp.resize(width*height*4);
			size_t totalOffset = 0;
			size_t outputOffset = 0;

			while (outputOffset < width*height*4) {
				if (bop::test(in[totalOffset], 7)) { // RLE Packet
					uint8_t repCount = bop::reset(in[totalOffset], 7)+1;

					uint8_t r = (in[totalOffset+2] >> 2) & 0x1F;
					uint8_t g = ((in[totalOffset+2] << 3) & 0x1C) | ((in[totalOffset+1] >> 5) & 0x07);
					uint8_t b = (in[totalOffset+1] & 0x1F);
					uint8_t a = ((in[totalOffset+2] & 0x80) >> 7) == 1 ? 255 : 1;

					// Space out 5 bit values to 8 bit values
					r = (r << 3) | (r >> 2);
					g = (g << 3) | (g >> 2);
					b = (b << 3) | (b >> 2);

					for (size_t i = 0; i < repCount; i++) {
						tmp[outputOffset+i*4] = r;
						tmp[outputOffset+i*4+1] = g;
						tmp[outputOffset+i*4+2] = b;
						tmp[outputOffset+i*4+3] = a;
					}

					totalOffset += 3;
					outputOffset += repCount*4;
				} else { // Raw packet
					uint8_t repCount = bop::reset(in[totalOffset], 7)+1;
					totalOffset++;

					for (size_t i = 0; i < repCount; i++) {
						uint8_t r = (in[totalOffset+i*2+1] >> 2) & 0x1F;
						uint8_t g = ((in[totalOffset+i*2+1] << 3) & 0x1C) | ((in[totalOffset+i*2] >> 5) & 0x07);
						uint8_t b = (in[totalOffset+i*2] & 0x1F);
						uint8_t a = ((in[totalOffset+i*2+1] & 0x80) >> 7) == 1 ? 255 : 1;

						// Space out 5 bit values to 8 bit values
						r = (r << 3) | (r >> 2);
						g = (g << 3) | (g >> 2);
						b = (b << 3) | (b >> 2);

						tmp[outputOffset+i*4] = r;
						tmp[outputOffset+i*4+1] = g;
						tmp[outputOffset+i*4+2] = b;
						tmp[outputOffset+i*4+3] = a;
					}

					totalOffset += repCount*2;
					outputOffset += repCount*4;
				}
			}
		} else if (bpp == 15) {
			tmp.resize(width*height*3);
			size_t totalOffset = 0;
			size_t outputOffset = 0;

			while (outputOffset < width*height*3) {
				if (bop::test(in[totalOffset], 7)) { // RLE Packet
					uint8_t repCount = bop::reset(in[totalOffset], 7)+1;

					uint8_t r = (in[totalOffset+2] >> 2) & 0x1F;
					uint8_t g = ((in[totalOffset+2] << 3) & 0x1C) | ((in[totalOffset+1] >> 5) & 0x07);
					uint8_t b = (in[totalOffset+1] & 0x1F);

					// Space out 5 bit values to 8 bit values
					r = (r << 3) | (r >> 2);
					g = (g << 3) | (g >> 2);
					b = (b << 3) | (b >> 2);

					for (size_t i = 0; i < repCount; i++) {
						tmp[outputOffset+i*3] = r;
						tmp[outputOffset+i*3+1] = g;
						tmp[outputOffset+i*3+2] = b;
					}

					totalOffset += 3;
					outputOffset += repCount*3;
				} else { // Raw packet
					uint8_t repCount = bop::reset(in[totalOffset], 7)+1;
					totalOffset++;

					for (size_t i = 0; i < repCount; i++) {
						uint8_t r = (in[totalOffset+i*2+1] >> 2) & 0x1F;
						uint8_t g = ((in[totalOffset+i*2+1] << 3) & 0x1C) | ((in[totalOffset+i*2] >> 5) & 0x07);
						uint8_t b = (in[totalOffset+i*2] & 0x1F);

						// Space out 5 bit values to 8 bit values
						r = (r << 3) | (r >> 2);
						g = (g << 3) | (g >> 2);
						b = (b << 3) | (b >> 2);

						tmp[outputOffset+i*3] = r;
						tmp[outputOffset+i*3+1] = g;
						tmp[outputOffset+i*3+2] = b;
					}

					totalOffset += repCount*2;
					outputOffset += repCount*3;
				}
			}
		}

		return tmp;
	}

	static std::vector<uint8_t> LoadRLEGrayscaleTGA(uint16_t width, uint16_t height, const uint8_t* in) {
		std::vector<uint8_t> tmp(width*height);
		size_t totalOffset = 0;
		size_t outputOffset = 0;

		while (outputOffset < width*height) {
			if (bop::test(in[totalOffset], 7)) { // RLE Packet
				uint8_t repCount = bop::reset(in[totalOffset], 7)+1;

				std::generate_n(tmp.begin()+outputOffset, repCount, [gray=in[totalOffset+1]]() {return gray;});

				totalOffset+=2;
				outputOffset+=repCount;
			} else { // Raw packet
				uint8_t repCount = bop::reset(in[totalOffset], 7)+1;
				totalOffset++;

				std::generate_n(tmp.begin()+outputOffset, repCount, [&,i=0]() mutable {return in[totalOffset + i++];});

				totalOffset+=repCount;
				outputOffset+=repCount;
			}
		}

		return tmp;
	}

	ImageInfoTGA LoadTGA(const std::string& path) {
		std::ifstream in(path, std::ios::binary);
		if (!in.good()) {
			throw except::file_open_error("TGA Loader: Failed to open file "s + path);
		}

		return LoadTGA(in);
	}

	ImageInfoTGA LoadTGA(std::ifstream& in) {
		in.seekg(0, std::ios::end);
		size_t fileSize = in.tellg();
		if (fileSize == 0) {
			throw except::file_open_error("TGA Loader: File size returned was 0");
		}

		in.seekg(0);

		auto fileBytes = std::make_unique<uint8_t[]>(fileSize);
		in.read(reinterpret_cast<char*>(fileBytes.get()), fileSize);

		return LoadTGA(fileBytes.get(), fileSize);
	}

	ImageInfoTGA LoadTGA(const uint8_t* in, size_t /*size*/) {
		if (in == nullptr) {
			throw except::invalid_image_data("TGA Loader: Image pointer was null");
		}

		ImageInfoTGA info;
		size_t currentOffset = 0;

		// Read header info
		uint8_t idFieldLength;
		uint8_t colorMapType;
		uint8_t imageType;

		TGAColorMapInfo colorMapInfo;
		TGAImageInfo imageInfo;

		std::memcpy(&idFieldLength, in+currentOffset, 1);
		currentOffset++;

		std::memcpy(&colorMapType, in+currentOffset, 1);
		currentOffset++;

		std::memcpy(&imageType, in+currentOffset, 1);
		currentOffset++;

		std::memcpy(&colorMapInfo, in+currentOffset, 5);
		currentOffset += 5;

		std::memcpy(&imageInfo, in+currentOffset, 10);
		currentOffset += 10;

		// Skip image ID
		currentOffset += idFieldLength;

		// TODO: Color mapped images
		currentOffset += colorMapInfo.mapLength;
		if (colorMapType != 0) {
			throw except::unsupported_image_type("TGA Loader: Color mapped image not yet supported");
		}

		// Set image parameters
		info.width = imageInfo.width;
		info.height = imageInfo.height;
		info.xOrigin = imageInfo.xOrigin;
		info.yOrigin = imageInfo.yOrigin;
		info.bpp = imageInfo.bpp;

		switch (imageType)
		{
		  case NoData:
			throw except::invalid_image_data("TGA Loader: Image type was 0 (no image data)");
		  case ColorMap:
		  case RLEColorMap:
			throw except::unsupported_image_type("TGA Loader: Color mapped image are not yet supported");
		  case RLETrueColor:
			info.data = LoadRLETrueColorTGA(info.width, info.height, info.bpp, in+currentOffset);
			break;
		  case RLEGrayscale:
			info.data = LoadRLEGrayscaleTGA(info.width, info.height, in+currentOffset);
			break;
		  case TrueColor: {
			info.data = LoadTrueColorTGA(info.width, info.height, info.bpp, in+currentOffset);
			break;
		  }
		  case Grayscale: {
			info.data.resize(info.width*info.height);
			std::memcpy(info.data.data(), in+currentOffset, info.width*info.height);
			break;
		  }
		  default:
			throw except::invalid_image_data("TGA Loader: Image type was outside of spec bounds");
		}

		return info;
	}
} // namespace wlib::image
#endif
