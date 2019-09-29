#include "image_format_loaders.hpp"
#include "img_loader_exceptions.hpp"

#include <memory>
#include <utility>
#include <stdexcept>
#include <cstring>

namespace wlib::image
{
	ImageInfo LoadBMP(const std::string& path) {
		std::ifstream in(path, std::ios::binary);
		return LoadBMP(in);
	}

	ImageInfo LoadBMP(std::ifstream& in) {
		in.seekg(0, std::ios::end);
		size_t fileSize = in.tellg();
		in.seekg(0);

		auto fileBytes = std::make_unique<uint8_t[]>(fileSize);
		in.read(reinterpret_cast<char*>(fileBytes.get()), fileSize);

		return LoadBMP(fileBytes.get());
	}

	ImageInfo LoadBMP(const uint8_t* in) {
		ImageInfo info;

		size_t currentOffset = 0;
		uint32_t pixelDataOffset;
		uint32_t sizeOfBMP;
		uint32_t sizeOfHeader;
		int32_t width;
		int32_t height;

		// Skip header identification
		currentOffset+=2;

		// Load BMP file size
		std::memcpy(&sizeOfBMP, in+currentOffset, 4);
		currentOffset+=4;

		// Skip reserved values
		currentOffset+=4;

		// Pixel offset
		std::memcpy(&pixelDataOffset, in+currentOffset, 4);
		currentOffset+=4;

		// Load header size
		std::memcpy(&sizeOfHeader, in+currentOffset, 4);
		currentOffset+=4;

		if (sizeOfHeader < 40) {
			throw except::unsupported_image_type("BMP Loader: Header version must be at least Windows V1 (40 bytes)");
		}

		// Load image width
		std::memcpy(&width, in+currentOffset, 4);
		currentOffset+=4;

		// Load image height
		std::memcpy(&height, in+currentOffset, 4);
		currentOffset+=4;

		// Color planes check
		{
			uint16_t cPlanes;
			std::memcpy(&cPlanes, in+currentOffset, 2);
			currentOffset+=2;

			if (cPlanes != 1) {
				throw except::unsupported_image_type("BMP Loader: Color planes field must be 1");
			}
		}

		// Bpp check - only support BGR24
		{
			uint16_t bpp;
			std::memcpy(&bpp, in+currentOffset, 2);
			currentOffset+=2;

			if (bpp != 24) {
				throw except::unsupported_image_type("BMP Loader: Image must be BGR 24 bits-per-pixel");
			}
		}

		// Skip unnecessary fields
		currentOffset = pixelDataOffset;

		// Pixel load
		const uint8_t rowPadding = 4 - (width*3)%4;
		info.pixels.resize(width*height*3);

		if (height > 0) {	// If height is positive, convert to top-to-bottom
			size_t pixelLoadOffset = width*(height-1)*3;
			size_t pixelLoadOffset_Source = pixelDataOffset;
			for (size_t i = 0; i < static_cast<size_t>(std::abs(height)); i++) {
				std::memcpy(info.pixels.data()+pixelLoadOffset, in+pixelLoadOffset_Source, width*3);
				pixelLoadOffset -= width*3;
				pixelLoadOffset_Source += width*3 + rowPadding;
			}
		} else if (height < 0) {	// If height is negative, it's already in top-to-bottom
			size_t pixelLoadOffset = 0;
			size_t pixelLoadOffset_Source = pixelDataOffset;
			for (size_t i = 0; i < static_cast<size_t>(std::abs(height)); i++) {
				std::memcpy(info.pixels.data()+pixelLoadOffset, in+pixelLoadOffset_Source, width*3);
				pixelLoadOffset += width*3;
				pixelLoadOffset_Source += width*3 + rowPadding;
			}
		} else if (height == 0) {	// Height cannot be 0
			throw except::invalid_image_data("BMP Loader: Height cannot be 0");
		}

		// Convert to RGB
		for (size_t i = 0; i < static_cast<size_t>(std::abs(width)*std::abs(height)); i++) {
			std::swap(info.pixels[i*3], info.pixels[i*3+2]);
		}

		info.colorChannels = 3;
		info.width = std::abs(width);
		info.height = std::abs(height);

		return info;
	}

} // namespace wlib::image
