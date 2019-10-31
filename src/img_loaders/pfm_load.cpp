#include "image_format_loaders.hpp"
#include "img_loader_exceptions.hpp"
#include "../common.hpp"
#include "../../include/weirdlib_bitops.hpp"
#include "../../include/weirdlib_math.hpp"

#include <fstream>
#include <memory>
#include <cstdint>
#include <algorithm>

namespace wlib::image
{
	ImageInfoPFM LoadPFM(const std::string& path) {
		std::ifstream in(path, std::ios::binary);
		return LoadPFM(in);
	}

	ImageInfoPFM LoadPFM(std::ifstream& in) {
		in.seekg(0, std::ios::end);
		size_t fileSize = in.tellg();
		in.seekg(0);

		auto fileBytes = std::make_unique<uint8_t[]>(fileSize);
		in.read(reinterpret_cast<char*>(fileBytes.get()), fileSize);

		return LoadPFM(fileBytes.get(), fileSize);
	}

	ImageInfoPFM LoadPFM(const uint8_t* in, size_t /*size*/) {
		ImageInfoPFM info;
		size_t currentOffset = 3;

		// Get width
		const auto& [IGNORESB, IGNORESB, tmpWidth] = GetNextNumber<size_t>(in, currentOffset);
		info.width = tmpWidth;

		// Get height
		const auto& [IGNORESB, IGNORESB, tmpHeight] = GetNextNumber<size_t>(in, currentOffset);
		info.height = tmpHeight;

		// Make space for data
		info.data.resize(info.width * info.height);

		// Get endianness
		std::array<char, 32> floatBuffer;

		auto digitLocation = std::find_first_of(in+currentOffset, in+currentOffset+128,
			digitsOrNeg.cbegin(), digitsOrNeg.cend());

		currentOffset = digitLocation - in + 1;

		auto whitespaceLocation = GetNextWhitespace(in, currentOffset);
		currentOffset = whitespaceLocation - in + 1;

		std::copy(digitLocation, whitespaceLocation, floatBuffer.begin());

		float endiannessValue = static_cast<float>(std::atof(floatBuffer.data()));

		info.isLittleEndian = math::float_eq(endiannessValue, -1.0f);

		// Get data
		auto inFloat = reinterpret_cast<const float*>(in+currentOffset);
		for (size_t i = 0; i < info.width * info.height; i++) {
			info.data[i] = *(inFloat+i);
		}

		return info;
	}
} // namespace wlib::image
