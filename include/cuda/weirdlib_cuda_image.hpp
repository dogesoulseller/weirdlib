#pragma once

#include "../weirdlib_image.hpp"
#include <vector>

namespace wlib
{
namespace image
{
namespace cu
{
	/// Collection of data returned by @{link GetHistogram}
	/// Only appropriate vectors aren't empty
	struct HistogramData
	{
		std::vector<uint64_t> Red;
		std::vector<uint64_t> Green;
		std::vector<uint64_t> Blue;
		std::vector<uint64_t> Gray;
		std::vector<uint64_t> Alpha;
	};

	/// Generate a random SoA image \n
	/// Uses cuRAND when specified, otherwise generates on-CPU and moves to GPU
	/// @param width image width
	/// @param width image height
	/// @param format image format
	/// @param constantAlpha whether alpha should be set to the value set in alphaValue
	/// @return image with random data
	ImageSoACUDA GenerateRandomImageSoA(int width, int height, ColorFormat format, bool constantAlpha = false, uint8_t alphaValue = 255);

	/// Generate a random AoS image \n
	/// Uses cuRAND when specified, otherwise generates on-CPU and moves to GPU
	/// @param width image width
	/// @param width image height
	/// @param format image format
	/// @param constantAlpha whether alpha should be set to the value set in alphaValue
	/// @return image with random data
	ImageCUDA GenerateRandomImage(int width, int height, ColorFormat, bool constantAlpha = false, uint8_t alphaValue = 255);

	/// Convert image (in GPU memory) into a grayscale representation using various methods described in @{link GrayscaleMethod}
	/// @param inImg image to be modified
	/// @param preserveAlpha whether alpha should be kept or discarded
	/// @param method grayscale calculation method
	/// @return reference to modified input image
	ImageSoACUDA& ConvertToGrayscale(ImageSoACUDA& inImg, bool preserveAlpha = false, GrayscaleMethod method = GrayscaleMethod::Luminosity);

	/// Convert all color values (in GPU memory) to their image negative version <br>
	/// Effectively does max_val - current_val
	/// @param in input image
	/// @param withAlpha whether to preserve alpha channel
	void NegateValues(ImageSoACUDA& in, bool withAlpha = false);

	/// Get histogram data for each channel
	/// @param in input image
	/// @return @{link HistogramData} with appropriate channel data filled in
	HistogramData GetHistogram(ImageSoACUDA& in);

} // namespace cu

} // namespace image

} // namespace wlib
