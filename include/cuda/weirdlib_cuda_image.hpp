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
