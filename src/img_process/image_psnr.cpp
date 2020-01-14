#include "../../include/weirdlib_image.hpp"
#include <cmath>

namespace wlib::image
{
	template<typename FloatT>
	constexpr FloatT RSqr = static_cast<FloatT>(255*255);

	static bool formatsAreCompatible(ColorFormat lhs, ColorFormat rhs) noexcept {
		bool compatible =
			(lhs == rhs) ||
			(lhs == F_RGB && rhs == F_RGBA) ||
			(lhs == F_RGBA && rhs == F_RGB) ||
			(lhs == F_BGR && rhs == F_BGRA) ||
			(lhs == F_BGRA && rhs == F_BGR) ||
			(lhs == F_Grayscale && rhs == F_GrayAlpha) ||
			(lhs == F_GrayAlpha && rhs == F_Grayscale);

		return compatible;
	}

	// TODO: SIMD
	static float getChannelMSE_float(float* lhs, float* rhs, size_t count) noexcept {
		float mseAccumulator = 0.0f;
		for (size_t i = 0; i < count; i++) {
			mseAccumulator += std::pow(lhs[i] - rhs[i], 2);
		}

		return mseAccumulator / count;
	}

	static double getChannelMSE_double(float* lhs, float* rhs, size_t count) noexcept {
		double mseAccumulator = 0.0;
		for (size_t i = 0; i < count; i++) {
			mseAccumulator += std::pow(static_cast<double>(lhs[i] - rhs[i]), 2);
		}

		return mseAccumulator / count;
	}

	template<typename FloatT>
	static PSNRData<FloatT> getPsnr(ImageSoA& image0, ImageSoA& image1) {
		PSNRData<FloatT> outData;
		if (!formatsAreCompatible(image0.format, image1.format)) {
			throw image_channel_error("Image formats are not compatible");
		}

		if (image0.width != image1.width || image0.height != image1.height) {
			throw image_dimensions_error("Image dimensions are different");
		}

		switch (image0.format)
		{
			case F_RGB:
			case F_BGR:
			case F_RGBA:
			case F_BGRA:
			{
				outData.MSEPerChannel.resize(3);
				outData.PSNRPerChannel.resize(3);

				for (size_t i = 0; i < 3; i++) {
					if constexpr (std::is_same_v<FloatT, float>) {
						outData.MSEPerChannel[i] = getChannelMSE_float(image0.channels[i], image1.channels[i], image0.width * image0.height);
					} else {
						outData.MSEPerChannel[i] = getChannelMSE_double(image0.channels[i], image1.channels[i], image0.width * image0.height);
					}

					outData.PSNRPerChannel[i] = 10 * std::log10(RSqr<FloatT> / outData.MSEPerChannel[i]);
				}

			}
				break;
			case F_GrayAlpha:
			case F_Grayscale:
			{
				outData.MSEPerChannel.resize(1);
				outData.PSNRPerChannel.resize(1);

				if constexpr (std::is_same_v<FloatT, float>) {
					outData.MSEPerChannel[0] = getChannelMSE_float(image0.channels[0], image1.channels[0], image0.width * image0.height);
				} else {
					outData.MSEPerChannel[0] = getChannelMSE_double(image0.channels[0], image1.channels[0], image0.width * image0.height);
				}

				outData.PSNRPerChannel[0] = 10 * std::log10(RSqr<FloatT> / outData.MSEPerChannel[0]);
			}
			case F_Default:
				throw image_channel_error("Image had invalid format");
				break;
		}

		return outData;
	}

namespace detail
{
	PSNRData<float> psnrFloat(ImageSoA& image0, ImageSoA& image1) {
		return getPsnr<float>(image0, image1);
	}

	PSNRData<double> psnrDouble(ImageSoA& image0, ImageSoA& image1) {
		return getPsnr<double>(image0, image1);
	}
} // namespace detail

} // namespace wlib::image
