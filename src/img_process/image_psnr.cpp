#include "../../include/weirdlib_image.hpp"
#include "../../include/cpu_detection.hpp"
#include <cmath>
#include <array>
#include <numeric>

#if __has_include(<execution>)
	#include <execution>
#elif __has_include(<experimental/execution>)
	#include <experimental/execution>
	using std::execution::seq = std::experimental::execution::seq;
#endif

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

	// TODO: AVX SIMD
	static float getChannelMSE_float(float* lhs, float* rhs, size_t count) noexcept {
		#if X86_SIMD_LEVEL >= LV_SSE2

		size_t iters = count / 4;
		size_t itersRem = count % 4;

		__m128 accumulatorVec = _mm_setzero_ps();

		// TODO: Maybe unroll?
		for (size_t i = 0; i < iters; i++) {
			__m128 lVals = _mm_loadu_ps(lhs+4*i);
			__m128 rVals = _mm_loadu_ps(rhs+4*i);
			__m128 diff = _mm_sub_ps(lVals, rVals);
			__m128 mseRes = _mm_mul_ps(diff, diff);

			accumulatorVec = _mm_add_ps(accumulatorVec, mseRes);
		}

		float accumulatorScal = 0.0f;

		for (size_t i = 0; i < itersRem; i++) {
			accumulatorScal += std::pow(lhs[iters*4+i] - rhs[iters*4+i], 2);
		}

		std::array<float, 4> accumulatorVec_arr;
		_mm_storeu_ps(accumulatorVec_arr.data(), accumulatorVec);

		accumulatorScal += std::reduce(std::execution::seq, accumulatorVec_arr.begin(), accumulatorVec_arr.end());

		return accumulatorScal / count;

		#else
		float mseAccumulator = 0.0f;
		for (size_t i = 0; i < count; i++) {
			mseAccumulator += std::pow(lhs[i] - rhs[i], 2);
		}

		return mseAccumulator / count;
		#endif
	}

	static double getChannelMSE_double(float* lhs, float* rhs, size_t count) noexcept {
		#if X86_SIMD_LEVEL >= LV_SSE2

		size_t iters = count / 2;
		size_t itersRem = count % 2;

		__m128d accumulatorVec = _mm_setzero_pd();

		// TODO: Unroll to prevent unnecessary loads
		for (size_t i = 0; i < iters; i++) {
			__m128d lVals = _mm_cvtps_pd(_mm_loadu_ps(lhs+2*i));
			__m128d rVals = _mm_cvtps_pd(_mm_loadu_ps(rhs+2*i));
			__m128d diff = _mm_sub_pd(lVals, rVals);
			__m128d mseRes = _mm_mul_pd(diff, diff);

			accumulatorVec = _mm_add_pd(accumulatorVec, mseRes);
		}

		double accumulatorScal = 0.0;

		for (size_t i = 0; i < itersRem; i++) {
			accumulatorScal += std::pow(static_cast<double>(lhs[iters*2+i] - rhs[iters*2+i]), 2);
		}

		std::array<double, 2> accumulatorVec_arr;
		_mm_storeu_pd(accumulatorVec_arr.data(), accumulatorVec);

		accumulatorScal += std::reduce(std::execution::seq, accumulatorVec_arr.begin(), accumulatorVec_arr.end());

		return accumulatorScal / count;

		#else

		double mseAccumulator = 0.0;
		for (size_t i = 0; i < count; i++) {
			mseAccumulator += std::pow(static_cast<double>(lhs[i] - rhs[i]), 2);
		}

		return mseAccumulator / count;

		#endif
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
