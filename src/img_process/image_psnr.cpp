#ifdef WEIRDLIB_ENABLE_IMAGE_OPERATIONS
#include "../../include/weirdlib_image.hpp"
#include "../../include/cpu_detection.hpp"
#include "../../include/weirdlib_simdhelper.hpp"
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

	inline static bool formatsAreCompatible(ColorFormat lhs, ColorFormat rhs) noexcept {
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

	inline static float getChannelMSE_float(const float* lhs, const float* rhs, size_t count) noexcept {
		#if X86_SIMD_LEVEL >= LV_AVX
			size_t iters = count / 8;
			size_t itersRem = count % 8;

			__m256 accumulatorVec = _mm256_setzero_ps();

			for (size_t i = 0; i < iters; i++) {
				__m256 lVals = _mm256_loadu_ps(lhs+8*i);
				__m256 rVals = _mm256_loadu_ps(rhs+8*i);
				__m256 diff = _mm256_sub_ps(lVals, rVals);
				__m256 mseRes = _mm256_mul_ps(diff, diff);

				accumulatorVec = _mm256_add_ps(accumulatorVec, mseRes);
			}

			float accumulatorScal = 0.0f;

			for (size_t i = 0; i < itersRem; i++) {
				accumulatorScal += std::pow(lhs[iters*8+i] - rhs[iters*8+i], 2);
			}

			std::array<float, 8> accumulatorVec_arr;
			_mm256_storeu_ps(accumulatorVec_arr.data(), accumulatorVec);

			accumulatorScal += std::reduce(std::execution::seq, accumulatorVec_arr.begin(), accumulatorVec_arr.end());

			return accumulatorScal / count;
		#elif X86_SIMD_LEVEL >= LV_SSE2
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

	inline static double getChannelMSE_double(const float* lhs, const float* rhs, size_t count) noexcept {
		#if X86_SIMD_LEVEL >= LV_AVX
			size_t iters = count / 4;
			size_t itersRem = count % 4;

			__m256d accumulatorVec = _mm256_setzero_pd();

			for (size_t i = 0; i < iters; i++) {
				__m256d lVals = _mm256_cvtps_pd(_mm_loadu_ps(lhs+4*i));
				__m256d rVals = _mm256_cvtps_pd(_mm_loadu_ps(rhs+4*i));

				__m256d diff = _mm256_sub_pd(lVals, rVals);
				__m256d mseRes = _mm256_mul_pd(diff, diff);

				accumulatorVec = _mm256_add_pd(accumulatorVec, mseRes);
			}

			double accumulatorScal = 0.0;

			for (size_t i = 0; i < itersRem; i++) {
				accumulatorScal += std::pow(static_cast<double>(lhs[iters*4+i] - rhs[iters*4+i]), 2);
			}

			std::array<double, 4> accumulatorVec_arr;
			_mm256_storeu_pd(accumulatorVec_arr.data(), accumulatorVec);

			accumulatorScal += std::reduce(std::execution::seq, accumulatorVec_arr.begin(), accumulatorVec_arr.end());

			return accumulatorScal / count;
		#elif X86_SIMD_LEVEL >= LV_SSE2
			size_t iters = count / 4;
			size_t itersRem = count % 4;

			__m128d accumulatorVec = _mm_setzero_pd();

			for (size_t i = 0; i < iters; i++) {
				// Loads 4
				__m128 lVals_flt = _mm_loadu_ps(lhs+4*i);
				__m128 rVals_flt = _mm_loadu_ps(rhs+4*i);

				// Converts first two values
				__m128d lVals0 = _mm_cvtps_pd(lVals_flt);
				__m128d rVals0 = _mm_cvtps_pd(rVals_flt);

				// Reverse lhs and rhs to get to high half
				lVals_flt = simd::reverse(lVals_flt);
				rVals_flt = simd::reverse(rVals_flt);

				// Convert next two values that are now in the low part
				__m128d lVals1 = _mm_cvtps_pd(lVals_flt);
				__m128d rVals1 = _mm_cvtps_pd(rVals_flt);

				__m128d diff0 = _mm_sub_pd(lVals0, rVals0);
				__m128d diff1 = _mm_sub_pd(lVals1, rVals1);

				__m128d mseRes0 = _mm_mul_pd(diff0, diff0);
				__m128d mseRes1 = _mm_mul_pd(diff1, diff1);

				accumulatorVec = _mm_add_pd(accumulatorVec, _mm_add_pd(mseRes0, mseRes1));
			}

			double accumulatorScal = 0.0;

			for (size_t i = 0; i < itersRem; i++) {
				accumulatorScal += std::pow(static_cast<double>(lhs[iters*4+i] - rhs[iters*4+i]), 2);
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
	static PSNRData<FloatT> getPsnr(const ImageSoA& image0, const ImageSoA& image1) {
		PSNRData<FloatT> outData;
		if (!formatsAreCompatible(image0.GetFormat(), image1.GetFormat())) {
			throw image_channel_error("Image formats are not compatible");
		}

		if (image0.GetWidth() != image1.GetWidth() || image0.GetHeight() != image1.GetHeight()) {
			throw image_dimensions_error("Image dimensions are different");
		}

		switch (image0.GetFormat())
		{
		  case F_RGB:
		  case F_BGR:
		  case F_RGBA:
		  case F_BGRA: {
			outData.MSEPerChannel.resize(3);
			outData.PSNRPerChannel.resize(3);

			for (size_t i = 0; i < 3; i++) {
				if constexpr (std::is_same_v<FloatT, float>) {
					outData.MSEPerChannel[i] = getChannelMSE_float(image0.channels[i], image1.channels[i], image0.GetWidth() * image0.GetHeight());
				} else {
					outData.MSEPerChannel[i] = getChannelMSE_double(image0.channels[i], image1.channels[i], image0.GetWidth() * image0.GetHeight());
				}

				outData.PSNRPerChannel[i] = 10 * std::log10(RSqr<FloatT> / outData.MSEPerChannel[i]);
			}

			break;
		  }
		  case F_GrayAlpha:
		  case F_Grayscale: {
			outData.MSEPerChannel.resize(1);
			outData.PSNRPerChannel.resize(1);

			if constexpr (std::is_same_v<FloatT, float>) {
				outData.MSEPerChannel[0] = getChannelMSE_float(image0.channels[0], image1.channels[0], image0.GetWidth() * image0.GetHeight());
			} else {
				outData.MSEPerChannel[0] = getChannelMSE_double(image0.channels[0], image1.channels[0], image0.GetWidth() * image0.GetHeight());
			}

			outData.PSNRPerChannel[0] = 10 * std::log10(RSqr<FloatT> / outData.MSEPerChannel[0]);

			break;
		  }
		  case F_Default:
			throw image_channel_error("Image had invalid format");
			break;
		}

		return outData;
	}

namespace detail
{
	PSNRData<float> psnrFloat(const ImageSoA& image0, const ImageSoA& image1) {
		return getPsnr<float>(image0, image1);
	}

	PSNRData<double> psnrDouble(const ImageSoA& image0, const ImageSoA& image1) {
		return getPsnr<double>(image0, image1);
	}
} // namespace detail

} // namespace wlib::image
#endif
