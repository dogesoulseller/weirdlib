#include "../../include/cuda/weirdlib_cuda_image.hpp"
#include "./cuda_utils.cuh"
#ifdef WLIB_ENABLE_CUDA
#include <cuda_runtime.h>

namespace wlib
{
namespace image
{
namespace cu
{
	template<GrayscaleMethod method = GrayscaleMethod::Luminosity>
	__global__ void __launch_bounds__(512) kernel_ConvertGrayscaleRGB(const float* __restrict__ r, const float* __restrict__ g, const float* __restrict__ b, float* __restrict__ outGray) {
		const int pixelID = getGlobalIdx_1x1();
		float3 comps {r[pixelID], g[pixelID], b[pixelID]};
		if (method == GrayscaleMethod::Luminosity) {
			outGray[pixelID] = fminf(fmaf(comps.x, 0.2126f, fmaf(comps.y, 0.7152f, comps.z * 0.0722f)), 255.0f);
		} else if (method == GrayscaleMethod::Lightness) {
			outGray[pixelID] = ((fmaxf(fmaxf(comps.x, comps.y), comps.z)) + (fminf(fminf(comps.x, comps.y), comps.z))) * 0.5f;
		} else if (method == GrayscaleMethod::Average) {
			outGray[pixelID] = (comps.x + comps.y + comps.z) * (1.0f/3.0f);
		} else if (method == GrayscaleMethod::LuminosityBT601) {
			outGray[pixelID] = fminf(fmaf(comps.x, 0.299f, fmaf(comps.y, 0.587f, comps.z * 0.114f)), 255.0f);
		}
	}

	ImageSoACUDA& ConvertToGrayscale(ImageSoACUDA& inImg, const bool preserveAlpha, const GrayscaleMethod method) {
		if (inImg.format == F_GrayAlpha || inImg.format == F_Grayscale) {
			return inImg;
		}

		float* outGray;
		const size_t channelSize = inImg.width * inImg.height * sizeof(float);
		cudaMalloc(&outGray, channelSize);

		float* red;
		float* green;
		float* blue;

		if (inImg.format == F_RGBA || inImg.format == F_RGB) {
			red = inImg.channels[0];
			green = inImg.channels[1];
			blue = inImg.channels[2];
		} else {
			red = inImg.channels[2];
			green = inImg.channels[1];
			blue = inImg.channels[0];
		}

		const size_t blockSize = getBlockSize(inImg.width * inImg.height);
		const size_t gridSize = inImg.width * inImg.height / blockSize;

		cudaStream_t stream;
		cudaStreamCreate(&stream);

		switch (method)
		{
		case GrayscaleMethod::Luminosity:
			kernel_ConvertGrayscaleRGB<GrayscaleMethod::Luminosity><<<gridSize, blockSize, 0, stream>>>(red, green, blue, outGray);
			break;
		case GrayscaleMethod::Lightness:
			kernel_ConvertGrayscaleRGB<GrayscaleMethod::Lightness><<<gridSize, blockSize, 0, stream>>>(red, green, blue, outGray);
			break;
		case GrayscaleMethod::Average:
			kernel_ConvertGrayscaleRGB<GrayscaleMethod::Average><<<gridSize, blockSize, 0, stream>>>(red, green, blue, outGray);
			break;
		case GrayscaleMethod::LuminosityBT601:
			kernel_ConvertGrayscaleRGB<GrayscaleMethod::Luminosity><<<gridSize, blockSize, 0, stream>>>(red, green, blue, outGray);
			break;
		}
		cudaStreamDestroy(stream);

		if (!preserveAlpha || inImg.format == F_BGR || inImg.format == F_RGB) {
			inImg.format = F_Grayscale;
			for (auto& c : inImg.channels) {
				cudaFree(c);
			}

			inImg.channels.resize(1);
			inImg.channels.shrink_to_fit();
			inImg.channels[0] = outGray;
		} else {
			inImg.format = F_GrayAlpha;
			for (size_t i = 0; i < 3; i++) {
				cudaFree(inImg.channels[i]);
			}

			auto imgAlpha = inImg.channels[3];

			inImg.channels.resize(2);
			inImg.channels.shrink_to_fit();
			inImg.channels[0] = outGray;
			inImg.channels[1] = imgAlpha;
		}

		return inImg;
	}

} // namespace cu
} // namespace image
} // namespace wlib
#endif
