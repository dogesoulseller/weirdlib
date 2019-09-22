#include "../../include/cuda/weirdlib_cuda_image.hpp"
#include "./cuda_utils.cuh"
#include <cuda_runtime.h>
#include <curand.h>
#include <array>
#include <random>
#include <algorithm>
#include <thread>

#if WEIRDLIB_MULTITHREADING_MODE == WEIRDLIB_MTMODE_TBB
#include <tbb/tbb.h>
#endif

namespace wlib
{
namespace image
{
namespace cu
{

	ImageCUDA GenerateRandomImage(const int width, const int height, const ColorFormat format, const bool constantAlpha, const uint8_t alphaValue) {
		ImageCUDA imgout;
		imgout.height = height;
		imgout.width = width;
		imgout.format = format;

		return imgout;
	}

} // namespace cu
} // namespace image
} // namespace wlib
