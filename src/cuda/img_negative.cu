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
	__global__ void kernel_NegateChannelValues(float* __restrict__ inout) {
		const int threadID = getGlobalIdx_1x1();
		inout[threadID] = 255.0f - inout[threadID];
	}

	void NegateValues(ImageSoACUDA& in, const bool withAlpha) {
		const size_t blockSize = getBlockSize(in.width * in.height);
		const size_t gridSize = in.width * in.height / blockSize;

		switch (in.format)
		{
		case F_BGR:
		case F_Grayscale:
		case F_RGB:
		{
			for (size_t i = 0; i < in.channels.size(); i++) {
				cudaStream_t stream;
				cudaStreamCreate(&stream);
				kernel_NegateChannelValues<<<gridSize, blockSize, 0, stream>>>(in.channels[i]);
				cudaStreamDestroy(stream);
			}
		}
			break;
		case F_GrayAlpha:
		case F_BGRA:
		case F_RGBA: {
			if (!withAlpha) {
				for (size_t i = 0; i < in.channels.size()-1; i++) {
					cudaStream_t stream;
					cudaStreamCreate(&stream);
					kernel_NegateChannelValues<<<gridSize, blockSize, 0, stream>>>(in.channels[i]);
					cudaStreamDestroy(stream);
				}
			} else {
				for (size_t i = 0; i < in.channels.size(); i++) {
					cudaStream_t stream;
					cudaStreamCreate(&stream);
					kernel_NegateChannelValues<<<gridSize, blockSize, 0, stream>>>(in.channels[i]);
					cudaStreamDestroy(stream);
				}
			}
		}
			break;
		}

	}
} // namespace cu
} // namespace image
} // namespace wlib
#endif
