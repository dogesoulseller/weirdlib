#include "../../include/cuda/weirdlib_cuda_image.hpp"
#ifdef WLIB_ENABLE_CUDA
#include <cuda_runtime.h>
#include "cuda_utils.cuh"
#include <array>

namespace wlib
{
namespace image
{
namespace cu
{
	__global__ void kernel_ChannelHistogram(float* __restrict__ input, uint64_t* __restrict__ channelOutput) {
		const int threadId = getGlobalIdx_1x1();

		atomicAdd((unsigned long long int*)&channelOutput[(int)input[threadId]], 1ULL);
	}

	HistogramData GetHistogram(ImageSoACUDA& in) {
		HistogramData d;
		const uint64_t blockSize = getBlockSize(in.height * in.width);
		const uint64_t gridSize = in.height * in.width / blockSize;

		switch (in.format)
		{
		case F_Grayscale: {
			uint64_t* gray_dev;

			cudaStream_t stream;
			cudaStreamCreate(&stream);

			cudaMalloc(&gray_dev, 256*sizeof(uint64_t));
			cudaMemsetAsync(gray_dev, 0, 256*sizeof(uint64_t), stream);

			kernel_ChannelHistogram<<<gridSize, blockSize, 0, stream>>>(in.channels[0], gray_dev);
			d.Gray.resize(256);

			cudaMemcpyAsync(d.Gray.data(), gray_dev, 256*sizeof(uint64_t), cudaMemcpyDeviceToHost, stream);
			cudaStreamDestroy(stream);
			cudaFree(gray_dev);
			break;
		}

		case F_GrayAlpha: {
			std::array<uint64_t*, 2> chans_dev;

			for (auto& c: chans_dev) {
				cudaMalloc(&c, 256*sizeof(uint64_t));
			}

			// Gray
			cudaStream_t streamG;
			cudaStreamCreate(&streamG);

			cudaMemsetAsync(in.channels[0], 0, 256*sizeof(uint64_t), streamG);

			kernel_ChannelHistogram<<<gridSize, blockSize, 0, streamG>>>(in.channels[0], chans_dev[0]);
			d.Gray.resize(256);

			cudaMemcpyAsync(d.Gray.data(), chans_dev[0], 256*sizeof(uint64_t), cudaMemcpyDeviceToHost, streamG);
			cudaStreamDestroy(streamG);
			cudaFree(chans_dev[0]);

			// Alpha
			cudaStream_t streamA;
			cudaStreamCreate(&streamA);

			cudaMemsetAsync(in.channels[1], 0, 256*sizeof(uint64_t), streamA);

			kernel_ChannelHistogram<<<gridSize, blockSize, 0, streamA>>>(in.channels[1], chans_dev[1]);
			d.Alpha.resize(256);

			cudaMemcpyAsync(d.Alpha.data(), chans_dev[1], 256*sizeof(uint64_t), cudaMemcpyDeviceToHost, streamA);
			cudaStreamDestroy(streamA);
			cudaFree(chans_dev[1]);
			break;
		}

		case F_RGB: {
			std::array<uint64_t*, 3> chans_dev;

			for (auto& c: chans_dev) {
				cudaMalloc(&c, 256*sizeof(uint64_t));
			}

			// Red
			cudaStream_t streamR;
			cudaStreamCreate(&streamR);

			cudaMemsetAsync(in.channels[0], 0, 256*sizeof(uint64_t), streamR);

			kernel_ChannelHistogram<<<gridSize, blockSize, 0, streamR>>>(in.channels[0], chans_dev[0]);
			d.Red.resize(256);

			cudaMemcpyAsync(d.Red.data(), chans_dev[0], 256*sizeof(uint64_t), cudaMemcpyDeviceToHost, streamR);
			cudaStreamDestroy(streamR);
			cudaFree(chans_dev[0]);

			// Green
			cudaStream_t streamG;
			cudaStreamCreate(&streamG);

			cudaMemsetAsync(in.channels[1], 0, 256*sizeof(uint64_t), streamG);

			kernel_ChannelHistogram<<<gridSize, blockSize, 0, streamG>>>(in.channels[1], chans_dev[1]);
			d.Green.resize(256);

			cudaMemcpyAsync(d.Green.data(), chans_dev[1], 256*sizeof(uint64_t), cudaMemcpyDeviceToHost, streamG);
			cudaStreamDestroy(streamG);
			cudaFree(chans_dev[1]);

			// Blue
			cudaStream_t streamB;
			cudaStreamCreate(&streamB);

			cudaMemsetAsync(in.channels[2], 0, 256*sizeof(uint64_t), streamB);

			kernel_ChannelHistogram<<<gridSize, blockSize, 0, streamB>>>(in.channels[2], chans_dev[2]);
			d.Blue.resize(256);

			cudaMemcpyAsync(d.Blue.data(), chans_dev[2], 256*sizeof(uint64_t), cudaMemcpyDeviceToHost, streamB);
			cudaStreamDestroy(streamB);
			cudaFree(chans_dev[2]);
			break;
		}

		case F_RGBA: {
			std::array<uint64_t*, 4> chans_dev;

			for (auto& c: chans_dev) {
				cudaMalloc(&c, 256*sizeof(uint64_t));
			}

			// Red
			cudaStream_t streamR;
			cudaStreamCreate(&streamR);

			cudaMemsetAsync(in.channels[0], 0, 256*sizeof(uint64_t), streamR);

			kernel_ChannelHistogram<<<gridSize, blockSize, 0, streamR>>>(in.channels[0], chans_dev[0]);
			d.Red.resize(256);

			cudaMemcpyAsync(d.Red.data(), chans_dev[0], 256*sizeof(uint64_t), cudaMemcpyDeviceToHost, streamR);
			cudaStreamDestroy(streamR);
			cudaFree(chans_dev[0]);

			// Green
			cudaStream_t streamG;
			cudaStreamCreate(&streamG);

			cudaMemsetAsync(in.channels[1], 0, 256*sizeof(uint64_t), streamG);

			kernel_ChannelHistogram<<<gridSize, blockSize, 0, streamG>>>(in.channels[1], chans_dev[1]);
			d.Green.resize(256);

			cudaMemcpyAsync(d.Green.data(), chans_dev[1], 256*sizeof(uint64_t), cudaMemcpyDeviceToHost, streamG);
			cudaStreamDestroy(streamG);
			cudaFree(chans_dev[1]);

			// Blue
			cudaStream_t streamB;
			cudaStreamCreate(&streamB);

			cudaMemsetAsync(in.channels[2], 0, 256*sizeof(uint64_t), streamB);

			kernel_ChannelHistogram<<<gridSize, blockSize, 0, streamB>>>(in.channels[2], chans_dev[2]);
			d.Blue.resize(256);

			cudaMemcpyAsync(d.Blue.data(), chans_dev[2], 256*sizeof(uint64_t), cudaMemcpyDeviceToHost, streamB);
			cudaStreamDestroy(streamB);
			cudaFree(chans_dev[2]);

			// Alpha
			cudaStream_t streamA;
			cudaStreamCreate(&streamA);

			cudaMemsetAsync(in.channels[3], 0, 256*sizeof(uint64_t), streamA);

			kernel_ChannelHistogram<<<gridSize, blockSize, 0, streamA>>>(in.channels[3], chans_dev[3]);
			d.Alpha.resize(256);

			cudaMemcpyAsync(d.Alpha.data(), chans_dev[3], 256*sizeof(uint64_t), cudaMemcpyDeviceToHost, streamA);
			cudaStreamDestroy(streamA);
			cudaFree(chans_dev[3]);
			break;
		}

		case F_BGR: {
			std::array<uint64_t*, 3> chans_dev;

			for (auto& c: chans_dev) {
				cudaMalloc(&c, 256*sizeof(uint64_t));
			}

			// Blue
			cudaStream_t streamB;
			cudaStreamCreate(&streamB);

			cudaMemsetAsync(in.channels[0], 0, 256*sizeof(uint64_t), streamB);

			kernel_ChannelHistogram<<<gridSize, blockSize, 0, streamB>>>(in.channels[0], chans_dev[0]);
			d.Blue.resize(256);

			cudaMemcpyAsync(d.Blue.data(), chans_dev[0], 256*sizeof(uint64_t), cudaMemcpyDeviceToHost, streamB);
			cudaStreamDestroy(streamB);
			cudaFree(chans_dev[0]);

			// Green
			cudaStream_t streamG;
			cudaStreamCreate(&streamG);

			cudaMemsetAsync(in.channels[1], 0, 256*sizeof(uint64_t), streamG);

			kernel_ChannelHistogram<<<gridSize, blockSize, 0, streamG>>>(in.channels[1], chans_dev[1]);
			d.Green.resize(256);

			cudaMemcpyAsync(d.Green.data(), chans_dev[1], 256*sizeof(uint64_t), cudaMemcpyDeviceToHost, streamG);
			cudaStreamDestroy(streamG);
			cudaFree(chans_dev[1]);


			// Red
			cudaStream_t streamR;
			cudaStreamCreate(&streamR);

			cudaMemsetAsync(in.channels[2], 0, 256*sizeof(uint64_t), streamR);

			kernel_ChannelHistogram<<<gridSize, blockSize, 0, streamR>>>(in.channels[2], chans_dev[2]);
			d.Red.resize(256);

			cudaMemcpyAsync(d.Red.data(), chans_dev[2], 256*sizeof(uint64_t), cudaMemcpyDeviceToHost, streamR);
			cudaStreamDestroy(streamR);
			cudaFree(chans_dev[2]);
			break;
		}

		case F_BGRA: {
			std::array<uint64_t*, 4> chans_dev;

			for (auto& c: chans_dev) {
				cudaMalloc(&c, 256*sizeof(uint64_t));
			}

			// Blue
			cudaStream_t streamB;
			cudaStreamCreate(&streamB);

			cudaMemsetAsync(in.channels[0], 0, 256*sizeof(uint64_t), streamB);

			kernel_ChannelHistogram<<<gridSize, blockSize, 0, streamB>>>(in.channels[0], chans_dev[0]);
			d.Blue.resize(256);

			cudaMemcpyAsync(d.Blue.data(), chans_dev[0], 256*sizeof(uint64_t), cudaMemcpyDeviceToHost, streamB);
			cudaStreamDestroy(streamB);
			cudaFree(chans_dev[0]);

			// Green
			cudaStream_t streamG;
			cudaStreamCreate(&streamG);

			cudaMemsetAsync(in.channels[1], 0, 256*sizeof(uint64_t), streamG);

			kernel_ChannelHistogram<<<gridSize, blockSize, 0, streamG>>>(in.channels[1], chans_dev[1]);
			d.Green.resize(256);

			cudaMemcpyAsync(d.Green.data(), chans_dev[1], 256*sizeof(uint64_t), cudaMemcpyDeviceToHost, streamG);
			cudaStreamDestroy(streamG);
			cudaFree(chans_dev[1]);

			// Red
			cudaStream_t streamR;
			cudaStreamCreate(&streamR);

			cudaMemsetAsync(in.channels[2], 0, 256*sizeof(uint64_t), streamR);

			kernel_ChannelHistogram<<<gridSize, blockSize, 0, streamR>>>(in.channels[2], chans_dev[2]);
			d.Red.resize(256);

			cudaMemcpyAsync(d.Red.data(), chans_dev[2], 256*sizeof(uint64_t), cudaMemcpyDeviceToHost, streamR);
			cudaStreamDestroy(streamR);
			cudaFree(chans_dev[2]);

			// Alpha
			cudaStream_t streamA;
			cudaStreamCreate(&streamA);

			cudaMemsetAsync(in.channels[3], 0, 256*sizeof(uint64_t), streamA);

			kernel_ChannelHistogram<<<gridSize, blockSize, 0, streamA>>>(in.channels[3], chans_dev[3]);
			d.Alpha.resize(256);

			cudaMemcpyAsync(d.Alpha.data(), chans_dev[3], 256*sizeof(uint64_t), cudaMemcpyDeviceToHost, streamA);
			cudaStreamDestroy(streamA);
			cudaFree(chans_dev[3]);

			break;
		}

		}

		return d;
	}

} // namespace cu
} // namespace image
} // namespace wlib
#endif
