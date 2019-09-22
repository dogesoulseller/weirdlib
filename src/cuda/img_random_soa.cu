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
	static __global__ void kernel_ScaleUniform(float* __restrict__ inout) {
		int tid = getGlobalIdx_1x1();
		inout[tid] *= 255.0f;
	}

	static __global__ void kernel_SetAllToFloat(float* __restrict__ output, const float input) {
		int tid = getGlobalIdx_1x1();
		output[tid] = input;
	}

	ImageSoACUDA GenerateRandomImageSoA(const int width, const int height, const ColorFormat format, const bool constantAlpha, const uint8_t alphaValue) {
		ImageSoACUDA outimg;
		outimg.width = width;
		outimg.height = height;
		outimg.format = format;
		switch (format)
		{
		case F_BGR:
		case F_RGB:
			outimg.channels.resize(3);
			break;
		case F_BGRA:
		case F_RGBA:
			outimg.channels.resize(4);
			break;
		case F_Grayscale:
			outimg.channels.resize(1);
			break;
		case F_GrayAlpha:
			outimg.channels.resize(2);
			break;
		default:
			break;
		}

		std::random_device rngdev;

		const int blockSize = getBlockSize(width*height);
		const int gridSize = width*height / blockSize;

		#ifdef WLIB_CUDA_USE_CURAND
		switch (format)
		{
		case F_BGR:
		case F_RGB: {
			for (size_t i = 0; i < 3; i++) {
				curandGenerator_t rng;
				curandCreateGenerator(&rng, curandRngType::CURAND_RNG_PSEUDO_PHILOX4_32_10);
				curandSetPseudoRandomGeneratorSeed(rng, rngdev());
				cudaMalloc(&outimg.channels[i], width*height*sizeof(float));

				cudaStream_t stream;
				cudaStreamCreate(&stream);

				curandSetStream(rng, stream);
				curandGenerateUniform(rng, outimg.channels[i], width*height);
				kernel_ScaleUniform<<<gridSize, blockSize, 0, stream>>>(outimg.channels[i]);

				curandDestroyGenerator(rng);
				cudaStreamDestroy(stream);
			}

			break;
		}
		case F_BGRA:
		case F_RGBA: {
			for (size_t i = 0; i < (constantAlpha ? 3 : 4); i++) {
				curandGenerator_t rng;
				curandCreateGenerator(&rng, curandRngType::CURAND_RNG_PSEUDO_PHILOX4_32_10);
				curandSetPseudoRandomGeneratorSeed(rng, rngdev());

				cudaMalloc(&outimg.channels[i], width*height*sizeof(float));

				cudaStream_t stream;
				cudaStreamCreate(&stream);

				curandSetStream(rng, stream);
				curandGenerateUniform(rng, outimg.channels[i], width*height);
				kernel_ScaleUniform<<<gridSize, blockSize, 0, stream>>>(outimg.channels[i]);

				curandDestroyGenerator(rng);
				cudaStreamDestroy(stream);
			}

			if (constantAlpha) {
				cudaMalloc(&outimg.channels[3], width*height*sizeof(float));
				const auto aConst = static_cast<float>(alphaValue);

				cudaStream_t stream;
				cudaStreamCreate(&stream);
				kernel_SetAllToFloat<<<gridSize, blockSize, 0, stream>>>(outimg.channels[3], aConst);
				cudaStreamDestroy(stream);
			}

			break;
		}
		case F_Grayscale: {
			const int blockSize = getBlockSize(width*height);
			const int gridSize = width*height / blockSize;

			curandGenerator_t rng;
			curandCreateGenerator(&rng, curandRngType::CURAND_RNG_PSEUDO_PHILOX4_32_10);
			curandSetPseudoRandomGeneratorSeed(rng, rngdev());

			cudaMalloc(&outimg.channels[0], width*height*sizeof(float));

			cudaStream_t stream;
			cudaStreamCreate(&stream);

			curandSetStream(rng, stream);
			curandGenerateUniform(rng, outimg.channels[0], width*height);
			kernel_ScaleUniform<<<gridSize, blockSize, 0, stream>>>(outimg.channels[0]);

			curandDestroyGenerator(rng);
			cudaStreamDestroy(stream);
			break;
		}
		case F_GrayAlpha: {
			const int blockSize = getBlockSize(width*height);
			const int gridSize = width*height / blockSize;

			for (size_t i = 0; i < (constantAlpha ? 1 : 2); i++) {
				curandGenerator_t rng;
				curandCreateGenerator(&rng, curandRngType::CURAND_RNG_PSEUDO_PHILOX4_32_10);
				curandSetPseudoRandomGeneratorSeed(rng, rngdev());

				cudaMalloc(&outimg.channels[i], width*height*sizeof(float));

				cudaStream_t stream;
				cudaStreamCreate(&stream);

				curandSetStream(rng, stream);
				curandGenerateUniform(rng, outimg.channels[i], width*height);
				kernel_ScaleUniform<<<gridSize, blockSize, 0, stream>>>(outimg.channels[i]);

				curandDestroyGenerator(rng);
				cudaStreamDestroy(stream);
			}

			if (constantAlpha) {
				cudaMalloc(&outimg.channels[3], width*height*sizeof(float));
				const auto aConst = static_cast<float>(alphaValue);

				cudaStream_t stream;
				cudaStreamCreate(&stream);
				kernel_SetAllToFloat<<<gridSize, blockSize, 0, stream>>>(outimg.channels[3], aConst);
				cudaStreamDestroy(stream);
			}
			break;
		}
		default:
			break;
		}

		#else

		switch (format)
		{
		case F_BGR:
		case F_RGB: {
			std::array<float*, 3> chans;
			cudaMallocHost(&chans[0], width*height*sizeof(float));
			cudaMallocHost(&chans[1], width*height*sizeof(float));
			cudaMallocHost(&chans[2], width*height*sizeof(float));

			#if WEIRDLIB_MULTITHREADING_MODE == WEIRDLIB_MTMODE_TBB
				tbb::parallel_for(tbb::blocked_range<size_t>(0, 3), [&](tbb::blocked_range<size_t>& i){
					for (size_t channel = i.begin(); channel != i.end(); channel++) {
						std::mt19937 rng(rngdev());
						std::uniform_real_distribution<float> dist(0.0f, 255.0f);
						tbb::parallel_for(tbb::blocked_range<size_t>(0, width*height, 1000), [&](tbb::blocked_range<size_t>& j) {
							for (size_t pixel = j.begin(); pixel != j.end(); pixel++) {
								chans[channel][pixel] = dist(rng);
							}
						});
					}
				});
			#else
				#if WEIRDLIB_MULTITHREADING_MODE == WEIRDLIB_MTMODE_OMP
				#pragma omp parallel for schedule(runtime)
				#endif
				for (size_t i = 0; i < 3; i++) {
					std::mt19937 rng(rngdev());
					std::uniform_real_distribution<float> dist(0.0f, 255.0f);
					#if WEIRDLIB_MULTITHREADING_MODE == WEIRDLIB_MTMODE_OMP
					#pragma omp parallel for schedule(runtime)
					#endif
					for (size_t j = 0; j < width * height; j++) {
						chans[i][j] = dist(rng);
					}
				}
			#endif

			cudaMalloc(&outimg.channels[0], width*height*sizeof(float));
			cudaMalloc(&outimg.channels[1], width*height*sizeof(float));
			cudaMalloc(&outimg.channels[2], width*height*sizeof(float));

			cudaMemcpy(outimg.channels[0], chans[0], width*height*sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(outimg.channels[1], chans[1], width*height*sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(outimg.channels[2], chans[2], width*height*sizeof(float), cudaMemcpyHostToDevice);

			cudaFreeHost(chans[0]);
			cudaFreeHost(chans[1]);
			cudaFreeHost(chans[2]);

			break;
		}
		case F_BGRA:
		case F_RGBA: {
			std::array<float*, 4> chans;
			cudaMallocHost(&chans[0], width*height*sizeof(float));
			cudaMallocHost(&chans[1], width*height*sizeof(float));
			cudaMallocHost(&chans[2], width*height*sizeof(float));
			if (!constantAlpha) {
				cudaMallocHost(&chans[3], width*height*sizeof(float));
			}

			#if WEIRDLIB_MULTITHREADING_MODE == WEIRDLIB_MTMODE_TBB
				tbb::parallel_for(tbb::blocked_range<size_t>(0, (constantAlpha ? 3 : 4)), [&](tbb::blocked_range<size_t>& i){
					for (size_t channel = i.begin(); channel != i.end(); channel++) {
						std::mt19937 rng(rngdev());
						std::uniform_real_distribution<float> dist(0.0f, 255.0f);
						tbb::parallel_for(tbb::blocked_range<size_t>(0, width*height, 1000), [&](tbb::blocked_range<size_t>& j) {
							for (size_t pixel = j.begin(); pixel != j.end(); pixel++) {
								chans[channel][pixel] = dist(rng);
							}
						});
					}
				});
			#else
				#if WEIRDLIB_MULTITHREADING_MODE == WEIRDLIB_MTMODE_OMP
				#pragma omp parallel for schedule(runtime)
				#endif
				for (size_t i = 0; i < (constantAlpha ? 3 : 4); i++) {
					std::mt19937 rng(rngdev());
					std::uniform_real_distribution<float> dist(0.0f, 255.0f);

					#if WEIRDLIB_MULTITHREADING_MODE == WEIRDLIB_MTMODE_OMP
					#pragma omp parallel for schedule(runtime)
					#endif
					for (size_t j = 0; j < width * height; j++) {
						chans[i][j] = dist(rng);
					}
				}
			#endif

			cudaMalloc(&outimg.channels[0], width*height*sizeof(float));
			cudaMalloc(&outimg.channels[1], width*height*sizeof(float));
			cudaMalloc(&outimg.channels[2], width*height*sizeof(float));
			cudaMalloc(&outimg.channels[3], width*height*sizeof(float));

			cudaMemcpy(outimg.channels[0], chans[0], width*height*sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(outimg.channels[1], chans[1], width*height*sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(outimg.channels[2], chans[2], width*height*sizeof(float), cudaMemcpyHostToDevice);

			if (!constantAlpha) {
				cudaMemcpy(outimg.channels[3], chans[3], width*height*sizeof(float), cudaMemcpyHostToDevice);
				cudaFreeHost(chans[0]);
				cudaFreeHost(chans[1]);
				cudaFreeHost(chans[2]);
				cudaFreeHost(chans[3]);
			} else {
				cudaStream_t stream;
				cudaStreamCreate(&stream);
				kernel_SetAllToFloat<<<gridSize, blockSize, 0, stream>>>(outimg.channels[3], static_cast<float>(alphaValue));
				cudaStreamDestroy(stream);

				cudaFreeHost(chans[0]);
				cudaFreeHost(chans[1]);
				cudaFreeHost(chans[2]);
			}
			break;
		}
		case F_Grayscale: {
			std::array<float*, 1> chans;
			cudaMallocHost(&chans[0], width*height*sizeof(float));

			std::mt19937 rng(rngdev());
			std::uniform_real_distribution<float> dist(0.0f, 255.0f);

			#if WEIRDLIB_MULTITHREADING_MODE == WEIRDLIB_MTMODE_TBB
				tbb::parallel_for(tbb::blocked_range<size_t>(0, width*height, 1000), [&](tbb::blocked_range<size_t>& i) {
					for (size_t pixel = i.begin(); pixel != i.end(); pixel++) {
						chans[0][pixel] = dist(rng);
					}
				});
			#else
				#if WEIRDLIB_MULTITHREADING_MODE == WEIRDLIB_MTMODE_OMP
				#pragma omp parallel for schedule(runtime)
				#endif
				for (size_t i = 0; i < width * height; i++) {
					chans[0][i] = dist(rng);
				}
			#endif

			cudaMalloc(&outimg.channels[0], width*height*sizeof(float));
			cudaMemcpy(outimg.channels[0], chans[0], width*height*sizeof(float), cudaMemcpyHostToDevice);

			cudaFreeHost(chans[0]);

			break;
		}
		case F_GrayAlpha: {
			std::array<float*, 2> chans;
			cudaMallocHost(&chans[0], width*height*sizeof(float));
			if (!constantAlpha) {
				cudaMallocHost(&chans[1], width*height*sizeof(float));
			}

			#if WEIRDLIB_MULTITHREADING_MODE == WEIRDLIB_MTMODE_TBB
				tbb::parallel_for(tbb::blocked_range<size_t>(0, (constantAlpha ? 1 : 2)), [&](tbb::blocked_range<size_t>& i){
					for (size_t channel = i.begin(); channel != i.end(); channel++) {
						std::mt19937 rng(rngdev());
						std::uniform_real_distribution<float> dist(0.0f, 255.0f);
						tbb::parallel_for(tbb::blocked_range<size_t>(0, width*height, 1000), [&](tbb::blocked_range<size_t>& j) {
							for (size_t pixel = j.begin(); pixel != j.end(); pixel++) {
								chans[channel][pixel] = dist(rng);
							}
						});
					}
				});
			#else
				#if WEIRDLIB_MULTITHREADING_MODE == WEIRDLIB_MTMODE_OMP
				#pragma omp parallel for schedule(runtime)
				#endif
				for (size_t i = 0; i < (constantAlpha ? 1 : 2); i++) {
					std::mt19937 rng(rngdev());
					std::uniform_real_distribution<float> dist(0.0f, 255.0f);

					#if WEIRDLIB_MULTITHREADING_MODE == WEIRDLIB_MTMODE_OMP
					#pragma omp parallel for schedule(runtime)
					#endif
					for (size_t j = 0; j < width * height; j++) {
						chans[i][j] = dist(rng);
					}
				}
			#endif

			cudaMalloc(&outimg.channels[0], width*height*sizeof(float));
			cudaMalloc(&outimg.channels[1], width*height*sizeof(float));

			cudaMemcpy(outimg.channels[0], chans[0], width*height*sizeof(float), cudaMemcpyHostToDevice);

			if (!constantAlpha) {
				cudaMemcpy(outimg.channels[1], chans[1], width*height*sizeof(float), cudaMemcpyHostToDevice);
				cudaFreeHost(chans[0]);
				cudaFreeHost(chans[1]);
			} else {
				cudaStream_t stream;
				cudaStreamCreate(&stream);
				kernel_SetAllToFloat<<<gridSize, blockSize, 0, stream>>>(outimg.channels[1], static_cast<float>(alphaValue));
				cudaStreamDestroy(stream);
				cudaFreeHost(chans[0]);
			}

			break;
		}
		}
		#endif

		return outimg;
	}

} // namespace cu

} // namespace image

} // namespace wlib
