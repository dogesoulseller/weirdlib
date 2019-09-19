#pragma once
#ifdef WLIB_ENABLE_CUDA
#include <cuda_runtime.h>

/// 1D grid of 1D blocks
inline __device__ int getGlobalIdx_1x1() {
	return blockIdx.x * blockDim.x + threadIdx.x;
}

/// 1D grid of 2D blocks
inline __device__ int getGlobalIdx_1x2() {
	return blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
}

/// 1D grid of 3D blocks
inline __device__ int getGlobalIdx_1x3() {
	return blockIdx.x * blockDim.x * blockDim.y
	* blockDim.z + threadIdx.z * blockDim.y
	* blockDim.x + threadIdx.y * blockDim.x
	+ threadIdx.x;
}

/// 2D grid of 1D blocks
inline __device__ int getGlobalIdx_2x1() {
	return (blockIdx.y * gridDim.x + blockIdx.x)
	* blockDim.x + threadIdx.x;
}

/// 2D grid of 2D blocks
inline __device__ int getGlobalIdx_2x2() {
	return (blockIdx.x + blockIdx.y * gridDim.x)
	* (blockDim.x * blockDim.y)
	+ (threadIdx.y * blockDim.x)
	+ threadIdx.x;
}

/// 2D grid of 3D blocks
inline __device__ int getGlobalIdx_2x3() {
	return (blockIdx.x + blockIdx.y * gridDim.x)
	* (blockDim.x * blockDim.y * blockDim.z)
	+ (threadIdx.z * (blockDim.x * blockDim.y))
	+ (threadIdx.y * blockDim.x) + threadIdx.x;
}

/// 3D grid of 1D blocks
inline __device__ int getGlobalIdx_3x1() {
	return (blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z)
	* blockDim.x + threadIdx.x;
}

/// 3D grid of 2D blocks
inline __device__ int getGlobalIdx_3x2() {
	return (blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z)
	* (blockDim.x * blockDim.y)
	+ (threadIdx.y * blockDim.x)
	+ threadIdx.x;
}

/// 3D grid of 3D blocks
inline __device__ int getGlobalIdx_3x3() {
	return (blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z)
	* (blockDim.x * blockDim.y * blockDim.z)
	+ (threadIdx.z * (blockDim.x * blockDim.y))
	+ (threadIdx.y * blockDim.x) + threadIdx.x;
}

inline int getImageSplit(const int imgHeight) {
	if (imgHeight % 6 == 0) {
		return 6;
	} else if (imgHeight % 5 == 0) {
		return 5;
	} else if (imgHeight % 4 == 0) {
		return 4;
	} else if (imgHeight % 3 == 0) {
		return 3;
	} else if (imgHeight % 2 == 0) {
		return 2;
	} else {
		return 1;
	}
}

inline size_t getBlockSize(const size_t imgHeight) {
	if (imgHeight % 512 == 0) {
		return 512;
	} else if (imgHeight % 256 == 0) {
		return 256;
	} else if (imgHeight % 128 == 0) {
		return 128;
	} else if (imgHeight % 64 == 0) {
		return 64;
	} else if (imgHeight % 32 == 0) {
		return 32;
	} else if (imgHeight % 16 == 0) {
		return 16;
	} else if (imgHeight % 8 == 0) {
		return 8;
	} else if (imgHeight % 4 == 0) {
		return 4;
	} else if (imgHeight % 2 == 0) {
		return 2;
	} else {
		return 1;
	}
}
#endif
