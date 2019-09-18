#include "../../include/weirdlib_image.hpp"
#include "./cuda_utils.cuh"
#include <cuda_runtime.h>
#include <fstream>
#include "../../external/stb/stb_image.h"

namespace wlib
{
namespace image
{
namespace cu
{
	__global__ static void kernel_ConvertUint8ToFloat(const uint8_t* __restrict__ in, float* __restrict__ out) {
		int tid = getGlobalIdx_1x1();
		out[tid] = static_cast<float>(in[tid]);
	}

	__global__ static void kernel_ConvertFloatToUint8(const float* __restrict__ in, uint8_t* __restrict__ out) {
		int tid = getGlobalIdx_1x1();
		out[tid] = static_cast<uint8_t>(in[tid]);
	}

	ImageCUDA::ImageCUDA(const std::string& path, const bool isRawData, const uint64_t _width, const uint64_t _height, const ColorFormat requestedFormat) {
		LoadImage(path, isRawData, _width, _height, requestedFormat);
	}

	ImageCUDA::ImageCUDA(const uint8_t* _pixels, const uint64_t _width, const uint64_t _height, const ColorFormat _format) {
		LoadImage(_pixels, _width, _height, _format);
	}

	ImageCUDA::ImageCUDA(const float* _pixels, const uint64_t _width, const uint64_t _height, const ColorFormat _format) {
		LoadImage(_pixels, _width, _height, _format);
	}

	ImageCUDA::ImageCUDA(const ImageCUDA& other) {
		*this = other;
	}

	ImageCUDA& ImageCUDA::operator=(const ImageCUDA& other) {
		if (this == &other) {
			return *this;
		}

		width = other.width;
		height = other.height;
		format = other.format;

		cudaFree(pixels);

		cudaMalloc(&pixels, GetTotalImageSize(width, height, format)*sizeof(float));
		cudaMemcpy(pixels, other.pixels, GetTotalImageSize(width, height, format)*sizeof(float), cudaMemcpyDeviceToDevice);

		return *this;
	}

	ImageCUDA::ImageCUDA(ImageCUDA&& other) {
		*this = std::move(other);
	}

	ImageCUDA& ImageCUDA::operator=(ImageCUDA&& other) {
		if (this == &other) {
			return *this;
		}

		width = other.width;
		height = other.height;
		format = other.format;

		cudaFree(pixels);

		pixels = other.pixels;
		other.pixels = nullptr;

		return *this;
	}

	ImageCUDA::~ImageCUDA() {
		cudaFree(pixels);
	}

	void ImageCUDA::LoadImage(const std::string& path, const bool isRawData, const uint64_t _width, const uint64_t _height, const ColorFormat requestedFormat) {
		if (isRawData) {
			width = _width;
			height = _height;
			format = requestedFormat;

			float* outPtr_dev;
			uint8_t* inPtr_dev;

			cudaMalloc(&outPtr_dev, GetTotalImageSize(width, height, format) * sizeof(float));
			cudaMalloc(&inPtr_dev, GetTotalImageSize(width, height, format));

			std::ifstream infile(path, std::ios::binary | std::ios::ate);
			size_t filesize = infile.tellg();
			infile.seekg(0);

			auto host_ptr = new uint8_t[GetTotalImageSize(width, height, format)];
			infile.read(reinterpret_cast<char*>(host_ptr), GetTotalImageSize(width, height, format));

			const size_t blockSize = getBlockSize(GetTotalImageSize(width, height, format));
			const size_t gridSize = GetTotalImageSize(width, height, format) / blockSize;

			cudaStream_t stream;
			cudaStreamCreate(&stream);

			cudaMemcpyAsync(inPtr_dev, host_ptr, GetTotalImageSize(width, height, format), cudaMemcpyHostToDevice, stream);
			kernel_ConvertUint8ToFloat<<<gridSize, blockSize, 0, stream>>>(inPtr_dev, outPtr_dev);

			cudaStreamDestroy(stream);
			cudaFree(inPtr_dev);
			pixels = outPtr_dev;

			return;
		}

		int w, h, c;
		auto pixin = stbi_load(path.c_str(), &w, &h, &c, 0);

		width = w;
		height = h;
		format = static_cast<ColorFormat>(c);

		float* outPtr_dev;
		uint8_t* inPtr_dev;

		cudaMalloc(&outPtr_dev, GetTotalImageSize(width, height, format) * sizeof(float));
		cudaMalloc(&inPtr_dev, GetTotalImageSize(width, height, format));

		const size_t blockSize = getBlockSize(GetTotalImageSize(width, height, format));
		const size_t gridSize = GetTotalImageSize(width, height, format) / blockSize;

		cudaStream_t stream;
		cudaStreamCreate(&stream);

		cudaMemcpyAsync(inPtr_dev, pixin, GetTotalImageSize(width, height, format), cudaMemcpyHostToDevice, stream);
		kernel_ConvertUint8ToFloat<<<gridSize, blockSize, 0, stream>>>(inPtr_dev, outPtr_dev);

		cudaStreamDestroy(stream);
		cudaFree(inPtr_dev);
		pixels = outPtr_dev;
	}

	void ImageCUDA::LoadImage(const uint8_t* _pixels, const uint64_t _width, const uint64_t _height, const ColorFormat _format) {
		width = _width;
		height = _height;
		format = _format;

		float* outPtr_dev;
		uint8_t* inPtr_dev;

		cudaMalloc(&outPtr_dev, GetTotalImageSize(width, height, format) * sizeof(float));
		cudaMalloc(&inPtr_dev, GetTotalImageSize(width, height, format));

		const size_t blockSize = getBlockSize(GetTotalImageSize(width, height, format));
		const size_t gridSize = GetTotalImageSize(width, height, format) / blockSize;

		cudaStream_t stream;
		cudaStreamCreate(&stream);

		cudaMemcpyAsync(inPtr_dev, _pixels, GetTotalImageSize(width, height, format), cudaMemcpyHostToDevice, stream);
		kernel_ConvertUint8ToFloat<<<gridSize, blockSize, 0, stream>>>(inPtr_dev, outPtr_dev);

		cudaStreamDestroy(stream);
		cudaFree(inPtr_dev);
		pixels = outPtr_dev;
	}

	void ImageCUDA::LoadImage(const float* _pixels, const uint64_t _width, const uint64_t _height, const ColorFormat _format) {
		width = _width;
		height = _height;
		format = _format;

		float* outPtr_dev;
		cudaMalloc(&outPtr_dev, GetTotalImageSize(width, height, format) * sizeof(float));
		cudaMemcpy(outPtr_dev, _pixels, GetTotalImageSize(width, height, format) * sizeof(float), cudaMemcpyDeviceToDevice);
		pixels = outPtr_dev;
	}

	size_t ImageCUDA::GetTotalImageSize(const uint64_t width, const uint64_t height, const ColorFormat format) noexcept {
		switch (format)
		{
		case ColorFormat::F_Grayscale:
			return width * height;
		case ColorFormat::F_GrayAlpha:
			return width * height * 2;
		case ColorFormat::F_RGB:
		case ColorFormat::F_BGR:
			return width * height * 3;
		case ColorFormat::F_Default:
		case ColorFormat::F_RGBA:
		case ColorFormat::F_BGRA:
			return width * height * 4;
		default:
			return 0;
		}
	}

	std::vector<uint8_t> ImageCUDA::GetPixelsAsInt() {
		uint8_t* dev_ptr;
		cudaMalloc(&dev_ptr, GetTotalImageSize(width, height, format));

		const size_t blockSize = getBlockSize(GetTotalImageSize(width, height, format));
		const size_t gridSize = GetTotalImageSize(width, height, format) / blockSize;

		cudaStream_t stream;
		cudaStreamCreate(&stream);

		kernel_ConvertFloatToUint8<<<gridSize, blockSize, 0, stream>>>(pixels, dev_ptr);
		std::vector<uint8_t> output(GetTotalImageSize(width, height, format));
		cudaMemcpyAsync(output.data(), dev_ptr, GetTotalImageSize(width, height, format), cudaMemcpyDeviceToHost, stream);

		cudaStreamDestroy(stream);
		cudaFree(dev_ptr);

		return std::move(output);
	}

} // namespace cu
} // namespace image
} // namespace wlib
