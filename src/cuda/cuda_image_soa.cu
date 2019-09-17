#include "../../include/cuda/weirdlib_cuda_image.hpp"
#ifdef WLIB_ENABLE_CUDA
#include <cuda_runtime.h>

namespace wlib
{
namespace image
{
namespace cu
{
	ImageSoACUDA::ImageSoACUDA(const Image& other) {
		width = other.GetWidth();
		height = other.GetHeight();
		format = other.GetFormat();

		ImageSoA tmp(other);

		for (const auto& c : tmp.channels) {
			float* dev_tmp;
			cudaMalloc(&dev_tmp, width * height * sizeof(float));
			cudaMemcpy(dev_tmp, c, width * height * sizeof(float), cudaMemcpyHostToDevice);
			channels.push_back(dev_tmp);
		}
	}

	ImageSoACUDA::ImageSoACUDA(const ImageSoA& other) {
		width = other.width;
		height = other.height;
		format = other.format;

		for (const auto& c : other.channels) {
			float* dev_tmp;
			cudaMalloc(&dev_tmp, width * height * sizeof(float));
			cudaMemcpy(dev_tmp, c, width * height * sizeof(float), cudaMemcpyHostToDevice);
			channels.push_back(dev_tmp);
		}
	}

	ImageSoACUDA::ImageSoACUDA(const ImageSoACUDA& other) {
		*this = other;
	}

	ImageSoACUDA::ImageSoACUDA(ImageSoACUDA&& other) {
		*this = other;
	}

	ImageSoACUDA::~ImageSoACUDA() {
		for (const auto& c : channels) {
			cudaFree(c);
		}
	}

	ImageSoACUDA& ImageSoACUDA::operator=(const ImageSoACUDA& img) {
		if (&img == this) {
			return *this;
		}

		height = img.height;
		width = img.width;
		format = img.format;

		for (auto& ptr : channels) {
			cudaFree(ptr);
		}

		if (channels.size() != img.channels.size()) {
			channels.resize(img.channels.size());
			channels.shrink_to_fit();
		}

		for (size_t i = 0; i < channels.size(); i++) {
			float* dev_chan;
			cudaMalloc(&dev_chan, width * height * sizeof(float));
			cudaMemcpy(dev_chan, img.channels[i], width * height * sizeof(float), cudaMemcpyDeviceToDevice);
			channels[i] = dev_chan;
		}

		return *this;
	}

	ImageSoACUDA& ImageSoACUDA::operator=(ImageSoACUDA&& img) {
		if (&img == this) {
			return *this;
		}

		height = img.height;
		width = img.width;
		format = img.format;

		for (auto& ptr : channels) {
			cudaFree(ptr);
		}

		channels.resize(img.channels.size());
		channels.shrink_to_fit();
		channels.assign(img.channels.begin(), img.channels.end());

		for (auto& ptr : img.channels) {
			ptr = nullptr;
		}

		return *this;
	}



	ImageSoA ImageSoACUDA::ConvertToImageSoA() {
		ImageSoA soa;
		soa.height = height;
		soa.width = width;
		soa.format = format;

		for (auto& c: channels) {
			float* host_tmp = new float[width * height];
			cudaMemcpy(host_tmp, c, width * height * sizeof(float), cudaMemcpyDeviceToHost);
			soa.channels.push_back(host_tmp);
		}

		return soa;
	}

	Image ImageSoACUDA::ConvertToImage() {
		auto soa = ConvertToImageSoA();
		return soa.ConvertToImage();
	}



} // namespace cu
} // namespace image
} // namespace wlib
#endif
