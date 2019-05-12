#include "../../include/weirdlib_image.hpp"
#include "../../include/cpu_detection.hpp"
#include <cmath>
#include <algorithm>

namespace wlib::image
{
	ImageSoA::ImageSoA(const Image& img) {
		format = img.GetFormat();
		width = img.GetWidth();
		height = img.GetHeight();

		switch (format)
		{
		case ColorFormat::F_Grayscale: {

			alignas(64) auto c0 = new float[width*height];

			auto source = img.GetPixels();

			#pragma GCC ivdep
			#pragma ivdep
			#pragma loop ivdep
			#pragma omp simd
			for (size_t i = 0; i < width * height; i++) {
				c0[i] = static_cast<float>(source[i]);
			}

			channels.push_back(c0);
			break;

		}
		case ColorFormat::F_GrayAlpha: {

			alignas(64) auto c0 = new float[width*height];
			alignas(64) auto c1 = new float[width*height];

			auto source = img.GetPixels();

			#pragma GCC ivdep
			#pragma ivdep
			#pragma loop ivdep
			#pragma omp simd
			for (size_t i = 0; i < width * height; i++) {
				c0[i] = static_cast<float>(source[i*2]);
				c1[i] = static_cast<float>(source[i*2+1]);
			}

			channels.push_back(c0);
			channels.push_back(c1);

		}
			break;
		case ColorFormat::F_RGB:
		case ColorFormat::F_BGR: {

			alignas(64) auto c0 = new float[width*height];
			alignas(64) auto c1 = new float[width*height];
			alignas(64) auto c2 = new float[width*height];

			auto source = img.GetPixels();

			#pragma GCC ivdep
			#pragma ivdep
			#pragma loop ivdep
			#pragma omp simd
			for (size_t i = 0; i < width * height; i++) {
				c0[i] = static_cast<float>(source[i*3]);
				c1[i] = static_cast<float>(source[i*3+1]);
				c2[i] = static_cast<float>(source[i*3+2]);
			}

			channels.push_back(c0);
			channels.push_back(c1);
			channels.push_back(c2);

		}
			break;
		case ColorFormat::F_RGBA:
		case ColorFormat::F_BGRA:
		case ColorFormat::F_Default: {

			alignas(64) auto c0 = new float[width*height];
			alignas(64) auto c1 = new float[width*height];
			alignas(64) auto c2 = new float[width*height];
			alignas(64) auto c3 = new float[width*height];

			auto source = img.GetPixels();

			#pragma GCC ivdep
			#pragma ivdep
			#pragma loop ivdep
			#pragma omp simd
			for (size_t i = 0; i < width * height; i++) {
				c0[i] = static_cast<float>(source[i*4]);
				c1[i] = static_cast<float>(source[i*4+1]);
				c2[i] = static_cast<float>(source[i*4+2]);
				c3[i] = static_cast<float>(source[i*4+3]);
			}

			channels.push_back(c0);
			channels.push_back(c1);
			channels.push_back(c2);
			channels.push_back(c3);
		}

			break;
		default:
			break;
		}
	}

	ImageSoA::ImageSoA(const ImageSoA& img) {
		*this = img;
	}

	ImageSoA& ImageSoA::operator=(const ImageSoA& img) {
		if (&img == this) {
			return *this;
		}

		height = img.height;
		width = img.width;
		format = img.format;

		for (auto& ptr : channels) {
			delete[] ptr;
		}

		if (channels.size() != img.channels.size()) {
			channels.resize(img.channels.size());
			channels.shrink_to_fit();
		}

		for (size_t i = 0; i < channels.size(); i++) {
			auto chan = new float[Image::GetTotalImageSize(width, height, format)];
			std::copy(img.channels[i], img.channels[i] + Image::GetTotalImageSize(width, height, format), chan);
			channels[i] = chan;
		}

		return *this;
	}

	ImageSoA::~ImageSoA() {
		for (auto& c : channels) {
			delete[] c;
		}
	}

	Image ImageSoA::ConvertToImage() {
		alignas(64) float* outputPix;

		// TODO: Unpack SIMD
		switch (format)
		{
		case ColorFormat::F_Grayscale:
			outputPix = new float[width*height];
			std::copy(channels[0], channels[0]+width*height, outputPix);
			break;
		case ColorFormat::F_GrayAlpha:
			outputPix = new float[width*height*2];
			for (size_t i = 0; i < width * height; i++) {
				outputPix[i*2] = channels[0][i];
				outputPix[i*2+1] = channels[1][i];
			}
			break;
		case ColorFormat::F_RGB:
		case ColorFormat::F_BGR:
			outputPix = new float[width*height*3];
			for (size_t i = 0; i < width * height; i++) {
				outputPix[i*3] = channels[0][i];
				outputPix[i*3+1] = channels[1][i];
				outputPix[i*3+2] = channels[2][i];
			}
			break;
		case ColorFormat::F_Default:
		case ColorFormat::F_RGBA:
		case ColorFormat::F_BGRA:
			outputPix = new float[width*height*4];
			for (size_t i = 0; i < width * height; i++) {
				outputPix[i*4] = channels[0][i];
				outputPix[i*4+1] = channels[1][i];
				outputPix[i*4+2] = channels[2][i];
				outputPix[i*4+3] = channels[3][i];
			}
			break;
		}

		Image imgOut(outputPix, width, height, format);
		delete[] outputPix;
		return imgOut;
	}

	ImageSoA& ConvertToGrayscale(ImageSoA& inImg, bool preserveAlpha) {
		// Converting to grayscale from grayscale...
		if (inImg.format == F_GrayAlpha || inImg.format == F_Grayscale) {
			return inImg;
		}

		size_t channelRedN;
		size_t channelGreenN = 1;
		size_t channelBlueN;

		if (inImg.format == F_RGB || inImg.format == F_RGBA) {
			channelRedN = 0;
			channelBlueN = 2;
		} else {
			channelRedN = 2;
			channelBlueN = 0;
		}

		alignas(64) auto outGr = new float[inImg.width * inImg.height];

		// Processing with SIMD
		#if X86_SIMD_LEVEL >= 7
			const auto redMul = _mm256_set1_ps(0.2126f);
			const auto greenMul = _mm256_set1_ps(0.7152f);
			const auto blueMul = _mm256_set1_ps(0.0722f);
			const auto maxMask = _mm256_set1_ps(255.0f);

			const auto iter_AVX = (inImg.width * inImg.height) / 8;
			const auto iterRem_AVX = (inImg.width * inImg.height) % 8;

			for (size_t i = 0; i < iter_AVX; i++) {
				const auto rChan = _mm256_load_ps(inImg.channels[channelRedN]+i*8);
				const auto gChan = _mm256_load_ps(inImg.channels[channelGreenN]+i*8);
				const auto bChan = _mm256_load_ps(inImg.channels[channelBlueN]+i*8);

				#ifdef X86_SIMD_FMA
					const auto resultGr = _mm256_min_ps(_mm256_fmadd_ps(rChan, redMul, _mm256_fmadd_ps(gChan, greenMul, _mm256_mul_ps(bChan, blueMul))), maxMask);
				#else
					const auto resultGr = _mm256_min_ps(
						_mm256_add_ps(_mm256_mul_ps(rChan, redMul), _mm256_add_ps(_mm256_mul_ps(gChan, greenMul), _mm256_mul_ps(bChan, blueMul))), maxMask);
				#endif
					_mm256_store_ps(outGr+8*i, resultGr);
			}

			for (size_t i = iter_AVX * 8; i < iterRem_AVX + iter_AVX*8; i++) {
				if (const auto result = std::fma(inImg.channels[channelRedN][i], 0.2126f, std::fma(inImg.channels[channelGreenN][i], 0.7152f, inImg.channels[channelBlueN][i] * 0.0722f)); result > 255.0f) {
					outGr[i] = 255.0f;
				} else {
					outGr[i] = result;
				}
			}

			_mm256_zeroupper();
		#elif X86_SIMD_LEVEL >= 1
			const auto redMul = _mm_set1_ps(0.2126f);
			const auto greenMul = _mm_set1_ps(0.7152f);
			const auto blueMul = _mm_set1_ps(0.0722f);
			const auto maxMask = _mm_set1_ps(255.0f);

			const auto iter_SSE = (inImg.width * inImg.height) / 4;
			const auto iterRem_SSE = (inImg.width * inImg.height) % 4;

			for (size_t i = 0; i < iter_SSE; i++) {
				const auto rChan = _mm_load_ps(inImg.channels[channelRedN]+i*4);
				const auto gChan = _mm_load_ps(inImg.channels[channelGreenN]+i*4);
				const auto bChan = _mm_load_ps(inImg.channels[channelBlueN]+i*4);

				#ifdef X86_SIMD_FMA	// Technically shouldn't be possible
					const auto resultGr = _mm_min_ps(_mm_fmadd_ps(rChan, redMul, _mm256_fmadd_ps(gChan, greenMul, _mm_mul_ps(bChan, blueMul))), maxMask);
				#else
					const auto resultGr = _mm_min_ps(
						_mm_add_ps(_mm_mul_ps(rChan, redMul), _mm_add_ps(_mm_mul_ps(gChan, greenMul), _mm_mul_ps(bChan, blueMul))), maxMask);
				#endif
					_mm_store_ps(outGr+4*i, resultGr);
			}

			for (size_t i = iter_SSE * 4; i < iterRem_SSE + iter_SSE * 4; i++) {
				if (const auto result = std::fma(inImg.channels[channelRedN][i], 0.2126f, std::fma(inImg.channels[channelGreenN][i], 0.7152f, inImg.channels[channelBlueN][i] * 0.0722f)); result > 255.0f) {
					outGr[i] = 255.0f;
				} else {
					outGr[i] = result;
				}
			}
		#else
			for (size_t i = 0; i < inImg.width * inImg.height; i++) {
				if (const auto result = std::fma(inImg.channels[channelRedN][i], 0.2126f, std::fma(inImg.channels[channelGreenN][i], 0.7152f, inImg.channels[channelBlueN][i] * 0.0722f)); result > 255.0f) {
					outGr[i] = 255.0f;
				} else {
					outGr[i] = result;
				}
			}
		#endif

		// Copy channels
		if (inImg.format == F_RGB || inImg.format == F_BGR || !preserveAlpha) {
			inImg.format = F_Grayscale;

			for (auto& ptr : inImg.channels) {
				delete[] ptr;
			}

			inImg.channels.resize(1);
			inImg.channels.shrink_to_fit();
			inImg.channels[0] = outGr;
		} else if (inImg.format == F_RGBA || inImg.format == F_BGRA) {
			inImg.format = F_GrayAlpha;

			std::for_each(inImg.channels.begin(), inImg.channels.end()-1, [](auto elem){delete[] elem;});

			auto imgAlpha = inImg.channels[3];

			inImg.channels.resize(2);
			inImg.channels.shrink_to_fit();
			inImg.channels[0] = outGr;
			inImg.channels[1] = imgAlpha;
		}

		return inImg;
	}

	void ConvertUint8ToFloat(const uint8_t* in, float* out, size_t fileSize) {
		#if X86_SIMD_LEVEL >= 5
			size_t iters = fileSize / 4;
			size_t itersRem = fileSize % 4;

			for (size_t i = 0; i < iters; i++) {
				__m128i pix_SSE = _mm_loadu_si128(reinterpret_cast<const __m128i*>(in + i*4));
				pix_SSE = _mm_cvtepu8_epi32(pix_SSE);
				__m128 pixf_SSE = _mm_cvtepi32_ps(pix_SSE);
				_mm_storeu_ps(out + i*4, pixf_SSE);
			}

			for (size_t i = iters * 4; i < iters * 4 + itersRem; i++) {
				out[i] = static_cast<float>(in[i]);
			}

		#else
			for (size_t i = 0; i < fileSize; i++) {
				out[i] = static_cast<float>(in[i]);
			}
		#endif
	}
} // namespace wlib::image
