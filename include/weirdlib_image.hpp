#pragma once
#include <vector>
#include <memory>
#include <string>
#include <type_traits>

namespace wlib
{

/// Image loading and processing
namespace image
{
	/// Image format information
	enum ColorFormat
	{
		F_Grayscale = 1,
		F_GrayAlpha = 2,
		F_RGB = 3,
		F_RGBA = 4,
		F_BGR,
		F_BGRA,
		F_Default = -1u
	};

	/// Method for computing grayscale
	/// Luminosity: accounts for human perception of colors (default)
	/// Lightness: average of maximum and minimum intensity
	/// Average: simple average of components
	enum class GrayscaleMethod
	{
		Luminosity = 0,
		Lightness = 1,
		Average = 2
	};

	class Image
	{
	  private:
		std::vector<float> pixels;
		ColorFormat format;
		uint64_t width;
		uint64_t height;

	  public:
		Image() = default;

		Image(const std::string& path, bool isRawData = false, const uint64_t width = 0, const uint64_t height = 0, ColorFormat requestedFormat = F_Default);
		Image(const uint8_t* pixels, const uint64_t width, const uint64_t height, const ColorFormat format);
		Image(const float* pixels, const uint64_t width, const uint64_t height, const ColorFormat format);

		Image(const Image&) = default;
		Image& operator=(const Image&) = default;

		Image(Image&&) = default;
		Image& operator=(Image&&) = default;

		~Image() = default;

		void LoadImage(const std::string& path, bool isRawData = false, const uint64_t width = 0, const uint64_t height = 0, ColorFormat requestedFormat = F_Default);
		void LoadImage(const uint8_t* pixels, const uint64_t width, const uint64_t height, const ColorFormat format);
		void LoadImage(const float* pixels, const uint64_t width, const uint64_t height, const ColorFormat format);

		inline auto GetWidth()  const noexcept { return width;  }
		inline auto GetHeight() const noexcept { return height; }
		inline auto GetFormat() const noexcept { return format; }
		inline const auto GetPixels() const noexcept { return pixels.data(); }

		static size_t GetTotalImageSize(const uint64_t width, const uint64_t height, const ColorFormat format) noexcept;
	};

	class ImageSoA
	{
		public:
		uint64_t width;
		uint64_t height;
		ColorFormat format;
		std::vector<float*> channels;

		ImageSoA() = delete;
		ImageSoA(const Image&);
		ImageSoA(const ImageSoA&);

		ImageSoA& operator=(const ImageSoA&);

		~ImageSoA();

		/// Realign channels using unpack instructions
		Image ConvertToImage();
	};

	ImageSoA& ConvertToGrayscale(ImageSoA& inImg, bool preserveAlpha = false, GrayscaleMethod method = GrayscaleMethod::Luminosity);

	void ConvertUint8ToFloat(const uint8_t* in, float* out, size_t fileSize);

	void ConvertToRGBA(ImageSoA&);
	void ConvertToBGR(ImageSoA&);
	void ConvertToRGB(ImageSoA&);
	void ConvertToBGRA(ImageSoA&);

} // namespace image
} // namespace wlib
