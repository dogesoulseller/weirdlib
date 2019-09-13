#pragma once
#include <vector>
#include <memory>
#include <string>
#include <type_traits>

namespace wlib
{

/// Image loading and processing <br>
/// Conversion rules: <br>
/// - Grayscale[A] to color duplicates gray channel to color channels
/// - RGB[A] to BGR[A] simply swaps channels
/// - RGB[A]/BGR[A] to grayscale uses method from @{link GrayscaleMethod}

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

	/// Method for computing grayscale <br>
	/// Luminosity: accounts for human perception of colors (default) <br>
	/// Lightness: average of maximum and minimum intensity <br>
	/// Average: simple average of components
	enum class GrayscaleMethod
	{
		Luminosity = 0,
		Lightness = 1,
		Average = 2
	};

	/// Class representing a 2D image in Array of Structures (i.e. RGB RGB RGB), converts to a floating point representation on load
	class Image
	{
	  private:
		std::vector<float> pixels;
		ColorFormat format;
		uint64_t width;
		uint64_t height;

	  public:
	  	/// Default constructor
		Image() = default;

		/// Constructor from a file readable by the filesystem
		/// @param path absolute or relative path to supported image file
		/// @param isRawData whether to load data as a dump of pixel values
		/// @param width width in pixels
		/// @param height height in pixels
		/// @param requestedFormat format to convert to on load, or in case of raw data, to interpret as
		Image(const std::string& path, const bool isRawData = false, const uint64_t width = 0, const uint64_t height = 0, const ColorFormat requestedFormat = F_Default);

		/// Constructor from in-memory 8-bit per color pixel data
		/// @param width width in pixels
		/// @param height height in pixels
		/// @param format format to interpret data as
		Image(const uint8_t* pixels, const uint64_t width, const uint64_t height, const ColorFormat format);

		/// Constructor from in-memory floating point pixel data
		/// @param width width in pixels
		/// @param height height in pixels
		/// @param format format to interpret data as
		Image(const float* pixels, const uint64_t width, const uint64_t height, const ColorFormat format);

		/// Copy constructor
		Image(const Image&) = default;

		/// Copy assignment operator
		Image& operator=(const Image&) = default;

		/// Move constructor
		Image(Image&&) = default;

		/// Move assignment operator
		Image& operator=(Image&&) = default;

		~Image() = default;


		/// Load image from a file readable by the filesystem
		/// @param path absolute or relative path to supported image file
		/// @param isRawData whether to load data as a dump of pixel values
		/// @param width width in pixels
		/// @param height height in pixels
		/// @param requestedFormat format to convert to on load, or in case of raw data, to interpret as
		void LoadImage(const std::string& path, const bool isRawData = false, const uint64_t width = 0, const uint64_t height = 0, const ColorFormat requestedFormat = F_Default);

		/// Load image from in-memory 8-bit per color pixel data
		/// @param width width in pixels
		/// @param height height in pixels
		/// @param format format to interpret data as
		void LoadImage(const uint8_t* pixels, const uint64_t width, const uint64_t height, const ColorFormat format);

		/// Load image from in-memory floating point pixel data
		/// @param width width in pixels
		/// @param height height in pixels
		/// @param format format to interpret data as
		void LoadImage(const float* pixels, const uint64_t width, const uint64_t height, const ColorFormat format);

		/// Get iamge width
		/// @return width
		inline auto GetWidth()  const noexcept { return width;  }

		/// Get image height
		/// @return height
		inline auto GetHeight() const noexcept { return height; }

		/// Get image format
		/// @return format as specified in @{link ColorFormat}
		inline auto GetFormat() const noexcept { return format; }

		/// Get raw image pixel data
		/// @return pointer to raw pixel data
		inline auto GetPixels() const noexcept { return pixels.data(); }

		/// Get total image size (i.e. width * height * format)
		/// @param width image width
		/// @param height image height
		/// @param format image format @{link ColorFormat}
		/// @return pixel count
		static size_t GetTotalImageSize(const uint64_t width, const uint64_t height, const ColorFormat format) noexcept;

		/// Get pixel values as 8-bit integers
		/// @return vector of pixels represented as 8-bit unsigned integers in AoS order
		std::vector<uint8_t> GetPixelsAsInt();
	};

	/// Class representing a 2D image in Structure of Arrays format (i.e. RRR GGG BBB) <br>
	/// This format is more optimal for threading and vectorization at a relatively small conversion cost
	class ImageSoA
	{
		public:
		uint64_t width;
		uint64_t height;
		ColorFormat format;
		std::vector<float*> channels;

		/// Object is in invalid state after default construction
		ImageSoA() = default;

		/// Construct from AoS @{link Image}
		ImageSoA(const Image&);

		/// Copy constructor
		ImageSoA(const ImageSoA&);

		/// Copy assignment operator
		ImageSoA& operator=(const ImageSoA&);

		~ImageSoA();

		/// Realign channels using unpack instructions
		/// @return AoS @{link Image}
		Image ConvertToImage();
	};

	namespace cu
	{
		/// Class representing a 2D image in Structure of Arrays format, stored on the GPU (i.e. RRR GGG BBB) <br>
		class ImageSoACUDA
		{
			public:
			uint64_t width;
			uint64_t height;
			ColorFormat format;
			std::vector<float*> channels;

			/// Object is in invalid state after default construction
			ImageSoACUDA() = default;

			/// Construct from CPU AoS @{link Image}
			ImageSoACUDA(const Image&);

			/// Construct from CPU SoA @{link ImageSoA}
			ImageSoACUDA(const ImageSoA&);

			/// Copy constructor
			ImageSoACUDA(const ImageSoACUDA&);

			/// Copy assignment operator
			ImageSoACUDA& operator=(const ImageSoACUDA&);

			~ImageSoACUDA();

			/// Convert to SoA image in CPU memory
			ImageSoA ConvertToImageSoA();

			/// Convert to AoS image in CPU memory
			Image ConvertToImage();
		};

	} // namespace cu


	/// Convert image into a grayscale representation using various methods described in @{link GrayscaleMethod}
	/// @param inImg image to be modified
	/// @param preserveAlpha whether alpha should be kept or discarded
	/// @param method grayscale calculation method
	/// @return reference to modified input image
	ImageSoA& ConvertToGrayscale(ImageSoA& inImg, const bool preserveAlpha = false, const GrayscaleMethod method = GrayscaleMethod::Luminosity);

	/// Function converting 8bpc into 32bpc float
	/// @param in pointer to raw image data
	/// @param out caller-managed pointer to destination
	/// @param fileSize total number of pixels
	void ConvertUint8ToFloat(const uint8_t* in, float* out, const size_t fileSize);

	/// Convert input image to RGBA
	/// @param in input image
	void ConvertToRGBA(ImageSoA& in);

	/// Convert input image to BGR
	/// @param in input image
	void ConvertToBGR(ImageSoA& in);

	/// Convert input image to RGB
	/// @param in input image
	void ConvertToRGB(ImageSoA& in);

	/// Convert input image to BGRA
	/// @param in input image
	void ConvertToBGRA(ImageSoA& in);

	/// Convert all color values to their image negative version <br>
	/// Effectively does max_val - current_val
	/// @param in input image
	/// @param withAlpha whether to preserve alpha channel
	void NegateValues(ImageSoA& in, const bool withAlpha = false);

} // namespace image
} // namespace wlib
