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
	/// Luminosity: accounts for human perception of colors (default) (BT.709) <br>
	/// Lightness: average of maximum and minimum intensity <br>
	/// Average: simple average of components <br>
	/// LuminosityBT601: Same as Luminosity, but uses BT.601 weights
	enum class GrayscaleMethod
	{
		Luminosity = 0,
		Lightness = 1,
		Average = 2,
		LuminosityBT601 = 3,
		LuminosityBT2100 = 4
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
		Image(const std::string& path, bool isRawData = false, uint64_t width = 0, uint64_t height = 0, ColorFormat requestedFormat = F_Default);

		/// Constructor from in-memory 8-bit per color pixel data
		/// @param width width in pixels
		/// @param height height in pixels
		/// @param format format to interpret data as
		Image(const uint8_t* pixels, uint64_t width, uint64_t height, ColorFormat format);

		/// Constructor from in-memory floating point pixel data
		/// @param width width in pixels
		/// @param height height in pixels
		/// @param format format to interpret data as
		Image(const float* pixels, uint64_t width, uint64_t height, ColorFormat format);

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
		void LoadImage(const std::string& path, bool isRawData = false, uint64_t width = 0, uint64_t height = 0, ColorFormat requestedFormat = F_Default);

		/// Load image from in-memory 8-bit per color pixel data
		/// @param width width in pixels
		/// @param height height in pixels
		/// @param format format to interpret data as
		void LoadImage(const uint8_t* pixels, uint64_t width, uint64_t height, ColorFormat format);

		/// Load image from in-memory floating point pixel data
		/// @param width width in pixels
		/// @param height height in pixels
		/// @param format format to interpret data as
		void LoadImage(const float* pixels, uint64_t width, uint64_t height, ColorFormat format);

		/// Get image width
		/// @return width
		inline auto GetWidth()  const noexcept { return width;  }

		/// Get image height
		/// @return height
		inline auto GetHeight() const noexcept { return height; }

		/// Get image format
		/// @return format as specified in @{link ColorFormat}
		inline auto GetFormat() const noexcept { return format; }

		/// Get read-only raw image pixel data
		/// @return pointer to raw pixel data
		inline auto GetPixels() const noexcept { return pixels.data(); }

		/// Set image width
		inline void SetWidth(uint64_t newVal) noexcept { width = newVal; }

		/// Set image height
		inline void SetHeight(uint64_t newVal) noexcept { height = newVal; }

		/// Set image format
		inline void SetFormat(ColorFormat _format) noexcept { format = _format; }

		/// Get raw image pixel data
		/// @return pointer to raw pixel data
		inline auto GetPixels_Unsafe() noexcept { return pixels.data(); }

		/// Get underlying vector <br>
		/// Invalidates all existing pointers returned ny @{link GetPixels()} and @{link GetPixels_Unsafe()}
		inline std::vector<float>& AccessStorage() { return pixels; }

		/// Get total image size (i.e. width * height * format)
		/// @param width image width
		/// @param height image height
		/// @param format image format @{link ColorFormat}
		/// @return pixel count
		static size_t GetTotalImageSize(uint64_t width, uint64_t height, ColorFormat format) noexcept;

		/// Get total image size (i.e. width * height * format)
		/// @return pixel count
		size_t GetTotalImageSize() const noexcept;

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

		/// Move constructor
		ImageSoA(ImageSoA&&);

		/// Copy assignment operator
		ImageSoA& operator=(const ImageSoA&);

		/// Move assignment operator
		ImageSoA& operator=(ImageSoA&&);

		~ImageSoA();

		/// Realign channels
		/// @return AoS @{link Image}
		Image ConvertToImage();
	};

	/// Convert image into a grayscale representation using various methods described in @{link GrayscaleMethod}
	/// @param inImg image to be modified
	/// @param preserveAlpha whether alpha should be kept or discarded
	/// @param method grayscale calculation method
	/// @return reference to modified input image
	ImageSoA& ConvertToGrayscale(ImageSoA& inImg, bool preserveAlpha = false, GrayscaleMethod method = GrayscaleMethod::Luminosity);

	/// Function converting 8bpc into 32bpc float
	/// @param in pointer to raw image data
	/// @param out caller-managed pointer to destination
	/// @param fileSize total number of pixels
	void ConvertUint8ToFloat(const uint8_t* in, float* out, size_t fileSize);

	/// Function converting 16bpc into 32bpc float
	/// @param in pointer to raw image data
	/// @param out caller-managed pointer to destination
	/// @param fileSize total number of pixels
	void ConvertUint16ToFloat(const uint16_t* in, float* out, const size_t fileSize);

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

	/// Convert input image to RGBA
	/// @param in input image
	void ConvertToRGBA(Image& in);

	/// Convert input image to BGR
	/// @param in input image
	void ConvertToBGR(Image& in);

	/// Convert input image to RGB
	/// @param in input image
	void ConvertToRGB(Image& in);

	/// Convert input image to BGRA
	/// @param in input image
	void ConvertToBGRA(Image& in);

	/// Convert all color values to their image negative version <br>
	/// Effectively does max_val - current_val
	/// @param in input image
	/// @param withAlpha whether to preserve alpha channel
	void NegateValues(ImageSoA& in, bool withAlpha = false);

namespace detail
{
	void swapRAndB_3c(float* in, size_t count);

	void swapRAndB_4c(float* in, size_t count);

	std::vector<float> dropAlpha_4c(Image& in);

	std::vector<float> dropAlpha_2c(Image& in);

	std::vector<float> appendAlpha_3c(Image& in);

	std::vector<float> broadcastGray_to3c(Image& in);

	std::vector<float> broadcastGray_to4c(Image& in);

	std::vector<float> broadcastGrayAlpha_to3c(Image& in);

	std::vector<float> broadcastGrayAlpha_to4c(Image& in);
} // namespace detail

} // namespace image
} // namespace wlib
