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
	class image_channel_error : public std::exception
	{
		std::string what_message;

		public:
		inline image_channel_error(const std::string& msg) noexcept {
			what_message = msg;
		}

		inline image_channel_error(std::string&& msg) noexcept {
			what_message = std::move(msg);
		}

		explicit inline image_channel_error(const char* msg) noexcept {
			what_message = std::string(msg);
		}

		[[nodiscard]] inline const char* what() const noexcept override
		{
			return what_message.c_str();
		}
	};

	class image_dimensions_error : public std::exception
	{
		std::string what_message;

		public:
		inline image_dimensions_error(const std::string& msg) noexcept {
			what_message = msg;
		}

		inline image_dimensions_error(std::string&& msg) noexcept {
			what_message = std::move(msg);
		}

		explicit inline image_dimensions_error(const char* msg) noexcept {
			what_message = std::string(msg);
		}

		[[nodiscard]] inline const char* what() const noexcept override
		{
			return what_message.c_str();
		}
	};

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
	/// LuminosityBT601: Same as Luminosity, but uses BT.601 weights <br>
	/// LuminosityBT2100: Same as Luminosity, but uses BT.2100 weights <br>
	enum class GrayscaleMethod
	{
		Luminosity = 0,
		Lightness = 1,
		Average = 2,
		LuminosityBT601 = 3,
		LuminosityBT2100 = 4
	};

	template<typename FloatT = float>
	struct PSNRData
	{
		std::vector<FloatT> PSNRPerChannel;
		std::vector<FloatT> MSEPerChannel;
	};


	/// Class representing a 2D image in Array of Structures (i.e. RGB RGB RGB), converts to a non-normalized floating point representation on load
	class Image
	{
	  private:
		std::vector<float> pixels;
		ColorFormat format;
		uint64_t width;
		uint64_t height;

	  public:
		Image() = default;

		/// Constructor from a file readable by the filesystem
		/// @param isRawData whether to load data as a dump of pixel values
		/// @param requestedFormat format to convert to on load, or in case of raw data, to interpret as
		Image(const std::string& path, bool isRawData = false, uint64_t width = 0, uint64_t height = 0, ColorFormat requestedFormat = F_Default);

		/// Constructor from in-memory 8-bit per color pixel data
		/// @param format format to interpret data as
		Image(const uint8_t* pixels, uint64_t width, uint64_t height, ColorFormat format);

		// TODO: Allow loading from normalized pixel data

		/// Constructor from in-memory non-normalized floating point pixel data
		/// @param format format to interpret data as
		Image(const float* pixels, uint64_t width, uint64_t height, ColorFormat format);

		Image(const Image&) = default;
		Image(Image&&) = default;

		Image& operator=(const Image&) = default;
		Image& operator=(Image&&) = default;

		~Image() = default;


		/// Convert input image to RGBA
		static void ConvertToRGBA(Image& in);

		/// Convert input image to BGR
		static void ConvertToBGR(Image& in);

		/// Convert input image to RGB
		static void ConvertToRGB(Image& in);

		/// Convert input image to BGRA
		static void ConvertToBGRA(Image& in);

		/// Convert input image to RGBA
		void ConvertToRGBA();

		/// Convert input image to BGR
		void ConvertToBGR();

		/// Convert input image to RGB
		void ConvertToRGB();

		/// Convert input image to BGRA
		void ConvertToBGRA();

		/// Load image from a file readable by the filesystem
		/// @param isRawData whether to load data as a dump of pixel values
		/// @param requestedFormat format to convert to on load, or in case of raw data, to interpret as
		void LoadImage(const std::string& path, bool isRawData = false, uint64_t width = 0, uint64_t height = 0, ColorFormat requestedFormat = F_Default);

		/// Load image from in-memory 8-bit per color pixel data
		/// @param format format to interpret data as
		void LoadImage(const uint8_t* pixels, uint64_t width, uint64_t height, ColorFormat format);

		/// Load image from in-memory floating point pixel data
		/// @param format format to interpret data as
		void LoadImage(const float* pixels, uint64_t width, uint64_t height, ColorFormat format);

		/// Get image width
		inline auto GetWidth()  const noexcept { return width;  }

		/// Get image height
		inline auto GetHeight() const noexcept { return height; }

		/// Get image format
		inline auto GetFormat() const noexcept { return format; }

		/// Get read-only raw image pixel data
		inline auto GetPixels() const noexcept { return pixels.data(); }

		/// Set image width
		inline void SetWidth(uint64_t newVal) noexcept { width = newVal; }

		/// Set image height
		inline void SetHeight(uint64_t newVal) noexcept { height = newVal; }

		/// Set image format
		inline void SetFormat(ColorFormat _format) noexcept { format = _format; }

		/// Get raw image pixel data
		inline auto GetPixels_Unsafe() noexcept { return pixels.data(); }

		/// Get underlying vector <br>
		/// Invalidates all existing pointers returned ny @{link GetPixels()} and @{link GetPixels_Unsafe()}
		inline std::vector<float>& AccessStorage() { return pixels; }

		/// Get total image size (i.e. width * height * format)
		static size_t GetTotalImageSize(uint64_t width, uint64_t height, ColorFormat format) noexcept;

		/// Get total image size (i.e. width * height * format)
		size_t GetTotalImageSize() const noexcept;

		/// Get pixel values as 8-bit integers
		/// @return vector of pixels represented as 8-bit unsigned integers in AoS order
		std::vector<uint8_t> GetPixelsAsInt();
	};

	// TODO: Unify Image and ImageSoA interfaces
	/// Class representing a 2D image in Structure of Arrays format (i.e. RRR GGG BBB) <br>
	/// This format is more optimal for threading and vectorization at a relatively small conversion cost
	class ImageSoA
	{
		private:
		uint64_t width;
		uint64_t height;
		ColorFormat format;
		std::vector<float*> channels;

		public:
		/// Object is in invalid state after default construction
		ImageSoA() = default;

		ImageSoA(const Image&);

		ImageSoA(const ImageSoA&);
		ImageSoA(ImageSoA&&);

		ImageSoA& operator=(const ImageSoA&);
		ImageSoA& operator=(ImageSoA&&);

		/// Convert input image to RGBA
		static void ConvertToRGBA(ImageSoA& in);

		/// Convert input image to BGR
		static void ConvertToBGR(ImageSoA& in);

		/// Convert input image to RGB
		static void ConvertToRGB(ImageSoA& in);

		/// Convert input image to BGRA
		static void ConvertToBGRA(ImageSoA& in);


		/// Convert input image to RGBA
		inline void ConvertToRGBA() {ConvertToRGBA(*this);};

		/// Convert input image to BGR
		inline void ConvertToBGR() {ConvertToBGR(*this);};

		/// Convert input image to RGB
		inline void ConvertToRGB() {ConvertToRGB(*this);};

		/// Convert input image to BGRA
		inline void ConvertToBGRA() {ConvertToBGRA(*this);};


		/// Get image width
		inline auto GetWidth()  const noexcept { return width;  }

		/// Get image height
		inline auto GetHeight() const noexcept { return height; }

		/// Get image format
		/// @return format as specified in @{link ColorFormat}
		inline auto GetFormat() const noexcept { return format; }

		/// Get read-only reference to vector of channels
		inline auto& GetChannels() const noexcept { return channels; }

		/// Get read-write reference to vector of channels
		inline auto& AccessChannels() noexcept { return channels; }


		/// Set image width
		inline void SetWidth(uint64_t newVal) noexcept { width = newVal; }

		/// Set image height
		inline void SetHeight(uint64_t newVal) noexcept { height = newVal; }

		/// Set image format
		inline void SetFormat(ColorFormat _format) noexcept { format = _format; }


		/// Get total image size (i.e. width * height * format)
		static size_t GetTotalImageSize(uint64_t width, uint64_t height, ColorFormat format) noexcept {return Image::GetTotalImageSize(width, height, format);}

		/// Get total image size (i.e. width * height * format)
		inline size_t GetTotalImageSize() const noexcept { return GetTotalImageSize(width, height, format);}

		~ImageSoA();
	};

	/// Make new SoA image using AoS image as source
	ImageSoA MakeSoAFromAoS(const Image& in);

	/// Make new AoS image using AoS image as source
	Image MakeAoSFromSoA(const ImageSoA& in);


	namespace detail
	{
		PSNRData<float> psnrFloat(const ImageSoA& image0, const ImageSoA& image1);
		PSNRData<double> psnrDouble(const ImageSoA& image0, const ImageSoA& image1);
	} // namespace detail


	template<typename FloatT = float>
	PSNRData<FloatT> CalculatePSNR(const ImageSoA& image0, const ImageSoA& image1) {
		static_assert(std::is_same_v<FloatT, float> || std::is_same_v<FloatT, double>);

		if constexpr (std::is_same_v<FloatT, float>) {
			return detail::psnrFloat(image0, image1);
		} else {
			return detail::psnrDouble(image0, image1);
		}
	}

	/// Convert image into a grayscale representation using methods described in @{link GrayscaleMethod}
	/// @param preserveAlpha whether alpha should be kept or discarded
	/// @param method grayscale calculation method
	/// @return reference to modified input image
	ImageSoA& ConvertToGrayscale(ImageSoA& inImg, bool preserveAlpha = false, GrayscaleMethod method = GrayscaleMethod::Luminosity);

	/// Convert 8bpc unsigned int data into 32bpc float data
	/// @param in pointer to raw image data
	/// @param out caller-managed pointer to destination
	/// @param fileSize total number of pixels
	void ConvertUint8ToFloat(const uint8_t* in, float* out, size_t fileSize);

	/// Convert 16bpc unsigned int data into 32bpc float data
	/// @param in pointer to raw image data
	/// @param out caller-managed pointer to destination
	/// @param fileSize total number of pixels
	void ConvertUint16ToFloat(const uint16_t* in, float* out, const size_t fileSize);

	/// Convert all color values to their image negative version <br>
	/// Effectively does max_val - current_val
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

		void extendGSTo3Chan(ImageSoA& in);

		void extendGSTo4Chan(ImageSoA& in, bool constantAlpha);

		void appendConstantAlpha(ImageSoA& in);
	} // namespace detail

} // namespace image
} // namespace wlib
