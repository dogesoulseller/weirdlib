#ifdef WEIRDLIB_ENABLE_FILE_OPERATIONS
#include "../../include/weirdlib_fileops.hpp"
#include <array>

namespace wlib::file
{

	constexpr std::array<const char*, 42> _makeFiletypeMappings() noexcept {
		std::array<const char*, 42> output = {};
		output[FILETYPE_JPEG] = ".jpg";
		output[FILETYPE_PNG] = ".png";
		output[FILETYPE_BMP] = ".bmp";
		output[FILETYPE_TIFF] = ".tiff";
		output[FILETYPE_TGA] = ".tga";
		output[FILETYPE_GIF] = ".gif";
		output[FILETYPE_PSD] = ".psd";
		output[FILETYPE_PSB] = ".psb";
		output[FILETYPE_WEBP] = ".webp";
		output[FILETYPE_FLIF] = ".flif";
		output[FILETYPE_PBM] = ".pbm";
		output[FILETYPE_PGM] = ".pgm";
		output[FILETYPE_PPM] = ".ppm";
		output[FILETYPE_PAM] = ".pam";
		output[FILETYPE_XML] = ".xml";
		output[FILETYPE_SVG] = ".svg";
		output[FILETYPE_PDF] = ".pdf";
		output[FILETYPE_MATROSKA] = ".mkv";
		output[FILETYPE_AVI] = ".avi";
		output[FILETYPE_MP4] = ".mp4";
		output[FILETYPE_FLV] = ".flv";
		output[FILETYPE_F4V] = ".f4v";
		output[FILETYPE_WEBM] = ".webm";
		output[FILETYPE_WAVE] = ".wav";
		output[FILETYPE_OGG] = ".ogg";
		output[FILETYPE_APE] = ".ape";
		output[FILETYPE_TTA] = ".tta";
		output[FILETYPE_WAVPACK] = ".wv";
		output[FILETYPE_FLAC] = ".flac";
		output[FILETYPE_CAF] = ".caf";
		output[FILETYPE_OPTIMFROG] = ".ofr";
		output[FILETYPE_3GP] = ".3gp";
		output[FILETYPE_3G2] = ".3g2";
		output[FILETYPE_AIFF] = ".aiff";
		output[FILETYPE_7Z] = ".7z";
		output[FILETYPE_RAR] = ".rar";
		output[FILETYPE_TAR] = ".tar";
		output[FILETYPE_BZIP2] = ".bz2";
		output[FILETYPE_GZIP] = ".gz";
		output[FILETYPE_LZIP] = ".lz";
		output[FILETYPE_ZSTD] = ".zst";
		output[FILETYPE_XZ] = ".xz";

		return output;
	}

	constexpr std::array<const char*, 42> FILETYPE_MAPPINGS = _makeFiletypeMappings();

	std::string GetFiletypeExtension(const FileType type) noexcept {
		if (type == FILETYPE_UNKNOWN || type > 41) {
			return "";
		}

		return FILETYPE_MAPPINGS[type];
	}

} // namespace wlib::file
#endif
