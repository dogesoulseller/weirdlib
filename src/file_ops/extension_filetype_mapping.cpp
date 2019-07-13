#include "../../include/weirdlib_fileops.hpp"

namespace wlib::file
{
	std::string GetFiletypeExtension(const FileType type) noexcept {
		switch (type)
		{
		case FILETYPE_JPEG:
			return ".jpg";
		case FILETYPE_PNG:
			return ".png";
		case FILETYPE_BMP:
			return ".bmp";
		case FILETYPE_TIFF:
			return ".tiff";
		case FILETYPE_TGA:
			return ".tga";
		case FILETYPE_GIF:
			return ".gif";
		case FILETYPE_PSD:
			return ".psd";
		case FILETYPE_PSB:
			return ".psb";
		case FILETYPE_WEBP:
			return ".webp";
		case FILETYPE_FLIF:
			return ".flif";
		case FILETYPE_PBM:
			return ".pbm";
		case FILETYPE_PGM:
			return ".pgm";
		case FILETYPE_PPM:
			return ".ppm";
		case FILETYPE_PAM:
			return ".pam";
		case FILETYPE_SVG:
			return ".svg";
		case FILETYPE_PDF:
			return ".pdf";
		case FILETYPE_UNKNOWN:
		default:
			return "";
		}
	}
} // namespace wlib::file
