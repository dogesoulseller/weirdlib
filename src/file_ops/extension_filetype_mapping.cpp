#ifdef WEIRDLIB_ENABLE_FILE_OPERATIONS
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
		  case FILETYPE_XML:
		  	return ".xml";
		  case FILETYPE_SVG:
			return ".svg";
		  case FILETYPE_PDF:
			return ".pdf";
		  case FILETYPE_MATROSKA:
			return ".mkv";
		  case FILETYPE_AVI:
			return ".avi";
		  case FILETYPE_MP4:
			return ".mp4";
		  case FILETYPE_FLV:
			return ".flv";
		  case FILETYPE_F4V:
			return ".f4v";
		  case FILETYPE_WEBM:
			return ".webm";
		  case FILETYPE_WAVE:
			return ".wav";
		  case FILETYPE_OGG:
			return ".ogg";
		  case FILETYPE_APE:
			return ".ape";
		  case FILETYPE_TTA:
			return ".tta";
		  case FILETYPE_WAVPACK:
			return ".wv";
		  case FILETYPE_FLAC:
			return ".flac";
		  case FILETYPE_CAF:
			return ".caf";
		  case FILETYPE_OPTIMFROG:
			return ".ofr";
		  case FILETYPE_3GP:
			return ".3gp";
		  case FILETYPE_3G2:
			return ".3g2";
		  case FILETYPE_AIFF:
			return ".aiff";
		  case FILETYPE_UNKNOWN:
		  default:
			return "";
		}
	}

} // namespace wlib::file
#endif
