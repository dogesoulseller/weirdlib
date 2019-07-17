#pragma once
#include <string>

namespace wlib
{

/// Functions for modifying files or gathering information
namespace file
{
	/// Types of detectable files
	enum FileType
	{
		FILETYPE_UNKNOWN = -1,
		FILETYPE_JPEG,
		FILETYPE_PNG,
		FILETYPE_BMP,
		FILETYPE_TIFF,
		FILETYPE_TGA,
		FILETYPE_GIF,
		FILETYPE_PSD,
		FILETYPE_PSB,
		FILETYPE_WEBP,
		FILETYPE_FLIF,
		FILETYPE_PBM,
		FILETYPE_PGM,
		FILETYPE_PPM,
		FILETYPE_PAM,
		FILETYPE_SVG,
		FILETYPE_PDF,
		FILETYPE_MATROSKA,
		FILETYPE_AVI,
		FILETYPE_MP4,
		FILETYPE_FLV,
		FILETYPE_F4V,
		FILETYPE_WEBM
	};

	/// Detect file type based on identifying features
	/// @param path path to the file to detect
	/// @return {@link FileType} enum value
	FileType DetectFileType(const std::string& path);

	/// Get extension (including .) corresponding to file type
	/// @param type file type to get extension for
	/// @return extension
	std::string GetFiletypeExtension(const FileType type) noexcept;

} // namespace file
} // namespace wlib
