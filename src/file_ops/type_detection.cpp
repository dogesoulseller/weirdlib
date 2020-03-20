#ifdef WEIRDLIB_ENABLE_FILE_OPERATIONS
#include "../../include/weirdlib_fileops.hpp"
#include "../../include/cpu_detection.hpp"

#include "magic_numbers.hpp"
#include "type_detection_detail.hpp"

#include <fstream>
#include <vector>
#include <array>

namespace wlib::file
{
	FileType DetectFileType(const std::string& path) {
		std::ifstream f(path, std::ios::ate | std::ios::binary);

		FileType detectedType = FILETYPE_UNKNOWN;

		// size_t fileSize = f.tellg();
		f.seekg(0);

		alignas(64) std::array<uint8_t, 64> headerTemp;
		f.read(reinterpret_cast<char*>(headerTemp.data()), 64);

		// Wait for sync
		f.sync();

		#if defined(WLIB_ENABLE_PREFETCH)
			_mm_prefetch(headerTemp.data(), _MM_HINT_T0);
		#endif

		// JPG
		{
			std::array<uint8_t, 2> eoiTemp;
			f.seekg(-2, std::ios::end);
			f.read(reinterpret_cast<char*>(eoiTemp.data()), 2);
			bool hasSOI = MatchIdentifier(JPEG_SOI_IDENTIFIER, headerTemp);
			bool hasEOI = MatchIdentifier(JPEG_EOI_IDENTIFIER, eoiTemp);
			detectedType = hasSOI && hasEOI ? FILETYPE_JPEG : detectedType;
			if (detectedType != FILETYPE_UNKNOWN) return detectedType;
		}

		// TGA
		{
			std::array<uint8_t, 16> tgaFooterTemp;
			f.seekg(-18, std::ios::end);
			f.read(reinterpret_cast<char*>(tgaFooterTemp.data()), 16);
			detectedType = MatchIdentifier(TGA_IDENTIFIER, tgaFooterTemp) ? FILETYPE_TGA : detectedType;
			if (detectedType != FILETYPE_UNKNOWN) return detectedType;
		}

		// Easily identified types
		detectedType = _getSimpleFileType(headerTemp);
		if (detectedType != FILETYPE_UNKNOWN) return detectedType;

		// Tar
		detectedType = isTar(path) ? FILETYPE_TAR : detectedType;
		if (detectedType != FILETYPE_UNKNOWN) return detectedType;

		// AIFF - keep last, expensive
		detectedType = isAIFF(path) ? FILETYPE_AIFF : detectedType;
		if (detectedType != FILETYPE_UNKNOWN) return detectedType;

		// 3GP - keep last, expensive
		detectedType = is3GP(path) ? FILETYPE_3GP : detectedType;
		if (detectedType != FILETYPE_UNKNOWN) return detectedType;

		// 3G2 - keep last, expensive
		detectedType = is3G2(path) ? FILETYPE_3G2 : detectedType;
		if (detectedType != FILETYPE_UNKNOWN) return detectedType;

		// F4V - keep last, expensive
		detectedType = isF4V(path) ? FILETYPE_F4V : detectedType;
		if (detectedType != FILETYPE_UNKNOWN) return detectedType;

		// MPEG-4 - keep last, expensive
		detectedType = isMPEG4(path) ? FILETYPE_MP4 : detectedType;
		if (detectedType != FILETYPE_UNKNOWN) return detectedType;

		// XML and XML-based
		detectedType = isXML(path) ? FILETYPE_XML : detectedType;
		if (detectedType == FILETYPE_XML) {
			// SVG
			detectedType = isSVG(path) ? FILETYPE_SVG : detectedType;
			if (detectedType != FILETYPE_XML) return detectedType;

			return detectedType;
		}

		return detectedType;
	}

	FileType DetectFileType(const uint8_t* fileData, const size_t size) {
		FileType detectedType = FILETYPE_UNKNOWN;

		alignas(64) std::array<uint8_t, 64> headerTemp;
		std::copy(fileData, fileData+std::min(size_t(64u), size), headerTemp.data());

		#if defined(WLIB_ENABLE_PREFETCH)
			_mm_prefetch(headerTemp.data(), _MM_HINT_T0);
		#endif

		// TGA
		{
			std::array<uint8_t, 18> tgaFooterTemp;
			std::copy(fileData+size-18, fileData+size, tgaFooterTemp.data());
			detectedType = MatchIdentifier(TGA_IDENTIFIER, tgaFooterTemp) ? FILETYPE_TGA : detectedType;
			if (detectedType != FILETYPE_UNKNOWN) return detectedType;
		}

		// JPG
		{
			std::array<uint8_t, 2> eoiTemp;
			std::copy(fileData+size-2, fileData+size, eoiTemp.data());
			bool hasSOI = MatchIdentifier(JPEG_SOI_IDENTIFIER, headerTemp);
			bool hasEOI = MatchIdentifier(JPEG_EOI_IDENTIFIER, eoiTemp);
			detectedType = hasSOI && hasEOI ? FILETYPE_JPEG : detectedType;
			if (detectedType != FILETYPE_UNKNOWN) return detectedType;
		}

		// Easily identified types
		detectedType = _getSimpleFileType(headerTemp);
		if (detectedType != FILETYPE_UNKNOWN) return detectedType;

		// Tar
		detectedType = isTar(fileData, fileData+size) ? FILETYPE_TAR : detectedType;
		if (detectedType != FILETYPE_UNKNOWN) return detectedType;

		// AIFF - keep last, expensive
		detectedType = isAIFF(fileData, fileData+size) ? FILETYPE_AIFF : detectedType;
		if (detectedType != FILETYPE_UNKNOWN) return detectedType;

		// 3GP - keep last, expensive
		detectedType = is3GP(fileData, fileData+size) ? FILETYPE_3GP : detectedType;
		if (detectedType != FILETYPE_UNKNOWN) return detectedType;

		// 3G2 - keep last, expensive
		detectedType = is3G2(fileData, fileData+size) ? FILETYPE_3G2 : detectedType;
		if (detectedType != FILETYPE_UNKNOWN) return detectedType;

		// F4V - keep last, expensive
		detectedType = isF4V(fileData, fileData+size) ? FILETYPE_F4V : detectedType;
		if (detectedType != FILETYPE_UNKNOWN) return detectedType;

		// MPEG-4 - keep last, expensive
		detectedType = isMPEG4(fileData, fileData+size) ? FILETYPE_MP4 : detectedType;
		if (detectedType != FILETYPE_UNKNOWN) return detectedType;

		// XML and XML-based
		detectedType = isXML(fileData, fileData+size) ? FILETYPE_XML : detectedType;
		if (detectedType == FILETYPE_XML) {
			// SVG
			detectedType = isSVG(fileData, fileData+size) ? FILETYPE_SVG : detectedType;
			if (detectedType != FILETYPE_XML) return detectedType;

			return detectedType;
		}

		return detectedType;
	}

} // namespace wlib::file
#endif
