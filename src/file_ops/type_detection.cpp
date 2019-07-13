#include "../../include/weirdlib_fileops.hpp"
#include "magic_numbers.hpp"

#include <string_view>
#include <filesystem>
#include <fstream>
#include <unordered_map>
#include <vector>
#include <array>
#include <algorithm>
#include <type_traits>

namespace wlib::file
{
	template<auto SizeIdent, auto SizeSource, typename ArrType>
	constexpr auto MatchIdentifier(const std::array<ArrType, SizeIdent>& identifier, const std::array<ArrType, SizeSource>& source) -> bool {
		return std::equal(identifier.cbegin(), identifier.cend(), source.cbegin());
	}

	template<auto SizeIdent, auto SizeSource, typename ArrT, typename OffsetT, typename = std::enable_if_t<std::is_integral_v<OffsetT>>>
	constexpr auto MatchIdentifier(const std::array<ArrT, SizeIdent>& identifier, const std::array<ArrT, SizeSource>& source, const OffsetT offset) -> bool {
		return std::equal(identifier.cbegin(), identifier.cend(), source.cbegin()+offset);
	}

	static bool isSVG(const std::string& path) {
		std::ifstream f(path, std::ios::ate | std::ios::binary);
		f.seekg(0);
		const char* xmlHeader = "<?xml";
		const char* svgStart = "<svg";

		std::array<uint8_t, 16384> svgBuffer;
		std::array<uint8_t, 512> headerStuff;
		f.read(reinterpret_cast<char*>(headerStuff.data()), 512);
		f.seekg(0);
		std::string headerStr(reinterpret_cast<char*>(headerStuff.data()));

		if (headerStr.find(xmlHeader) == std::string::npos) return false;

		f.read(reinterpret_cast<char*>(svgBuffer.data()), 16384);
		std::string svgStr(reinterpret_cast<char*>(svgBuffer.data()));

		return svgStr.find(svgStart) != std::string::npos;
	}

	FileType DetectFileType(const std::string& path) {
		std::ifstream f(path, std::ios::ate | std::ios::binary);

		FileType detectedType = FILETYPE_UNKNOWN;

		// size_t fileSize = f.tellg();
		f.seekg(0);

		std::array<uint8_t, 16> headerTemp;
		f.read(reinterpret_cast<char*>(headerTemp.data()), 16);

		// BMP
		detectedType = MatchIdentifier(BMP_IDENTIFIER, headerTemp) ? FILETYPE_BMP : detectedType;
		if (detectedType != FILETYPE_UNKNOWN) return detectedType;

		// PNG
		detectedType = MatchIdentifier(PNG_IDENTIFIER, headerTemp) ? FILETYPE_PNG : detectedType;
		if (detectedType != FILETYPE_UNKNOWN) return detectedType;

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

		// TIFF
		detectedType = MatchIdentifier(TIFF_BE_IDENTIFIER, headerTemp) || MatchIdentifier(TIFF_LE_IDENTIFIER, headerTemp) ? FILETYPE_TIFF : detectedType;
		if (detectedType != FILETYPE_UNKNOWN) return detectedType;

		// TGA
		{
			std::array<uint8_t, 16> tgaFooterTemp;
			f.seekg(-18, std::ios::end);
			f.read(reinterpret_cast<char*>(tgaFooterTemp.data()), 16);
			detectedType = MatchIdentifier(TGA_IDENTIFIER, tgaFooterTemp) ? FILETYPE_TGA : detectedType;
			if (detectedType != FILETYPE_UNKNOWN) return detectedType;
		}

		// GIF
		detectedType = MatchIdentifier(GIF_IDENTIFIER_87, headerTemp) || MatchIdentifier(GIF_IDENTIFIER_89, headerTemp) ? FILETYPE_GIF : detectedType;
		if (detectedType != FILETYPE_UNKNOWN) return detectedType;

		// PSD/PSB
		if (MatchIdentifier(PSD_GENERAL_IDENTIFIER, headerTemp)) {
			detectedType = MatchIdentifier(PSD_PSD_IDENTIFIER, headerTemp, 4) ? FILETYPE_PSD : detectedType;
			detectedType = MatchIdentifier(PSD_PSB_IDENTIFIER, headerTemp, 4) ? FILETYPE_PSB : detectedType;
			if (detectedType != FILETYPE_UNKNOWN) return detectedType;
		}

		// --------------------------- RIFF-based -------------------------- //
		if (MatchIdentifier(RIFF_IDENTIFIER, headerTemp)) {
			// WebP
			detectedType = MatchIdentifier(WEBP_IDENTIFIER, headerTemp, 8) ? FILETYPE_WEBP : detectedType;
			if (detectedType != FILETYPE_UNKNOWN) return detectedType;
		}

		// FLIF
		detectedType = MatchIdentifier(FLIF_IDENTIFIER, headerTemp) ? FILETYPE_FLIF : detectedType;
		if (detectedType != FILETYPE_UNKNOWN) return detectedType;

		// PDF
		detectedType = MatchIdentifier(PDF_IDENTIFIER, headerTemp) ? FILETYPE_PDF : detectedType;
		if (detectedType != FILETYPE_UNKNOWN) return detectedType;

		// PNM / PAM - keep last, error prone
		detectedType = MatchIdentifier(PBM_ASCII_IDENTIFIER, headerTemp) || MatchIdentifier(PBM_BIN_IDENTIFIER, headerTemp) ? FILETYPE_PBM : detectedType;
		detectedType = MatchIdentifier(PGM_ASCII_IDENTIFIER, headerTemp) || MatchIdentifier(PGM_BIN_IDENTIFIER, headerTemp) ? FILETYPE_PGM : detectedType;
		detectedType = MatchIdentifier(PPM_ASCII_IDENTIFIER, headerTemp) || MatchIdentifier(PPM_BIN_IDENTIFIER, headerTemp) ? FILETYPE_PPM : detectedType;
		detectedType = MatchIdentifier(PAM_IDENTIFIER, headerTemp) ? FILETYPE_PAM : detectedType;
		if (detectedType != FILETYPE_UNKNOWN) return detectedType;

		// SVG
		detectedType = isSVG(path) ? FILETYPE_SVG : detectedType;
		if (detectedType != FILETYPE_UNKNOWN) return detectedType;

		return detectedType;
	}

} // namespace wlib::file
