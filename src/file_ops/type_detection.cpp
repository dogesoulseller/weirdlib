#include "../../include/weirdlib_fileops.hpp"
#include "../../include/cpu_detection.hpp"

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

	static bool isSVG(const std::string& path) noexcept {
		std::ifstream f(path, std::ios::ate | std::ios::binary);
		f.seekg(0);
		constexpr std::array<uint8_t, 5> xmlHeader {'<', '?', 'x', 'm', 'l'};
		constexpr std::array<uint8_t, 4> svgStart = {'<', 's', 'v', 'g'};

		std::array<uint8_t, 16384> svgBuffer;
		std::array<uint8_t, 512> headerStuff;
		f.read(reinterpret_cast<char*>(headerStuff.data()), 512);
		f.seekg(0);

		if (std::search(headerStuff.begin(), headerStuff.end(), std::default_searcher(xmlHeader.begin(), xmlHeader.end())) == headerStuff.end()) {
			return false;
		}

		f.read(reinterpret_cast<char*>(svgBuffer.data()), 16384);

		return std::search(svgBuffer.begin(), svgBuffer.end(), std::boyer_moore_horspool_searcher(svgStart.begin(), svgStart.end())) != svgBuffer.end();
	}

	static bool isMPEG4(const std::string& path) noexcept {
		std::ifstream f(path, std::ios::binary);
		constexpr std::array<uint8_t, 4> mpeg4_1 {'m', 'p', '4', '1'};
		constexpr std::array<uint8_t, 4> mpeg4_2 {'m', 'p', '4', '2'};
		constexpr std::array<uint8_t, 4> moovatom {'m', 'o', 'o', 'v'};

		std::array<uint8_t, 256> header;
		f.read(reinterpret_cast<char*>(header.data()), 256);

		// Most common format
		const bool tryGeneral = (std::search(header.begin(), header.end(), std::default_searcher(mpeg4_1.begin(), mpeg4_1.end())) != header.end())
			^ (std::search(header.begin(), header.end(), std::default_searcher(mpeg4_2.begin(), mpeg4_2.end())) != header.end());

		// Less common streaming format
		const bool tryMoovAtom = std::search(header.begin(), header.end(), std::default_searcher(moovatom.begin(), moovatom.end())) != header.end();

		return tryGeneral || tryMoovAtom;
	}

	static bool isF4V(const std::string& path) noexcept {
		std::ifstream f(path, std::ios::binary);
		constexpr std::array<uint8_t, 7> ident {'f', 't', 'y', 'p', 'f', '4', 'v'};

		std::array<uint8_t, 64> header;
		f.read(reinterpret_cast<char*>(header.data()), 64);

		return std::search(header.begin(), header.end(), std::default_searcher(ident.begin(), ident.end())) != header.end();
	}

	static bool is3GP(const std::string& path) noexcept {
		std::ifstream f(path, std::ios::binary);
		constexpr std::array<uint8_t, 7> ident {'f', 't', 'y', 'p', '3', 'g', 'p'};

		std::array<uint8_t, 64> header;
		f.read(reinterpret_cast<char*>(header.data()), 64);

		return std::search(header.begin(), header.end(), std::default_searcher(ident.begin(), ident.end())) != header.end();
	}

	static bool is3G2(const std::string& path) noexcept {
		std::ifstream f(path, std::ios::binary);
		constexpr std::array<uint8_t, 7> ident {'f', 't', 'y', 'p', '3', 'g', '2'};

		std::array<uint8_t, 64> header;
		f.read(reinterpret_cast<char*>(header.data()), 64);

		return std::search(header.begin(), header.end(), std::default_searcher(ident.begin(), ident.end())) != header.end();
	}

	static bool isAIFF(const std::string& path) noexcept {
		std::ifstream f(path, std::ios::binary);
		constexpr std::array<uint8_t, 4> ident {'F', 'O', 'R', 'M'};
		constexpr std::array<uint8_t, 4> ident_aiff {'A', 'I', 'F', 'F'};

		std::array<uint8_t, 128> header;
		f.read(reinterpret_cast<char*>(header.data()), 128);

		auto isFORM = std::search(header.begin(), header.begin()+4, std::default_searcher(ident.begin(), ident.end())) != header.end();
		return std::search(header.begin(), header.end(), std::default_searcher(ident_aiff.begin(), ident_aiff.end())) != header.end() && isFORM;
	}

	template<typename FileIterT>
	bool isSVG(const FileIterT dataStart, const FileIterT dataEnd) noexcept {
		constexpr std::array<uint8_t, 5> xmlHeader {'<', '?', 'x', 'm', 'l'};
		constexpr std::array<uint8_t, 4> svgStart = {'<', 's', 'v', 'g'};

		auto headerScanEnd = dataStart + (std::min(std::distance(dataStart, dataEnd), std::ptrdiff_t(512)));
		if (std::search(dataStart, headerScanEnd, std::default_searcher(xmlHeader.begin(), xmlHeader.end())) == headerScanEnd) {
			return false;
		}

		auto svgDataScanEnd = dataStart + (std::min(std::distance(dataStart, dataEnd), std::ptrdiff_t(16384)));

		return std::search(dataStart, svgDataScanEnd, std::boyer_moore_horspool_searcher(svgStart.begin(), svgStart.end())) != svgDataScanEnd;
	}

	template<typename FileIterT>
	bool isMPEG4(const FileIterT dataStart, const FileIterT dataEnd) noexcept {
		constexpr std::array<uint8_t, 4> mpeg4_1 {'m', 'p', '4', '1'};
		constexpr std::array<uint8_t, 4> mpeg4_2 {'m', 'p', '4', '2'};
		constexpr std::array<uint8_t, 4> moovatom {'m', 'o', 'o', 'v'};

		auto searchEnd = dataStart + (std::min(std::distance(dataStart, dataEnd), std::ptrdiff_t(256)));

		// Most common format
		const bool tryGeneral = (std::search(dataStart, searchEnd, std::default_searcher(mpeg4_1.begin(), mpeg4_1.end())) != searchEnd)
			^ (std::search(dataStart, searchEnd, std::default_searcher(mpeg4_2.begin(), mpeg4_2.end())) != searchEnd);

		// Less common streaming format
		const bool tryMoovAtom = std::search(dataStart, searchEnd, std::default_searcher(moovatom.begin(), moovatom.end())) != searchEnd;

		return tryGeneral || tryMoovAtom;
	}

	template<typename FileIterT>
	bool isF4V(const FileIterT dataStart, const FileIterT dataEnd) noexcept {
		constexpr std::array<uint8_t, 7> ident {'f', 't', 'y', 'p', 'f', '4', 'v'};

		auto searchEnd = dataStart + (std::min(std::distance(dataStart, dataEnd), std::ptrdiff_t(64)));

		return std::search(dataStart, searchEnd, std::default_searcher(ident.begin(), ident.end())) != searchEnd;
	}

	template<typename FileIterT>
	bool is3GP(const FileIterT dataStart, const FileIterT dataEnd) noexcept {
		constexpr std::array<uint8_t, 7> ident {'f', 't', 'y', 'p', '3', 'g', 'p'};

		auto searchEnd = dataStart + (std::min(std::distance(dataStart, dataEnd), std::ptrdiff_t(64)));

		return std::search(dataStart, searchEnd, std::default_searcher(ident.begin(), ident.end())) != searchEnd;
	}

	template<typename FileIterT>
	bool is3G2(const FileIterT dataStart, const FileIterT dataEnd) noexcept {
		constexpr std::array<uint8_t, 7> ident {'f', 't', 'y', 'p', '3', 'g', '2'};

		auto searchEnd = dataStart + (std::min(std::distance(dataStart, dataEnd), std::ptrdiff_t(64)));

		return std::search(dataStart, searchEnd, std::default_searcher(ident.begin(), ident.end())) != searchEnd;
	}

	template<typename FileIterT>
	bool isAIFF(const FileIterT dataStart, const FileIterT dataEnd) noexcept {
		constexpr std::array<uint8_t, 4> ident {'F', 'O', 'R', 'M'};
		constexpr std::array<uint8_t, 4> ident_aiff {'A', 'I', 'F', 'F'};

		auto searchEnd = dataStart + (std::min(std::distance(dataStart, dataEnd), std::ptrdiff_t(128)));

		auto isFORM = std::search(dataStart, searchEnd+4, std::default_searcher(ident.begin(), ident.end())) != searchEnd;
		return std::search(dataStart, searchEnd, std::default_searcher(ident_aiff.begin(), ident_aiff.end())) != searchEnd && isFORM;
	}

	template<typename ArrT, auto ArrSize>
	FileType _getSimpleFileType(const std::array<ArrT, ArrSize>& arrayData) noexcept {
		FileType detectedType = FILETYPE_UNKNOWN;

		// BMP
		detectedType = MatchIdentifier(BMP_IDENTIFIER, arrayData) ? FILETYPE_BMP : detectedType;
		if (detectedType != FILETYPE_UNKNOWN) return detectedType;

		// PNG
		detectedType = MatchIdentifier(PNG_IDENTIFIER, arrayData) ? FILETYPE_PNG : detectedType;
		if (detectedType != FILETYPE_UNKNOWN) return detectedType;

		// TIFF
		detectedType = MatchIdentifier(TIFF_BE_IDENTIFIER, arrayData) || MatchIdentifier(TIFF_LE_IDENTIFIER, arrayData) ? FILETYPE_TIFF : detectedType;
		if (detectedType != FILETYPE_UNKNOWN) return detectedType;

		// GIF
		detectedType = MatchIdentifier(GIF_IDENTIFIER_87, arrayData) || MatchIdentifier(GIF_IDENTIFIER_89, arrayData) ? FILETYPE_GIF : detectedType;
		if (detectedType != FILETYPE_UNKNOWN) return detectedType;

		// PSD/PSB
		if (MatchIdentifier(PSD_GENERAL_IDENTIFIER, arrayData)) {
			detectedType = MatchIdentifier(PSD_PSD_IDENTIFIER, arrayData, 4) ? FILETYPE_PSD : detectedType;
			detectedType = MatchIdentifier(PSD_PSB_IDENTIFIER, arrayData, 4) ? FILETYPE_PSB : detectedType;
			if (detectedType != FILETYPE_UNKNOWN) return detectedType;
		}

		// --------------------------- RIFF-based -------------------------- //
		if (MatchIdentifier(RIFF_IDENTIFIER, arrayData)) {
			// WebP
			detectedType = MatchIdentifier(WEBP_IDENTIFIER, arrayData, 8) ? FILETYPE_WEBP : detectedType;
			if (detectedType != FILETYPE_UNKNOWN) return detectedType;

			// AVI
			detectedType = MatchIdentifier(AVI_IDENTIFIER, arrayData, 8) ? FILETYPE_AVI : detectedType;
			if (detectedType != FILETYPE_UNKNOWN) return detectedType;

			// WAVE
			detectedType = MatchIdentifier(WAVE_IDENTIFIER, arrayData, 8) ? FILETYPE_WAVE : detectedType;
			if (detectedType != FILETYPE_UNKNOWN) return detectedType;
		}

		// FLIF
		detectedType = MatchIdentifier(FLIF_IDENTIFIER, arrayData) ? FILETYPE_FLIF : detectedType;
		if (detectedType != FILETYPE_UNKNOWN) return detectedType;

		// PDF
		detectedType = MatchIdentifier(PDF_IDENTIFIER, arrayData) ? FILETYPE_PDF : detectedType;
		if (detectedType != FILETYPE_UNKNOWN) return detectedType;

		// Matroska
		detectedType = MatchIdentifier(MATROSKA_IDENTIFIER, arrayData, 0x18) || MatchIdentifier(MATROSKA_IDENTIFIER, arrayData, 0x1F) ? FILETYPE_MATROSKA : detectedType;
		if (detectedType != FILETYPE_UNKNOWN) return detectedType;

		// FLV
		detectedType = MatchIdentifier(FLV_IDENTIFIER, arrayData) ? FILETYPE_FLV : detectedType;
		if (detectedType != FILETYPE_UNKNOWN) return detectedType;

		// WebM
		detectedType = MatchIdentifier(WEBM_IDENTIFIER, arrayData, 0x18) || MatchIdentifier(WEBM_IDENTIFIER, arrayData, 0x1F) ? FILETYPE_WEBM : detectedType;
		if (detectedType != FILETYPE_UNKNOWN) return detectedType;

		// Ogg
		detectedType = MatchIdentifier(OGG_IDENTIFIER, arrayData) ? FILETYPE_OGG : detectedType;
		if (detectedType != FILETYPE_UNKNOWN) return detectedType;

		// APE
		detectedType = MatchIdentifier(APE_IDENTIFIER, arrayData) ? FILETYPE_APE : detectedType;
		if (detectedType != FILETYPE_UNKNOWN) return detectedType;

		// TTA
		detectedType = MatchIdentifier(TTA1_IDENTIFIER, arrayData) || MatchIdentifier(TTA2_IDENTIFIER, arrayData) ? FILETYPE_TTA : detectedType;
		if (detectedType != FILETYPE_UNKNOWN) return detectedType;

		// WavPack
		detectedType = MatchIdentifier(WAVPACK_IDENTIFIER, arrayData) ? FILETYPE_WAVPACK : detectedType;
		if (detectedType != FILETYPE_UNKNOWN) return detectedType;

		// FLAC
		detectedType = MatchIdentifier(FLAC_IDENTIFIER, arrayData) ? FILETYPE_FLAC : detectedType;
		if (detectedType != FILETYPE_UNKNOWN) return detectedType;

		// Core Audio
		detectedType = MatchIdentifier(CAF_IDENTIFIER, arrayData) ? FILETYPE_CAF : detectedType;
		if (detectedType != FILETYPE_UNKNOWN) return detectedType;

		// OptimFROG
		detectedType = MatchIdentifier(OPTIMFROG_IDENTIFIER, arrayData) ? FILETYPE_OPTIMFROG : detectedType;
		if (detectedType != FILETYPE_UNKNOWN) return detectedType;

		// PNM / PAM - keep last, error prone
		detectedType = MatchIdentifier(PBM_ASCII_IDENTIFIER, arrayData) || MatchIdentifier(PBM_BIN_IDENTIFIER, arrayData) ? FILETYPE_PBM : detectedType;
		detectedType = MatchIdentifier(PGM_ASCII_IDENTIFIER, arrayData) || MatchIdentifier(PGM_BIN_IDENTIFIER, arrayData) ? FILETYPE_PGM : detectedType;
		detectedType = MatchIdentifier(PPM_ASCII_IDENTIFIER, arrayData) || MatchIdentifier(PPM_BIN_IDENTIFIER, arrayData) ? FILETYPE_PPM : detectedType;
		detectedType = MatchIdentifier(PAM_IDENTIFIER, arrayData) ? FILETYPE_PAM : detectedType;
		if (detectedType != FILETYPE_UNKNOWN) return detectedType;

		return detectedType;
	}

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

		// SVG - keep last, expensive
		detectedType = isSVG(path) ? FILETYPE_SVG : detectedType;
		if (detectedType != FILETYPE_UNKNOWN) return detectedType;

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

		// SVG - keep last, expensive
		detectedType = isSVG(fileData, fileData+size) ? FILETYPE_SVG : detectedType;
		if (detectedType != FILETYPE_UNKNOWN) return detectedType;

		return detectedType;
	}

} // namespace wlib::file
