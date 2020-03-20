#pragma once
#ifdef WEIRDLIB_ENABLE_FILE_OPERATIONS
#include "../../include/weirdlib_fileops.hpp"

#include <fstream>
#include <array>
#include <algorithm>
#include <type_traits>

namespace wlib::file
{
	template<auto SizeIdent, auto SizeSource, typename ArrType>
	constexpr bool MatchIdentifier(const std::array<ArrType, SizeIdent>& identifier, const std::array<ArrType, SizeSource>& source) noexcept {
		return std::equal(identifier.cbegin(), identifier.cend(), source.cbegin());
	}

	template<auto SizeIdent, auto SizeSource, typename ArrT, typename OffsetT, typename = std::enable_if_t<std::is_integral_v<OffsetT>>>
	constexpr bool MatchIdentifier(const std::array<ArrT, SizeIdent>& identifier, const std::array<ArrT, SizeSource>& source, const OffsetT offset) noexcept {
		return std::equal(identifier.cbegin(), identifier.cend(), source.cbegin()+offset);
	}

	inline bool isTar(const std::string& path) noexcept {
		std::ifstream f(path, std::ios::ate | std::ios::binary);
		f.seekg(257);
		constexpr std::array<uint8_t, 5> tarHeader {'u', 's', 't', 'a', 'r'};

		std::array<uint8_t, 5> buffer;
		f.read(reinterpret_cast<char*>(buffer.data()), 5);

		return std::equal(tarHeader.begin(), tarHeader.end(), buffer.data());
	}

	inline bool isXML(const std::string& path) noexcept {
		std::ifstream f(path, std::ios::ate | std::ios::binary);
		f.seekg(0);
		constexpr std::array<uint8_t, 5> xmlHeader {'<', '?', 'x', 'm', 'l'};

		std::array<uint8_t, 512> headerStuff;
		f.read(reinterpret_cast<char*>(headerStuff.data()), 512);
		f.seekg(0);

		return std::search(headerStuff.begin(), headerStuff.end(), std::default_searcher(xmlHeader.begin(), xmlHeader.end())) != headerStuff.end();
	}

	inline bool isSVG(const std::string& path) noexcept {
		std::ifstream f(path, std::ios::ate | std::ios::binary);
		f.seekg(0);
		constexpr std::array<uint8_t, 4> svgStart = {'<', 's', 'v', 'g'};

		std::array<uint8_t, 16384> svgBuffer;

		f.read(reinterpret_cast<char*>(svgBuffer.data()), 16384);

		return std::search(svgBuffer.begin(), svgBuffer.end(), std::boyer_moore_horspool_searcher(svgStart.begin(), svgStart.end())) != svgBuffer.end();
	}

	inline bool isMPEG4(const std::string& path) noexcept {
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

	inline bool isF4V(const std::string& path) noexcept {
		std::ifstream f(path, std::ios::binary);
		constexpr std::array<uint8_t, 7> ident {'f', 't', 'y', 'p', 'f', '4', 'v'};

		std::array<uint8_t, 64> header;
		f.read(reinterpret_cast<char*>(header.data()), 64);

		return std::search(header.begin(), header.end(), std::default_searcher(ident.begin(), ident.end())) != header.end();
	}

	inline bool is3GP(const std::string& path) noexcept {
		std::ifstream f(path, std::ios::binary);
		constexpr std::array<uint8_t, 7> ident {'f', 't', 'y', 'p', '3', 'g', 'p'};

		std::array<uint8_t, 64> header;
		f.read(reinterpret_cast<char*>(header.data()), 64);

		return std::search(header.begin(), header.end(), std::default_searcher(ident.begin(), ident.end())) != header.end();
	}

	inline bool is3G2(const std::string& path) noexcept {
		std::ifstream f(path, std::ios::binary);
		constexpr std::array<uint8_t, 7> ident {'f', 't', 'y', 'p', '3', 'g', '2'};

		std::array<uint8_t, 64> header;
		f.read(reinterpret_cast<char*>(header.data()), 64);

		return std::search(header.begin(), header.end(), std::default_searcher(ident.begin(), ident.end())) != header.end();
	}

	inline bool isAIFF(const std::string& path) noexcept {
		std::ifstream f(path, std::ios::binary);
		constexpr std::array<uint8_t, 4> ident {'F', 'O', 'R', 'M'};
		constexpr std::array<uint8_t, 4> ident_aiff {'A', 'I', 'F', 'F'};

		std::array<uint8_t, 128> header;
		f.read(reinterpret_cast<char*>(header.data()), 128);

		auto isFORM = std::search(header.begin(), header.begin()+4, std::default_searcher(ident.begin(), ident.end())) != header.end();
		return std::search(header.begin(), header.end(), std::default_searcher(ident_aiff.begin(), ident_aiff.end())) != header.end() && isFORM;
	}

	template<typename FileIterT>
	inline bool isTar(const FileIterT dataStart, const FileIterT dataEnd) noexcept {
		constexpr std::array<uint8_t, 5> tarHeader {'u', 's', 't', 'a', 'r'};

		// Magic number is at offset 257
		if (std::distance(dataStart, dataEnd) <= 261) {
			return false;
		}

		if (std::equal(tarHeader.begin(), tarHeader.end(), dataStart+257)) {
			return true;
		}

		return false;
	}

	template<typename FileIterT>
	inline bool isXML(const FileIterT dataStart, const FileIterT dataEnd) noexcept {
		constexpr std::array<uint8_t, 5> xmlHeader {'<', '?', 'x', 'm', 'l'};

		auto headerScanEnd = dataStart + (std::min(std::distance(dataStart, dataEnd), std::ptrdiff_t(512)));
		return std::search(dataStart, headerScanEnd, std::default_searcher(xmlHeader.begin(), xmlHeader.end())) != headerScanEnd;
	}

	template<typename FileIterT>
	inline bool isSVG(const FileIterT dataStart, const FileIterT dataEnd) noexcept {
		constexpr std::array<uint8_t, 4> svgStart = {'<', 's', 'v', 'g'};

		auto svgDataScanEnd = dataStart + (std::min(std::distance(dataStart, dataEnd), std::ptrdiff_t(16384)));

		return std::search(dataStart, svgDataScanEnd, std::boyer_moore_horspool_searcher(svgStart.begin(), svgStart.end())) != svgDataScanEnd;
	}

	template<typename FileIterT>
	inline bool isMPEG4(const FileIterT dataStart, const FileIterT dataEnd) noexcept {
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
	inline bool isF4V(const FileIterT dataStart, const FileIterT dataEnd) noexcept {
		constexpr std::array<uint8_t, 7> ident {'f', 't', 'y', 'p', 'f', '4', 'v'};

		auto searchEnd = dataStart + (std::min(std::distance(dataStart, dataEnd), std::ptrdiff_t(64)));

		return std::search(dataStart, searchEnd, std::default_searcher(ident.begin(), ident.end())) != searchEnd;
	}

	template<typename FileIterT>
	inline bool is3GP(const FileIterT dataStart, const FileIterT dataEnd) noexcept {
		constexpr std::array<uint8_t, 7> ident {'f', 't', 'y', 'p', '3', 'g', 'p'};

		auto searchEnd = dataStart + (std::min(std::distance(dataStart, dataEnd), std::ptrdiff_t(64)));

		return std::search(dataStart, searchEnd, std::default_searcher(ident.begin(), ident.end())) != searchEnd;
	}

	template<typename FileIterT>
	inline bool is3G2(const FileIterT dataStart, const FileIterT dataEnd) noexcept {
		constexpr std::array<uint8_t, 7> ident {'f', 't', 'y', 'p', '3', 'g', '2'};

		auto searchEnd = dataStart + (std::min(std::distance(dataStart, dataEnd), std::ptrdiff_t(64)));

		return std::search(dataStart, searchEnd, std::default_searcher(ident.begin(), ident.end())) != searchEnd;
	}

	template<typename FileIterT>
	inline bool isAIFF(const FileIterT dataStart, const FileIterT dataEnd) noexcept {
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

		// 7-Zip
		detectedType = MatchIdentifier(SEVENZIP_IDENTIFIER, arrayData) ? FILETYPE_7Z : detectedType;
		if (detectedType != FILETYPE_UNKNOWN) return detectedType;

		// RAR
		detectedType = MatchIdentifier(RAR_IDENTIFIER, arrayData) ? FILETYPE_RAR : detectedType;
		if (detectedType != FILETYPE_UNKNOWN) return detectedType;

		// Tar
		detectedType = MatchIdentifier(TAR_IDENTIFIER, arrayData) ? FILETYPE_TAR : detectedType;
		if (detectedType != FILETYPE_UNKNOWN) return detectedType;

		// BZIP2
		detectedType = MatchIdentifier(BZIP2_IDENTIFIER, arrayData) ? FILETYPE_BZIP2 : detectedType;
		if (detectedType != FILETYPE_UNKNOWN) return detectedType;

		// GZIP
		detectedType = MatchIdentifier(GZIP_IDENTIFIER, arrayData) ? FILETYPE_GZIP : detectedType;
		if (detectedType != FILETYPE_UNKNOWN) return detectedType;

		// LZIP
		detectedType = MatchIdentifier(LZIP_IDENTIFIER, arrayData) ? FILETYPE_LZIP : detectedType;
		if (detectedType != FILETYPE_UNKNOWN) return detectedType;

		// ZSTD
		detectedType = MatchIdentifier(ZSTD_IDENTIFIER, arrayData) ? FILETYPE_ZSTD : detectedType;
		if (detectedType != FILETYPE_UNKNOWN) return detectedType;

		// ZSTD
		detectedType = MatchIdentifier(XZ_IDENTIFIER, arrayData) ? FILETYPE_XZ : detectedType;
		if (detectedType != FILETYPE_UNKNOWN) return detectedType;

		return detectedType;
	}
} // namespace wlib::file
#endif
