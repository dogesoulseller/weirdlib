#pragma once
#ifdef WEIRDLIB_ENABLE_FILE_OPERATIONS
#include <array>
#include <cstdint>

namespace wlib::file
{
	constexpr std::array<uint8_t, 2> BMP_IDENTIFIER {'B', 'M'};

	constexpr std::array<uint8_t, 3> JPEG_SOI_IDENTIFIER {0xFF, 0xD8, 0xFF};
	constexpr std::array<uint8_t, 2> JPEG_EOI_IDENTIFIER {0xFF, 0xD9};

	constexpr std::array<uint8_t, 8> PNG_IDENTIFIER {0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A};

	constexpr std::array<uint8_t, 4> TIFF_LE_IDENTIFIER {'I', 'I', 42, 0x00};
	constexpr std::array<uint8_t, 4> TIFF_BE_IDENTIFIER {'M', 'M', 0x00, 42};

	constexpr std::array<uint8_t, 16> TGA_IDENTIFIER {'T', 'R', 'U', 'E', 'V', 'I', 'S', 'I', 'O', 'N', '-', 'X', 'F', 'I', 'L', 'E'};

	constexpr std::array<uint8_t, 4> PDF_IDENTIFIER {'%', 'P', 'D', 'F'};

	constexpr std::array<uint8_t, 6> GIF_IDENTIFIER_87 {'G', 'I', 'F', '8', '7', 'a'};
	constexpr std::array<uint8_t, 6> GIF_IDENTIFIER_89 {'G', 'I', 'F', '8', '9', 'a'};

	constexpr std::array<uint8_t, 4> PSD_GENERAL_IDENTIFIER {'8', 'B', 'P', 'S'};
	constexpr std::array<uint8_t, 2> PSD_PSD_IDENTIFIER {0, 1};
	constexpr std::array<uint8_t, 2> PSD_PSB_IDENTIFIER {0, 2};

	constexpr std::array<uint8_t, 4> RIFF_IDENTIFIER {'R', 'I', 'F', 'F'};
	constexpr std::array<uint8_t, 4> WEBP_IDENTIFIER {'W', 'E', 'B', 'P'};
	constexpr std::array<uint8_t, 3> AVI_IDENTIFIER {'A', 'V', 'I'};
	constexpr std::array<uint8_t, 4> WAVE_IDENTIFIER {'W', 'A', 'V', 'E'};

	constexpr std::array<uint8_t, 2> PBM_ASCII_IDENTIFIER {'P', '1'};
	constexpr std::array<uint8_t, 2> PBM_BIN_IDENTIFIER {'P', '4'};

	constexpr std::array<uint8_t, 2> PGM_ASCII_IDENTIFIER {'P', '2'};
	constexpr std::array<uint8_t, 2> PGM_BIN_IDENTIFIER {'P', '5'};

	constexpr std::array<uint8_t, 2> PPM_ASCII_IDENTIFIER {'P', '3'};
	constexpr std::array<uint8_t, 2> PPM_BIN_IDENTIFIER {'P', '6'};

	constexpr std::array<uint8_t, 2> PAM_IDENTIFIER {'P', '7'};

	constexpr std::array<uint8_t, 4> FLIF_IDENTIFIER {'F', 'L', 'I', 'F'};

	constexpr std::array<uint8_t, 8> MATROSKA_IDENTIFIER {'m', 'a', 't', 'r', 'o', 's', 'k', 'a'};

	constexpr std::array<uint8_t, 3> FLV_IDENTIFIER {'F', 'L', 'V'};

	constexpr std::array<uint8_t, 4> WEBM_IDENTIFIER {'w', 'e', 'b', 'm'};

	constexpr std::array<uint8_t, 4> OGG_IDENTIFIER {'O', 'g', 'g', 'S'};

	constexpr std::array<uint8_t, 3> APE_IDENTIFIER {'M', 'A', 'C'};

	constexpr std::array<uint8_t, 4> TTA1_IDENTIFIER {'T', 'T', 'A', '1'};
	constexpr std::array<uint8_t, 4> TTA2_IDENTIFIER {'T', 'T', 'A', '2'};

	constexpr std::array<uint8_t, 4> WAVPACK_IDENTIFIER {'w', 'v', 'p', 'k'};

	constexpr std::array<uint8_t, 4> FLAC_IDENTIFIER {'f', 'L', 'a', 'C'};

	constexpr std::array<uint8_t, 4> CAF_IDENTIFIER {'c', 'a', 'f', 'f'};

	constexpr std::array<uint8_t, 3> OPTIMFROG_IDENTIFIER {'O', 'F', 'R'};

	constexpr std::array<uint8_t, 6> SEVENZIP_IDENTIFIER {'7', 'z', 0xBC, 0xAF, 0x27, 0x1C};
	constexpr std::array<uint8_t, 6> RAR_IDENTIFIER {0x52, 0x61, 0x72, 0x21, 0x1A, 0x07};
	constexpr std::array<uint8_t, 5> TAR_IDENTIFIER {'u', 's', 't', 'a', 'r'};
	constexpr std::array<uint8_t, 3> BZIP2_IDENTIFIER {'B', 'Z', 'h'};
	constexpr std::array<uint8_t, 3> GZIP_IDENTIFIER {0x1F, 0x8B, 0x08};
	constexpr std::array<uint8_t, 4> LZIP_IDENTIFIER {'L', 'Z', 'I', 'P'};
	constexpr std::array<uint8_t, 4> ZSTD_IDENTIFIER {0x28, 0xB5, 0x2F, 0xFD};
	constexpr std::array<uint8_t, 7> XZ_IDENTIFIER {0xFD, 0x37, 0x7A, 0x58, 0x5A, 0x00, 0x00};

} // namespace wlib::file
#endif
