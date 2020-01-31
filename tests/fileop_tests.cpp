#ifdef WEIRDLIB_ENABLE_FILE_OPERATIONS
#include <gtest/gtest.h>
#include "../include/weirdlib.hpp"
#include <filesystem>
#include <fstream>
#include <vector>
#include <utility>

const std::filesystem::path fileOpDir = std::filesystem::path(WLIBTEST_TESTING_DIRECTORY) / std::filesystem::path("fileop_files");

const auto bmpFilePath = fileOpDir / "bmp.bmp";
const auto jpgFilePath = fileOpDir / "jpg.jpg";
const auto pngFilePath = fileOpDir / "png.png";
const auto tiffFilePath = fileOpDir / "tiff.tiff";
const auto tgaFilePath = fileOpDir / "tga.tga";
const auto pdfFilePath = fileOpDir / "pdf.pdf";
const auto gifFilePath = fileOpDir / "gif.gif";
const auto psdFilePath = fileOpDir / "psd.psd";
const auto psbFilePath = fileOpDir / "psb.psb";
const auto pbmFilePath = fileOpDir / "pbm.pbm";
const auto pgmFilePath = fileOpDir / "pgm.pgm";
const auto ppmFilePath = fileOpDir / "ppm.ppm";
const auto pamFilePath = fileOpDir / "pam.pam";
const auto webpFilePath = fileOpDir / "webp.webp";
const auto flifFilePath = fileOpDir / "flif.flif";
const auto svgFilePath = fileOpDir / "svg.svg";
const auto mkvFilePath = fileOpDir / "mkv.mkv";
const auto aviFilePath = fileOpDir / "avi.avi";
const auto mp4FilePath = fileOpDir / "mp4.mp4";
const auto flvFilePath = fileOpDir / "flv.flv";
const auto f4vFilePath = fileOpDir / "f4v.f4v";
const auto webmFilePath = fileOpDir / "webm.webm";
const auto waveFilePath = fileOpDir / "wav.wav";
const auto oggFilePath = fileOpDir / "ogg.ogg";
const auto apeFilePath = fileOpDir / "ape.ape";
const auto ttaFilePath = fileOpDir / "tta.tta";
const auto wvpackFilePath = fileOpDir / "wv.wv";
const auto flacFilePath = fileOpDir / "flac.flac";
const auto cafFilePath = fileOpDir / "caf.caf";
const auto ofrFilePath = fileOpDir / "ofr.ofr";
const auto _3gpFilePath = fileOpDir / "3gp.3gp";
const auto _3g2FilePath = fileOpDir / "3g2.3g2";
const auto aiffFilePath = fileOpDir / "aiff.aiff";
const auto randomFilePath = fileOpDir / "random.random";

const std::vector<std::pair<std::filesystem::path, wlib::file::FileType>> FileMappings {
	std::pair<std::filesystem::path, wlib::file::FileType>(bmpFilePath, wlib::file::FileType::FILETYPE_BMP),
	std::pair<std::filesystem::path, wlib::file::FileType>(jpgFilePath, wlib::file::FileType::FILETYPE_JPEG),
	std::pair<std::filesystem::path, wlib::file::FileType>(pngFilePath, wlib::file::FileType::FILETYPE_PNG),
	std::pair<std::filesystem::path, wlib::file::FileType>(tiffFilePath, wlib::file::FileType::FILETYPE_TIFF),
	std::pair<std::filesystem::path, wlib::file::FileType>(tgaFilePath, wlib::file::FileType::FILETYPE_TGA),
	std::pair<std::filesystem::path, wlib::file::FileType>(gifFilePath, wlib::file::FileType::FILETYPE_GIF),
	std::pair<std::filesystem::path, wlib::file::FileType>(psdFilePath, wlib::file::FileType::FILETYPE_PSD),
	std::pair<std::filesystem::path, wlib::file::FileType>(psbFilePath, wlib::file::FileType::FILETYPE_PSB),
	std::pair<std::filesystem::path, wlib::file::FileType>(webpFilePath, wlib::file::FileType::FILETYPE_WEBP),
	std::pair<std::filesystem::path, wlib::file::FileType>(aviFilePath, wlib::file::FileType::FILETYPE_AVI),
	std::pair<std::filesystem::path, wlib::file::FileType>(waveFilePath, wlib::file::FileType::FILETYPE_WAVE),
	std::pair<std::filesystem::path, wlib::file::FileType>(pbmFilePath, wlib::file::FileType::FILETYPE_PBM),
	std::pair<std::filesystem::path, wlib::file::FileType>(pgmFilePath, wlib::file::FileType::FILETYPE_PGM),
	std::pair<std::filesystem::path, wlib::file::FileType>(ppmFilePath, wlib::file::FileType::FILETYPE_PPM),
	std::pair<std::filesystem::path, wlib::file::FileType>(pamFilePath, wlib::file::FileType::FILETYPE_PAM),
	std::pair<std::filesystem::path, wlib::file::FileType>(flifFilePath, wlib::file::FileType::FILETYPE_FLIF),
	std::pair<std::filesystem::path, wlib::file::FileType>(svgFilePath, wlib::file::FileType::FILETYPE_SVG),
	std::pair<std::filesystem::path, wlib::file::FileType>(pdfFilePath, wlib::file::FileType::FILETYPE_PDF),
	std::pair<std::filesystem::path, wlib::file::FileType>(mkvFilePath, wlib::file::FileType::FILETYPE_MATROSKA),
	std::pair<std::filesystem::path, wlib::file::FileType>(mp4FilePath, wlib::file::FileType::FILETYPE_MP4),
	std::pair<std::filesystem::path, wlib::file::FileType>(flvFilePath, wlib::file::FileType::FILETYPE_FLV),
	std::pair<std::filesystem::path, wlib::file::FileType>(f4vFilePath, wlib::file::FileType::FILETYPE_F4V),
	std::pair<std::filesystem::path, wlib::file::FileType>(webmFilePath, wlib::file::FileType::FILETYPE_WEBM),
	std::pair<std::filesystem::path, wlib::file::FileType>(oggFilePath, wlib::file::FileType::FILETYPE_OGG),
	std::pair<std::filesystem::path, wlib::file::FileType>(apeFilePath, wlib::file::FileType::FILETYPE_APE),
	std::pair<std::filesystem::path, wlib::file::FileType>(ttaFilePath, wlib::file::FileType::FILETYPE_TTA),
	std::pair<std::filesystem::path, wlib::file::FileType>(wvpackFilePath, wlib::file::FileType::FILETYPE_WAVPACK),
	std::pair<std::filesystem::path, wlib::file::FileType>(flacFilePath, wlib::file::FileType::FILETYPE_FLAC),
	std::pair<std::filesystem::path, wlib::file::FileType>(cafFilePath, wlib::file::FileType::FILETYPE_CAF),
	std::pair<std::filesystem::path, wlib::file::FileType>(ofrFilePath, wlib::file::FileType::FILETYPE_OPTIMFROG),
	std::pair<std::filesystem::path, wlib::file::FileType>(_3gpFilePath, wlib::file::FileType::FILETYPE_3GP),
	std::pair<std::filesystem::path, wlib::file::FileType>(_3g2FilePath, wlib::file::FileType::FILETYPE_3G2),
	std::pair<std::filesystem::path, wlib::file::FileType>(aiffFilePath, wlib::file::FileType::FILETYPE_AIFF),


	std::pair<std::filesystem::path, wlib::file::FileType>(randomFilePath, wlib::file::FileType::FILETYPE_UNKNOWN)
};

TEST(WlibFileop, DetectTypeFStream) {
	for (const auto& [fPath, fType] : FileMappings) {
		EXPECT_EQ(wlib::file::DetectFileType(fPath), fType) << "Expected type of file at " << fPath << " to be equal to enum value " << fType;
	}
}

TEST(WlibFileop, DetectTypeMemory) {
	for (const auto& [fPath, fType] : FileMappings) {
		std::vector<uint8_t> fileBytes;

		ASSERT_TRUE(std::filesystem::exists(fPath)) << fPath << " does not exist";

		std::ifstream f(fPath, std::ios::binary | std::ios::ate);
		ASSERT_TRUE(f.good()) << "Failed to open file " << fPath;

		auto fileSize = f.tellg();
		ASSERT_GT(fileSize, 0) << "File at " << fPath << " is empty";
		f.seekg(0);

		fileBytes.resize(static_cast<size_t>(fileSize));
		f.read(reinterpret_cast<char*>(fileBytes.data()), fileSize);

		EXPECT_EQ(wlib::file::DetectFileType(fileBytes.data(), static_cast<size_t>(fileSize)), fType) << "Expected type of file at " << fPath << " to be equal to enum value " << fType;
	}
}

TEST(WlibFileop, FiletypeExtension) {
	EXPECT_EQ(wlib::file::GetFiletypeExtension(wlib::file::FileType::FILETYPE_BMP), ".bmp");

	EXPECT_EQ(wlib::file::GetFiletypeExtension(wlib::file::FileType::FILETYPE_JPEG), ".jpg");

	EXPECT_EQ(wlib::file::GetFiletypeExtension(wlib::file::FileType::FILETYPE_PNG), ".png");

	EXPECT_EQ(wlib::file::GetFiletypeExtension(wlib::file::FileType::FILETYPE_TIFF), ".tiff");

	EXPECT_EQ(wlib::file::GetFiletypeExtension(wlib::file::FileType::FILETYPE_TGA), ".tga");

	EXPECT_EQ(wlib::file::GetFiletypeExtension(wlib::file::FileType::FILETYPE_GIF), ".gif");

	EXPECT_EQ(wlib::file::GetFiletypeExtension(wlib::file::FileType::FILETYPE_PSD), ".psd");
	EXPECT_EQ(wlib::file::GetFiletypeExtension(wlib::file::FileType::FILETYPE_PSB), ".psb");

	EXPECT_EQ(wlib::file::GetFiletypeExtension(wlib::file::FileType::FILETYPE_WEBP), ".webp");
	EXPECT_EQ(wlib::file::GetFiletypeExtension(wlib::file::FileType::FILETYPE_AVI), ".avi");
	EXPECT_EQ(wlib::file::GetFiletypeExtension(wlib::file::FileType::FILETYPE_WAVE), ".wav");

	EXPECT_EQ(wlib::file::GetFiletypeExtension(wlib::file::FileType::FILETYPE_PBM), ".pbm");
	EXPECT_EQ(wlib::file::GetFiletypeExtension(wlib::file::FileType::FILETYPE_PGM), ".pgm");
	EXPECT_EQ(wlib::file::GetFiletypeExtension(wlib::file::FileType::FILETYPE_PPM), ".ppm");
	EXPECT_EQ(wlib::file::GetFiletypeExtension(wlib::file::FileType::FILETYPE_PAM), ".pam");

	EXPECT_EQ(wlib::file::GetFiletypeExtension(wlib::file::FileType::FILETYPE_FLIF), ".flif");
	EXPECT_EQ(wlib::file::GetFiletypeExtension(wlib::file::FileType::FILETYPE_SVG), ".svg");

	EXPECT_EQ(wlib::file::GetFiletypeExtension(wlib::file::FileType::FILETYPE_PDF), ".pdf");

	EXPECT_EQ(wlib::file::GetFiletypeExtension(wlib::file::FileType::FILETYPE_MATROSKA), ".mkv");

	EXPECT_EQ(wlib::file::GetFiletypeExtension(wlib::file::FileType::FILETYPE_MP4), ".mp4");

	EXPECT_EQ(wlib::file::GetFiletypeExtension(wlib::file::FileType::FILETYPE_FLV), ".flv");

	EXPECT_EQ(wlib::file::GetFiletypeExtension(wlib::file::FileType::FILETYPE_F4V), ".f4v");

	EXPECT_EQ(wlib::file::GetFiletypeExtension(wlib::file::FileType::FILETYPE_WEBM), ".webm");

	EXPECT_EQ(wlib::file::GetFiletypeExtension(wlib::file::FileType::FILETYPE_OGG), ".ogg");

	EXPECT_EQ(wlib::file::GetFiletypeExtension(wlib::file::FileType::FILETYPE_APE), ".ape");

	EXPECT_EQ(wlib::file::GetFiletypeExtension(wlib::file::FileType::FILETYPE_TTA), ".tta");

	EXPECT_EQ(wlib::file::GetFiletypeExtension(wlib::file::FileType::FILETYPE_WAVPACK), ".wv");

	EXPECT_EQ(wlib::file::GetFiletypeExtension(wlib::file::FileType::FILETYPE_FLAC), ".flac");

	EXPECT_EQ(wlib::file::GetFiletypeExtension(wlib::file::FileType::FILETYPE_CAF), ".caf");

	EXPECT_EQ(wlib::file::GetFiletypeExtension(wlib::file::FileType::FILETYPE_OPTIMFROG), ".ofr");

	EXPECT_EQ(wlib::file::GetFiletypeExtension(wlib::file::FileType::FILETYPE_3GP), ".3gp");

	EXPECT_EQ(wlib::file::GetFiletypeExtension(wlib::file::FileType::FILETYPE_3G2), ".3g2");

	EXPECT_EQ(wlib::file::GetFiletypeExtension(wlib::file::FileType::FILETYPE_AIFF), ".aiff");

	EXPECT_EQ(wlib::file::GetFiletypeExtension(wlib::file::FileType::FILETYPE_UNKNOWN), "");
}
#endif
