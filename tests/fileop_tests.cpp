#include <gtest/gtest.h>
#include "../include/weirdlib.hpp"
#include <filesystem>
#include <fstream>

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

TEST(WlibFileop, DetectType) {
	EXPECT_EQ(wlib::file::DetectFileType(bmpFilePath), wlib::file::FileType::FILETYPE_BMP);

	EXPECT_EQ(wlib::file::DetectFileType(jpgFilePath), wlib::file::FileType::FILETYPE_JPEG);

	EXPECT_EQ(wlib::file::DetectFileType(pngFilePath), wlib::file::FileType::FILETYPE_PNG);

	EXPECT_EQ(wlib::file::DetectFileType(tiffFilePath), wlib::file::FileType::FILETYPE_TIFF);

	EXPECT_EQ(wlib::file::DetectFileType(tgaFilePath), wlib::file::FileType::FILETYPE_TGA);

	EXPECT_EQ(wlib::file::DetectFileType(gifFilePath), wlib::file::FileType::FILETYPE_GIF);

	EXPECT_EQ(wlib::file::DetectFileType(psdFilePath), wlib::file::FileType::FILETYPE_PSD);
	EXPECT_EQ(wlib::file::DetectFileType(psbFilePath), wlib::file::FileType::FILETYPE_PSB);

	EXPECT_EQ(wlib::file::DetectFileType(webpFilePath), wlib::file::FileType::FILETYPE_WEBP);
	EXPECT_EQ(wlib::file::DetectFileType(aviFilePath), wlib::file::FileType::FILETYPE_AVI);

	EXPECT_EQ(wlib::file::DetectFileType(pbmFilePath), wlib::file::FileType::FILETYPE_PBM);
	EXPECT_EQ(wlib::file::DetectFileType(pgmFilePath), wlib::file::FileType::FILETYPE_PGM);
	EXPECT_EQ(wlib::file::DetectFileType(ppmFilePath), wlib::file::FileType::FILETYPE_PPM);
	EXPECT_EQ(wlib::file::DetectFileType(pamFilePath), wlib::file::FileType::FILETYPE_PAM);

	EXPECT_EQ(wlib::file::DetectFileType(flifFilePath), wlib::file::FileType::FILETYPE_FLIF);
	EXPECT_EQ(wlib::file::DetectFileType(svgFilePath), wlib::file::FileType::FILETYPE_SVG);

	EXPECT_EQ(wlib::file::DetectFileType(pdfFilePath), wlib::file::FileType::FILETYPE_PDF);

	EXPECT_EQ(wlib::file::DetectFileType(mkvFilePath), wlib::file::FileType::FILETYPE_MATROSKA);

	EXPECT_EQ(wlib::file::DetectFileType(mp4FilePath), wlib::file::FileType::FILETYPE_MP4);

	EXPECT_EQ(wlib::file::DetectFileType(flvFilePath), wlib::file::FileType::FILETYPE_FLV);

	EXPECT_EQ(wlib::file::DetectFileType(f4vFilePath), wlib::file::FileType::FILETYPE_F4V);

	EXPECT_EQ(wlib::file::DetectFileType(webmFilePath), wlib::file::FileType::FILETYPE_WEBM);
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
}
