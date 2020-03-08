#ifdef WEIRDLIB_ENABLE_FILE_PARSERS
#if !defined(WEIRDLIB_ENABLE_STRING_OPERATIONS)
	#error "File parsers module requires the string operations module"
#endif

#include <filesystem>
#include <fstream>
#include <vector>
#include <array>
#include <string>
#include <algorithm>
#include <optional>
#include <string_view>

#include "../../include/weirdlib_parsers.hpp"
#include "../../include/weirdlib_string.hpp"
#include "../common.hpp"

using namespace std::string_literals;

namespace detail
{
	enum class FieldType
	{
		ARTIST,
		TITLE,
		PREGAP_START,
		TRACK_START,
		FILE_PATH,
		TRACK_ID
	};

	inline constexpr static const char* FieldToSearch(FieldType field) {
		switch (field)
		{
		case FieldType::ARTIST:
			return "PERFORMER";
		case FieldType::TITLE:
			return "TITLE";
		case FieldType::PREGAP_START:
			return "INDEX 00";
		case FieldType::TRACK_START:
			return "INDEX 01";
		case FieldType::FILE_PATH:
			return "FILE";
		case FieldType::TRACK_ID:
			return "TRACK";
		default:
			return "";
		}
	}

	template<FieldType field>
	inline constexpr static const char* FieldToSearch() {
		if constexpr (field == FieldType::ARTIST) {
			return "PERFORMER";
		} else if constexpr (field == FieldType::TITLE) {
			return "TITLE";
		} else if constexpr (field == FieldType::PREGAP_START) {
			return "INDEX 00";
		} else if constexpr (field == FieldType::TRACK_START) {
			return "INDEX 01";
		} else if constexpr (field == FieldType::FILE_PATH) {
			return "FILE";
		} else if constexpr (field == FieldType::TRACK_ID) {
			return "TRACK";
		} else {
			return "";
		}
	}


	enum class SectionName
	{
		FILE,
		TRACK
	};

	template<SectionName section>
	static inline const char* SectionSearchString() {
		if constexpr (section == SectionName::FILE) {
			return "FILE";
		} else {
			return "TRACK";
		}
	}

	constexpr std::array<uint8_t, 3> UTF8_BOM_SEQ = {0xEF, 0xBB, 0xBF};
	constexpr std::array<uint8_t, 4> UTF32_BOM_SEQ = {0xFF, 0xFE, 0x00, 0x00};
	constexpr std::array<uint8_t, 2> UTF16_BOM_SEQ = {0xFF, 0xFE};

	// Currently only fixes the UTF-8 BOM
	// Detects and throws on UTF-16 and UTF-32 (BOM)
	static inline void FixEncoding(const uint8_t*& in, size_t& len) {
		if (std::equal(in, in+3, UTF8_BOM_SEQ.cbegin())) { // UTF-8 with BOM - correctable
			in += 3;
			len -= 3;
		} else if (std::equal(in, in+2, UTF16_BOM_SEQ.cbegin()) || std::equal(in, in+2, UTF16_BOM_SEQ.crbegin())) { // UTF-16 LE/BE uncorrectable
			throw wlib::parse::file_encoding_error("Cuesheets must be encoded as UTF-8, but file was UTF-16");
		} else if (std::equal(in, in+4, UTF32_BOM_SEQ.cbegin()) || std::equal(in, in+4, UTF32_BOM_SEQ.crbegin())) { // UTF-32 LE/BE uncorrectable
			throw wlib::parse::file_encoding_error("Cuesheets must be encoded as UTF-8, but file was UTF-32");
		} else {
			return;
		}
	}

	template<SectionName section>
	static inline auto SplitIntoSections(const std::vector<std::string>& lines) {
		std::vector<std::vector<std::string>> out;

		size_t firstFile = std::distance(lines.begin(), std::find_if(lines.begin(), lines.end(),
			[](auto l){return wlib::str::StartsWith(l, SectionSearchString<section>());})
		);

		// Split into chunks of FILEs
		std::vector<std::string> tmpCurrent;
		for (size_t i = firstFile; i < lines.size(); i++) {
			if (wlib::str::StartsWith(lines[i], SectionSearchString<section>())) {
				out.push_back(tmpCurrent);
				tmpCurrent.clear();
				tmpCurrent.shrink_to_fit();
			}

			tmpCurrent.push_back(lines[i]);
		}
		out.push_back(tmpCurrent);

		std::vector<std::vector<std::string>> newOut(out.begin()+1, out.end());

		return newOut;
	}

	static inline std::optional<std::string> GetQuotedValue(const std::string& str) {
		auto first = str.find_first_of('"');
		auto last  = str.find_last_of('"');

		if (first == last || first == std::string::npos || last == std::string::npos) {
			return std::nullopt;
		}

		return str.substr(first+1, last-first-1);
	}

	static inline std::string GetUnquotedValue(const std::string& str, FieldType field) {
		using namespace std::string_view_literals;
		std::string_view strView(str);

		if (field == FieldType::FILE_PATH) {
			if (wlib::str::EndsWith(strView, "MP3"sv)) {
				strView.remove_suffix(3);
			} else if (wlib::str::EndsWith(strView, "AIFF"sv) || wlib::str::EndsWith(strView, "WAVE"sv)) {
				strView.remove_suffix(4);
			} else {
				throw wlib::parse::cuesheet_format_error("FILE type must be MP3, AIFF, or WAVE");
			}
		} else if (field == FieldType::TRACK_ID) {
			if (wlib::str::EndsWith(strView, "AUDIO"sv)) {
				strView.remove_suffix(5);
			} else {
				throw wlib::parse::cuesheet_format_error("Only AUDIO mode is supported");
			}
		}

		strView.remove_prefix(wlib::str::strlen(FieldToSearch(field)));

		size_t toRemoveFront = 0;
		size_t toRemoveBack = 0;

		// Leading whitespace
		for (auto i = strView.begin(); i != strView.end(); ++i) {
			if (std::isspace(*i)) {
				toRemoveFront++;
			} else {
				break;
			}
		}

		// Trailing whitespace
		for (auto i = strView.rbegin(); i != strView.rend(); ++i) {
			if (std::isspace(*i)) {
				toRemoveBack++;
			} else {
				break;
			}
		}

		strView.remove_prefix(toRemoveFront);
		strView.remove_suffix(toRemoveBack);

		return std::string(strView);
	}


	// TODO: Parse other fields

	template<FieldType field>
	static inline std::optional<std::string> GetField(const std::vector<std::string>& lines) {
		static_assert(field == FieldType::ARTIST || field == FieldType::FILE_PATH || field == FieldType::PREGAP_START
			|| field == FieldType::TITLE || field == FieldType::TRACK_START || field == FieldType::TRACK_ID
		);

		std::string fieldLine;
		for (const auto& l : lines) {
			if (wlib::str::StartsWith(l, FieldToSearch(field))) {
				fieldLine = l;
				break;
			}
		}

		if (fieldLine.empty()) {
			return std::nullopt;
		} else {
			if constexpr (field == FieldType::ARTIST || field == FieldType::TITLE || field == FieldType::FILE_PATH) {
				return GetQuotedValue(fieldLine).value_or(GetUnquotedValue(fieldLine, field));
			} else if constexpr (field == FieldType::PREGAP_START || field == FieldType::TRACK_START || field == FieldType::TRACK_ID) {
				return GetUnquotedValue(fieldLine, field);
			} else {
				return std::nullopt;
			}
		}
	}

} // namespace detail

namespace wlib::parse
{
	Cuesheet::Cuesheet(const std::string& path) {
		std::vector<uint8_t> fileBytes;
		std::ifstream file(path, std::ios::binary | std::ios::ate);
		if (!file.good()) {
			throw std::runtime_error("Cuesheet loader: Failed to load file "s + path);
		}

		auto fileSize = file.tellg();
		fileBytes.resize(fileSize);
		file.seekg(0);
		file.read(reinterpret_cast<char*>(fileBytes.data()), fileSize);

		ParseFormat(fileBytes.data(), fileBytes.size());
	}

	Cuesheet::Cuesheet(const uint8_t* ptr, size_t len) {
		ParseFormat(ptr, len);
	}

	void Cuesheet::ParseFormat(const uint8_t* ptr, size_t len) {
		detail::FixEncoding(ptr, len);

		std::string fileContents(reinterpret_cast<const char*>(ptr), reinterpret_cast<const char*>(ptr)+len);

		auto fileLines = wlib::str::ToLines(fileContents);
		for (auto& l : fileLines) {
			wlib::str::RemoveLeadingWhitespace(l);

			// Fix non-standard CRLF line ending
			if (l[l.size()-1] == '\r') {
				l.pop_back();
			}
		}

		// Remove remarks
		fileLines.erase(std::remove_if(fileLines.begin(), fileLines.end(),
			[](auto l){return wlib::str::StartsWith(l, "REM");}), fileLines.end()
		);

		// Get global metadata
		std::vector<std::string> metadataLines(fileLines.begin(), std::find_if(fileLines.begin(), fileLines.end(),
			[](const auto l) {return wlib::str::StartsWith(l, "FILE");}));

		auto albumArtist = detail::GetField<detail::FieldType::ARTIST>(metadataLines).value_or("");
		auto albumTitle = detail::GetField<detail::FieldType::TITLE>(metadataLines).value_or("");

		// Split into sections and parse
		auto fileSections = detail::SplitIntoSections<detail::SectionName::FILE>(fileLines);
		for (auto& fileSection : fileSections) {
			CuesheetFile cueFileInfo;
			auto trackSections = detail::SplitIntoSections<detail::SectionName::TRACK>(fileSection);

			cueFileInfo.artist = albumArtist;
			cueFileInfo.title  = albumTitle;

			// File path is a required field
			if (auto path = detail::GetField<detail::FieldType::FILE_PATH>(fileSection); path.has_value()) {
				cueFileInfo.path = path.value();
			} else {
				throw cuesheet_format_error("No valid path found for FILE");
			}

			for (auto& section : trackSections) {
				CuesheetTrack cueTrackInfo;

				// Track index is a required field
				if (auto idx = detail::GetField<detail::FieldType::TRACK_ID>(section); idx.has_value()) {
					if (!wlib::str::ParseString(idx.value(), cueTrackInfo.idx)) {
						throw cuesheet_format_error("Failed to parse track index");
					}
				} else {
					throw cuesheet_format_error("No track index found in track");
				}

				cueTrackInfo.artist = detail::GetField<detail::FieldType::ARTIST>(section).value_or("");
				cueTrackInfo.title = detail::GetField<detail::FieldType::TITLE>(section).value_or("");
				cueTrackInfo.pregapTimestamp = detail::GetField<detail::FieldType::PREGAP_START>(section).value_or("");

				// Start timestamp is a required field
				if (auto startTimestamp = detail::GetField<detail::FieldType::TRACK_START>(section); startTimestamp.has_value()) {
					cueTrackInfo.startTimestamp = startTimestamp.value();
				} else {
					throw cuesheet_format_error("No start timestamp found in track");
				}

				cueFileInfo.tracks.push_back(std::move(cueTrackInfo));
			}

			contents.push_back(std::move(cueFileInfo));
		}

	}
} // namespace wlib::parse

#endif
