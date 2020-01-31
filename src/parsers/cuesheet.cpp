#ifdef WEIRDLIB_ENABLE_FILE_PARSERS
#if !defined(WEIRDLIB_ENABLE_STRING_OPERATIONS)
	#error "File parsers module requires the string operations module"
#endif

#include "../../include/weirdlib_parsers.hpp"
#include "../../include/weirdlib_traits.hpp"
#include "../../include/weirdlib_string.hpp"
#include "../common.hpp"

#include <cstring>
#include <algorithm>
#include <fstream>
#include <vector>
#include <string>
#include <string_view>
#include <utility>

#include "parser_common.hpp"

using namespace std::string_literals;

namespace wlib::parse
{
	static std::pair<const char*, bool> detectFiletype(const std::string& cmd) {
		if (auto pos = wlib::str::strstr(cmd, "BINARY"); pos != nullptr) {
			return std::make_pair(pos, false);
		}

		if (auto pos = wlib::str::strstr(cmd, "MOTOROLA"); pos != nullptr) {
			return std::make_pair(pos, false);
		}

		if (auto pos = wlib::str::strstr(cmd, "WAVE"); pos != nullptr) {
			return std::make_pair(pos, true);
		}

		if (auto pos = wlib::str::strstr(cmd, "MP3"); pos != nullptr) {
			return std::make_pair(pos, true);
		}

		if (auto pos = wlib::str::strstr(cmd, "AIFF"); pos != nullptr) {
			return std::make_pair(pos, true);
		}

		throw cuesheet_invalid_filetype("Cuesheet loader: no valid filetype detected");
	}

	static std::vector<std::string> splitIntoLines(const uint8_t* ptr, size_t len) {
		std::vector<std::string> lines;

		// Split into lines
		size_t offset = 0;
		bool isEOL = false;
		auto oldNewlinePos = ptr;

		auto newlinePos = std::find(ptr+offset, ptr+len, '\n');
		if (newlinePos == ptr+len) {
			isEOL = true;
		}

		lines.emplace_back(oldNewlinePos, newlinePos);
		oldNewlinePos = newlinePos;
		offset = newlinePos - ptr + 1;

		while(!isEOL) {
			newlinePos = std::find(ptr+offset, ptr+len, '\n');
			if (newlinePos == ptr+len) {
				isEOL = true;
			}

			lines.emplace_back(oldNewlinePos+1, newlinePos);
			oldNewlinePos = newlinePos;
			offset = newlinePos - ptr + 1;
		}

		return lines;
	}

	static std::pair<std::vector<std::vector<std::string>>, std::vector<std::vector<std::string>>> splitByFILECommands(std::vector<std::string>& fileLines) {
		// Split sheet by FILE commands
		auto findNextFILE = [&fileLines](size_t sLine) -> int {
			for (size_t i = sLine; i < fileLines.size(); i++) {
				auto commPos = std::strstr(fileLines[i].c_str(), "FILE ");
				if (commPos == nullptr) {
					continue;
				}

				return i;
			}

			return -1;
		};

		size_t currentLine = 0;
		std::vector<std::pair<size_t, size_t>> fileBounds;
		while(true) {
			// Detect first file
			auto startLine = findNextFILE(currentLine);
			if (startLine == -1) {
				break;
			}

			// If only one command remains, bounds are until the EOF
			auto endLine = findNextFILE(startLine+1);
			if (endLine == -1) {
				fileBounds.emplace_back(startLine, fileLines.size());
				break;
			}

			fileBounds.emplace_back(startLine, endLine);
			currentLine = endLine;

		}

		if (fileBounds.empty()) {
			throw std::runtime_error("Cuesheet loader: no FILE commands");
		}

		// Split off FILE metadata
		// Start with a special case for the first FILE
		std::vector<std::vector<std::string>> linesPerFileMeta;
		{
			std::vector<std::string> tmp(fileLines.data(), fileLines.data()+fileBounds[0].first);
			linesPerFileMeta.push_back(std::move(tmp));
		}

		for (size_t i = 0; i < fileBounds.size()-1; i++) {
			// If there are no metadata lines
			if (fileBounds[i].second+1 > fileBounds[i+1].first) {
				linesPerFileMeta.emplace_back();
				continue;
			}

			std::vector<std::string> tmp(fileLines.data()+fileBounds[i].second+1,
				fileLines.data()+fileBounds[i+1].first);

			linesPerFileMeta.push_back(std::move(tmp));
		}

		// Split to per-FILE lines
		std::vector<std::vector<std::string>> linesPerFile;
		linesPerFile.reserve(fileBounds.size());
		for (const auto& c: fileBounds) {
			linesPerFile.emplace_back(fileLines.data()+c.first, fileLines.data()+c.second);
		}

		return std::make_pair(linesPerFile, linesPerFileMeta);
	}

	static std::vector<std::vector<std::string>> splitByTRACKCommands(std::vector<std::string>& fileCommandLines) {
		// Split sheet by TRACK commands
		auto findNextTRACK = [&fileCommandLines](size_t sLine) -> int {
			for (size_t i = sLine; i < fileCommandLines.size(); i++) {
					auto commPos = wlib::str::strstr(fileCommandLines[i], "TRACK", fileCommandLines[i].size(), 5);
					if (commPos == nullptr) {
						continue;
					}

					return i;
				}

				return -1;
		};

		size_t currentFileLine = 1;
		std::vector<std::pair<size_t, size_t>> trackBounds;

		while(true) {
			// Detect first track
			auto startLine = findNextTRACK(currentFileLine);
			if (startLine == -1) {
				break;
			}

			// If only one command remains, bounds are until the EOF
			auto endLine = findNextTRACK(startLine+1);
			if (endLine == -1) {
				trackBounds.emplace_back(startLine, fileCommandLines.size());
				break;
			}

			trackBounds.emplace_back(startLine, endLine);
			currentFileLine = endLine;
		}

		std::vector<std::vector<std::string>> linesPerTrack;
		linesPerTrack.reserve(trackBounds.size());
		for (const auto& c: trackBounds) {
			std::vector<std::string> tmp(fileCommandLines.data()+c.first, fileCommandLines.data()+c.second);
			std::for_each(tmp.begin(), tmp.end(), common::RemoveLeadingWhitespace);
			linesPerTrack.push_back(std::move(tmp));
		}

		return linesPerTrack;
	}

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
		auto lines = splitIntoLines(ptr, len);

		// Remove empty lines
		lines.erase(std::remove_if(lines.begin(), lines.end(), [](const std::string& s){
			return s.empty() || std::all_of(s.begin(), s.end(), isspace);
		}), lines.end());

		std::for_each(lines.begin(), lines.end(), common::RemoveLeadingWhitespace);

		// Remove CR from line endings
		std::for_each(lines.begin(), lines.end(), [](auto& line){
			line.erase(std::remove(line.begin(), line.end(), '\r'), line.end());
		});

		// Parse FILEs
		auto [linesPerFILE, linesPerFILEMetadata] = splitByFILECommands(lines);
		for (auto& currentFILELines: linesPerFILE) {
			CuesheetFile fileInfo;
			bool isAudio;

			// Get filetype and file name
			if (auto pathStart = currentFILELines[0].find_first_of('"'); pathStart == std::string::npos) {
				common::RemoveAllOccurences(currentFILELines[0], "FILE ");

				auto [typePos, isAudio_tmp] = detectFiletype(currentFILELines[0]);

				fileInfo.path = std::string(currentFILELines[0].c_str(), typePos-1);
				isAudio = isAudio_tmp;
			} else {
				auto pathEnd = currentFILELines[0].find_last_of('"');
				auto [IGNORESB, isAudio_tmp] = detectFiletype(currentFILELines[0]);

				fileInfo.path = currentFILELines[0].substr(pathStart+1, pathEnd-pathStart-1);
				isAudio = isAudio_tmp;
			}

			auto linesPerTRACK = splitByTRACKCommands(currentFILELines);

			if (isAudio) {
				fileInfo.tracks.reserve(linesPerTRACK.size());

				// Process all tracks inside FILE
				for (const auto& trackLines: linesPerTRACK) {
					CuesheetTrack trackInfo;

					std::string_view trackHeader(trackLines[0]);

					// Remove 'TRACK ' and type
					trackHeader.remove_prefix(6);
					trackHeader.remove_suffix(trackHeader.size() - 2);

					// Parse index
					std::from_chars(trackHeader.begin(), trackHeader.end(), trackInfo.idx);

					// Search lines for additional info
					for (const auto& line: trackLines) {
						auto findCommand = [&line](const std::string& needle){
							return wlib::str::strstr(line, needle);
						};

						auto titlePos = findCommand("TITLE");
						if (titlePos != nullptr) {
							if (auto titleStart = line.find_first_of('"'); titleStart == std::string::npos) {
								auto tmpLine = line;
								common::RemoveAllOccurences(tmpLine, "TITLE ");
								trackInfo.title = tmpLine;
							} else {
								auto titleEnd = line.find_last_of('"');
								trackInfo.title = line.substr(titleStart+1, titleEnd-titleStart-1);
							}

							continue;
						}

						auto artistPos = findCommand("PERFORMER");
						if (artistPos != nullptr) {
							if (auto artistStart = line.find_first_of('"'); artistStart == std::string::npos) {
								auto tmpLine = line;
								common::RemoveAllOccurences(tmpLine, "PERFORMER ");
								trackInfo.artist = tmpLine;
							} else {
								auto artistEnd = line.find_last_of('"');
								trackInfo.artist = line.substr(artistStart+1, artistEnd-artistStart-1);
							}

							continue;
						}

						auto pregapPos = findCommand("INDEX 00");
						if (pregapPos != nullptr) {
							trackInfo.pregapTimestamp = line.substr(9);
							continue;
						}

						auto idx1Pos = findCommand("INDEX 01");
						if (idx1Pos != nullptr) {
							trackInfo.startTimestamp = line.substr(9);
							continue;
						}
					}
					fileInfo.tracks.push_back(trackInfo);
				}

				contents.push_back(fileInfo);
			} else {
				//TODO:
				for (const auto& trackLines: linesPerTRACK) {

				}
			}
		}

		for (size_t i = 0; i < linesPerFILEMetadata.size(); i++) {
			// Parse metadata
			for (const auto& line: linesPerFILEMetadata[i]) {
				auto findCommand = [&line](const std::string& needle){
					return wlib::str::strstr(line, needle);
				};

				// TODO: Additional metadata

				auto titlePos = findCommand("TITLE");
				if (titlePos != nullptr) {
					if (auto titleStart = line.find_first_of('"'); titleStart == std::string::npos) {
						auto tmpLine = line;
						common::RemoveAllOccurences(tmpLine, "TITLE ");
						contents[i].title = tmpLine;
					} else {
						auto titleEnd = line.find_last_of('"');
						contents[i].title = line.substr(titleStart+1, titleEnd-titleStart-1);
					}

					continue;
				}

				auto artistPos = findCommand("PERFORMER");
				if (artistPos != nullptr) {
					if (auto artistStart = line.find_first_of('"'); artistStart == std::string::npos) {
						auto tmpLine = line;
						common::RemoveAllOccurences(tmpLine, "PERFORMER ");
						contents[i].artist = tmpLine;
					} else {
						auto artistEnd = line.find_last_of('"');
						contents[i].artist = line.substr(artistStart+1, artistEnd-artistStart-1);
					}

					continue;
				}
			}

			// Update missing metadata
			for (auto& track: contents[i].tracks) {
				// Update artist to album artist if no artist is given
				if (track.artist.empty()) {
					track.artist = contents[i].artist;
				}
			}
		}

	}
} // namespace wlib::parse
#endif
