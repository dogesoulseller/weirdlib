#include <weirdlib_fileops.hpp>
#include <string>
#include <filesystem>
#include <vector>
#include <iostream>
#include <functional>
#include <unordered_map>
#include <algorithm>
#include <string_view>
#include <charconv>
#include <tuple>

#include "messages.hpp"
#include "param_process.hpp"

static auto _getFileStem_Name_CorrExt(const std::filesystem::path& path) noexcept {
	return std::tuple<std::string, std::string, std::string>(path.stem().string(),
		path.filename().string(),
		wlib::file::GetFiletypeExtension(wlib::file::DetectFileType(path)));
}

int main(int argc, char* argv[]) {
	if (argc < 2) {
		std::cerr << "Not enough arguments passed" << "\n";
		std::cout << helpMessage << std::endl;
		std::exit(1);
	}

	std::vector<std::string> arguments;
	arguments.reserve(static_cast<size_t>(argc));
	for (auto i = 0; i < argc; i++) {
		arguments.emplace_back(std::string(argv[i]));
	}

	// Check for path existence
	if (!std::filesystem::exists(arguments[1])) {
		std::cerr << "Input file/dir does not exist" << '\n';
		std::exit(3);
	}

	int64_t _recdepth;
	const auto Parameters = GetParameters(arguments);
	std::filesystem::path inputPath = arguments[1];
	std::string _recdepthAsString = Parameters.at(RECURSION_DEPTH);
	std::from_chars(_recdepthAsString.data(), _recdepthAsString.data()+_recdepthAsString.length(), _recdepth, 10);

	const auto maxRecursionDepth = static_cast<size_t>(_recdepth);

	// Check if input is file or directory
	if (std::filesystem::is_regular_file(inputPath)) {	// File
		const auto [filenameStem, oldFilename, newExtension] = _getFileStem_Name_CorrExt(inputPath);
		const auto inputPathAsString = inputPath.string();

		// Absolute base output directory
		// If is absolute, use it without modification
		// If not absolute, strip filename from source path and append suffix from parameter
		const auto outputDir = std::filesystem::path(Parameters.at(OUTPUT_DIR)).is_absolute()
			? Parameters.at(OUTPUT_DIR)
			: inputPathAsString.substr(0, inputPathAsString.length() - oldFilename.length()) + "/" + Parameters.at(OUTPUT_DIR);

		// Make sure dirs exist
		std::filesystem::create_directories(outputDir);

		try {
			const auto newFilename = outputDir + "/" + filenameStem + newExtension;
			std::filesystem::copy_file(inputPath, newFilename);
		} catch (std::exception&) {
			const auto newFilename = outputDir + "/" + filenameStem + "_other" + newExtension;
			std::filesystem::copy_file(inputPath, newFilename);
		}

	} else if (std::filesystem::is_directory(inputPath)) {	// Directory
		inputPath = std::filesystem::canonical(inputPath);

		std::filesystem::recursive_directory_iterator dirIter(inputPath, std::filesystem::directory_options::skip_permission_denied);
		for (const auto& f : dirIter) {
			// Prevent crossing over max iteration depth
			if (static_cast<size_t>(dirIter.depth()) > maxRecursionDepth) {
				dirIter.pop();
			}

			// Skip non-files
			if (!std::filesystem::is_regular_file(f)) {
				continue;
			}

			// Simplify getting file path
			const auto currentFilePath = std::filesystem::canonical(f.path());
			const auto currentFilePath_str = currentFilePath.string();
			std::string_view currentFilePath_sv(currentFilePath_str);

			// Process file
			const auto [filenameStem, oldFilename, newExtension] = _getFileStem_Name_CorrExt(currentFilePath);

			// Absolute base output directory
			// If is absolute, append recursed tree
			// If not absolute, strip filename from source path and append suffix from parameter
			std::string outputDir;
			if (std::filesystem::path(Parameters.at(OUTPUT_DIR)).is_absolute()) {
				// Difference from base path
				currentFilePath_sv.remove_prefix(inputPath.string().length());
				currentFilePath_sv.remove_suffix(oldFilename.length());

				outputDir = Parameters.at(OUTPUT_DIR) + "/" + std::string(currentFilePath_sv);
			} else {
				outputDir = currentFilePath_str.substr(0, currentFilePath_str.length() - oldFilename.length()) + "/" + Parameters.at(OUTPUT_DIR);
			}

			// Create full tree of dirs
			std::filesystem::create_directories(outputDir);

			// Work around nested exception handling
			[&, filenameStem = std::cref(filenameStem), newExtension = std::cref(newExtension)]() -> void {
				try {
					const auto newFilename = outputDir + "/" + filenameStem.get() + newExtension.get();
					std::filesystem::copy_file(f.path(), newFilename);
				} catch (std::exception&) {
					try {
						const auto newFilename = outputDir + "/" + filenameStem.get() + "_other" + newExtension.get();
						std::filesystem::copy_file(f.path(), newFilename);
					} catch (std::exception&) {
						return;
					}
				}
			}();
		}

		std::exit(-1);
	} else {	// Neither
		std::cerr << "Input must be a regular file or directory" << '\n';
		std::exit(3);
	}

	return 0;
}
