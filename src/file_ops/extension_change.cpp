#include "../../include/weirdlib_fileops.hpp"
#include <filesystem>

namespace wlib::file
{
	std::string ChangeExtensionToMatchType(const std::string& _path) noexcept {
		std::filesystem::path path(_path);

		FileType fileT = DetectFileType(path);
		std::string fileTExt = GetFiletypeExtension(fileT);

		if (fileT == FileType::FILETYPE_UNKNOWN) {
			return _path;
		} else if (auto oldExtension = path.extension(); oldExtension == fileTExt) {
			return _path;
		}

		return path.replace_extension(fileTExt);
	}

} // namespace wlib::file
