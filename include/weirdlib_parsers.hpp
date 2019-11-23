#pragma once
#include <string>
#include <variant>
#include <utility>
#include <vector>
#include <type_traits>
#include <exception>

#include "./weirdlib_containers.hpp"

namespace wlib::parse
{
	class comfyg_value_get_error : public std::exception
	{
		std::string what_message;

		public:
		inline comfyg_value_get_error(const std::string& msg) noexcept {
			what_message = msg;
		}

		inline comfyg_value_get_error(std::string&& msg) noexcept {
			what_message = std::move(msg);
		}

		explicit inline comfyg_value_get_error(const char* msg) noexcept {
			what_message = std::string(msg);
		}

		[[nodiscard]] inline const char* what() const noexcept override
		{
			return what_message.c_str();
		}
	};

	class cuesheet_invalid_filetype : public std::exception
	{
		std::string what_message;

		public:
		inline cuesheet_invalid_filetype(const std::string& msg) noexcept {
			what_message = msg;
		}

		inline cuesheet_invalid_filetype(std::string&& msg) noexcept {
			what_message = std::move(msg);
		}

		explicit inline cuesheet_invalid_filetype(const char* msg) noexcept {
			what_message = std::string(msg);
		}

		[[nodiscard]] inline const char* what() const noexcept override
		{
			return what_message.c_str();
		}
	};

	enum class ComfygType
	{
		Integer,
		Float,
		Boolean,
		String,
		Comment
	};

	enum class ParseErrorType
	{
		TokenizeFailed,
		InvalidValue,
		InvalidType
	};

	struct CuesheetTrack
	{
		uint8_t idx;
		std::string pregapTimestamp;
		std::string startTimestamp;
		std::string title;
		std::string artist;
	};

	struct CuesheetFile
	{
		std::string path;
		std::string title;
		std::string artist;
		std::vector<CuesheetTrack> tracks;
	};

	using ComfygValue = std::variant<int64_t, double, bool, std::string>;
	using ComfygErrType = std::pair<std::string, ParseErrorType>;

	class Comfyg
	{
		public:
		Comfyg(const std::string& path);
		Comfyg(const uint8_t* ptr, size_t len);

		template<typename GetType>
		std::enable_if_t<std::is_floating_point_v<GetType>,
		GetType> GetVal(const std::string& key) {
			using namespace std::string_literals;
			auto val = values.at(key);

			if (auto value = std::get_if<double>(&val); value != nullptr) {
				return static_cast<GetType>(*value);
			} else {
				throw comfyg_value_get_error("Comfyg Parse: Failed to get floating point value at key "s + key);
			}
		}

		template<typename GetType>
		std::enable_if_t<std::is_integral_v<GetType> &&
		(!std::is_same_v<GetType, bool>),
		GetType> GetVal(const std::string& key) {
			using namespace std::string_literals;
			auto val = values.at(key);

			if (auto value = std::get_if<int64_t>(&val); value != nullptr) {
				return static_cast<GetType>(*value);
			} else {
				throw comfyg_value_get_error("Comfyg Parse: Failed to get integer value at key "s + key);
			}
		}

		template<typename GetType>
		std::enable_if_t<std::is_same_v<GetType, bool>,
		GetType> GetVal(const std::string& key) {
			using namespace std::string_literals;
			auto val = values.at(key);

			if (auto value = std::get_if<bool>(&val); value != nullptr) {
				return *value;
			} else {
				throw comfyg_value_get_error("Comfyg Parse: Failed to get boolean value at key "s + key);
			}
		}

		template<typename GetType>
		std::enable_if_t<std::is_same_v<GetType, std::string>,
		GetType> GetVal(const std::string& key) {
			using namespace std::string_literals;
			auto val = values.at(key);

			if (auto value = std::get_if<std::string>(&val); value != nullptr) {
				return *value;
			} else {
				throw comfyg_value_get_error("Comfyg Parse: Failed to get string value at key "s + key);
			}
		}

		inline const std::vector<ComfygErrType>& GetErrors() const noexcept {
			return errors;
		}

		private:
		void ParseFormat(const uint8_t* ptr, size_t len);

		std::vector<ComfygErrType> errors;
		unordered_flat_map<std::string, ComfygValue> values;
		std::vector<std::pair<std::string, ComfygType>> sortedLines;
	};

	class Cuesheet
	{
		public:
		Cuesheet(const std::string& path);
		Cuesheet(const uint8_t* ptr, size_t len);

		inline const auto& GetContents() const noexcept {
			return contents;
		}

		private:
		void ParseFormat(const uint8_t* ptr, size_t len);

		std::vector<CuesheetFile> contents;
	};

} // namespace wlib::parse
