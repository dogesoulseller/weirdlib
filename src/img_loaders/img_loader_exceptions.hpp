#pragma once
#include <stdexcept>
#include <exception>

namespace wlib::image::except
{
	class image_load_exception : public std::exception
	{
	};

	class unsupported_image_type : public image_load_exception
	{
		std::string what_message;

		public:
		template<typename StrT>
		inline unsupported_image_type(const std::string& msg) noexcept {
			what_message = msg;
		}

		explicit inline unsupported_image_type(const char* msg) noexcept {
			what_message = std::move(std::string(msg));
		}

		inline unsupported_image_type(std::string&& msg) noexcept {
			what_message = std::move(msg);
		}

		[[nodiscard]] inline const char* what() const noexcept override
		{
			return what_message.c_str();
		}
	};

	class invalid_image_data : public image_load_exception
	{
		std::string what_message;

		public:
		inline invalid_image_data(const std::string& msg) noexcept {
			what_message = msg;
		}

		inline invalid_image_data(std::string&& msg) noexcept {
			what_message = std::move(msg);
		}

		explicit inline invalid_image_data(const char* msg) noexcept {
			what_message = std::move(std::string(msg));
		}

		[[nodiscard]] inline const char* what() const noexcept override
		{
			return what_message.c_str();
		}
	};

	class file_open_error : public image_load_exception
	{
		std::string what_message;

		public:
		inline file_open_error(const std::string& msg) noexcept {
			what_message = msg;
		}

		inline file_open_error(std::string&& msg) noexcept {
			what_message = std::move(msg);
		}

		explicit inline file_open_error(const char* msg) noexcept {
			what_message = std::move(std::string(msg));
		}

		[[nodiscard]] inline const char* what() const noexcept override
		{
			return what_message.c_str();
		}
	};

} // namespace wlib::image::except
