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
		inline unsupported_image_type(StrT&& msg) noexcept {
			what_message = msg;
		}

		explicit inline unsupported_image_type(const char* msg) noexcept {
			what_message = std::move(std::string(msg));
		}

		inline const char* what() const noexcept
		{
			return what_message.c_str();
		}
	};

	class invalid_image_data : public image_load_exception
	{
		std::string what_message;

		public:
		template<typename StrT>
		inline invalid_image_data(StrT&& msg) noexcept {
			what_message = msg;
		}

		explicit inline invalid_image_data(const char* msg) noexcept {
			what_message = std::move(std::string(msg));
		}

		inline const char* what() const noexcept
		{
			return what_message.c_str();
		}
	};

} // namespace wlib::image::except
