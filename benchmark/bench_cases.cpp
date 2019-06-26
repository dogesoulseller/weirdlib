#include "bench_cases.hpp"
#include <sstream>
#include <iomanip>
#include <iostream>

#include "../include/weirdlib.hpp"
#include "bench_common.hpp"

constexpr size_t BENCHMARK_ITERATIONS_STRING = 250;
constexpr size_t BENCHMARK_ITERATIONS_IMAGE = 5;

namespace bench
{
	volatile std::string testStrings(const char* stringToTest, const size_t length) {
		std::stringstream output;
		output << std::noskipws;

		// Strlen
		auto startStrlen = bench::now();
		for (size_t i = 0; i < BENCHMARK_ITERATIONS_STRING; i++) {
			wlib::str::strlen(stringToTest);
		}
		auto endStrlen = bench::now();
		output << "strlen: " << bench::as_us(endStrlen - startStrlen) / BENCHMARK_ITERATIONS_STRING << " microseconds\n";

		// Strcmp
		auto startStrcmp = bench::now();
		for (size_t i = 0; i < BENCHMARK_ITERATIONS_STRING; i++) {
			wlib::str::strncmp(stringToTest, stringToTest, length);
		}
		auto endStrcmp = bench::now();
		output << "strncmp: " << bench::as_us(endStrcmp - startStrcmp) / BENCHMARK_ITERATIONS_STRING << " microseconds\n";

		// Final output
		auto outString = output.str();
		return outString;
	}

	volatile std::string testImages(const std::vector<uint8_t>* rawPixels, const int width, const int height) {
		std::stringstream output;
		output << std::noskipws;

		// Construct AoS image
		auto startAoS = bench::now();
		for (size_t i = 0; i < BENCHMARK_ITERATIONS_IMAGE-1; i++) {
			wlib::image::Image imageAoS_RGB(rawPixels->data(), width, height, wlib::image::F_RGB);
		}
		wlib::image::Image imageAoS_RGB(rawPixels->data(), width, height, wlib::image::F_RGB);
		output << "constructing image: " << static_cast<float>(bench::as_us(bench::now() - startAoS)) / 1000.0f / BENCHMARK_ITERATIONS_IMAGE << " milliseconds\n";

		// Construct SoA image
		auto startToSoA = bench::now();
		for (size_t i = 0; i < BENCHMARK_ITERATIONS_IMAGE-1; i++) {
			wlib::image::ImageSoA imageSoA_RGB(imageAoS_RGB);
		}
		wlib::image::ImageSoA imageSoA_RGB(imageAoS_RGB);
		output << "conversion to SoA: " << static_cast<float>(bench::as_us(bench::now() - startToSoA)) / 1000.0f / BENCHMARK_ITERATIONS_IMAGE << " milliseconds\n";

		// Convert RGB to Grayscale
		auto startToGrayscale = bench::now();
		imageSoA_RGB = wlib::image::ConvertToGrayscale(imageSoA_RGB);
		output << "conversion to grayscale: " << static_cast<float>(bench::as_us(bench::now() - startToSoA)) / 1000.0f << " milliseconds\n";

		// Final output
		auto outString = output.str();
		return outString;
	}
} // namespace bench
