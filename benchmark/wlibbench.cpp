#ifdef WEIRDLIB_ENABLE_IMAGE_OPERATIONS
#include "../include/weirdlib.hpp"
#include "bench_common.hpp"
#include "bench_load.hpp"
#include "bench_cases.hpp"

#include <fstream>
#include <string>
#include <cstring>
#include <thread>
#include <iostream>
#include <filesystem>
#include <functional>
#include <tuple>

alignas(64) static char lipsumString[10000000];

static const char* lipsumLongName = "loremipsum-thicc.txt";
static const char* imageVeryLargeName = "image-vlarge.jpg";
static const char* imageLargeName = "image-large.jpg";
static const char* imageMediumName = "image-medium.jpg";
static const char* imageSmallName = "image-small.jpg";

static std::filesystem::path lipsumLongPath;
static std::filesystem::path imageVeryLargePath;
static std::filesystem::path imageLargePath;
static std::filesystem::path imageMediumPath;
static std::filesystem::path imageSmallPath;

//TODO: Get dimensions from file
constexpr std::pair<int, int> imageSmallDimensions = {512, 492};
constexpr std::pair<int, int> imageMediumDimensions = {1024, 983};
constexpr std::pair<int, int> imageLargeDimensions = {2048, 1966};
constexpr std::pair<int, int> imageVeryLargeDimensions = {4096, 3933};

int main(int argc, char* argv[]) {
	if (argc < 2) {
		std::cerr << "Not enough arguments passed. Usage:\n" << "wlibbench {bench materials dir}\n";
		return 1;
	}

	// Create all test file paths
	if (auto rootDir = std::filesystem::path(argv[1]); std::filesystem::exists(rootDir)) {
		lipsumLongPath = rootDir / lipsumLongName;
		imageVeryLargePath = rootDir / imageVeryLargeName;
		imageLargePath = rootDir / imageLargeName;
		imageMediumPath = rootDir / imageMediumName;
		imageSmallPath = rootDir / imageSmallName;
	} else {
		std::cerr << "Root dir is invalid or does not exist\n";
		return 2;
	}

	// Start asynchronously loading images
	std::vector<uint8_t>* vlargeImage;
	std::vector<uint8_t>* largeImage;
	std::vector<uint8_t>* mediumImage;
	std::vector<uint8_t>* smallImage;

	std::thread t1([&](){vlargeImage = bench::loadImage(imageVeryLargePath, 3);});
	std::thread t2([&](){largeImage = bench::loadImage(imageLargePath, 3);});
	std::thread t3([&](){mediumImage = bench::loadImage(imageMediumPath, 3);});
	std::thread t4([&](){smallImage = bench::loadImage(imageSmallPath, 3);});

	// Load text
	std::ifstream loremIpsum(lipsumLongPath);

	loremIpsum.seekg(0, std::ios::end);
	auto fSize = loremIpsum.tellg();

	loremIpsum.seekg(0, std::ios::beg);
	loremIpsum.read(lipsumString, fSize);
	loremIpsum.close();

	// Wait for all images
	t1.join(); t2.join(); t3.join(); t4.join();

	// Benchmark cases

	std::this_thread::sleep_for(std::chrono::milliseconds(100));

	std::cout << "String processing:\n" << bench::testStrings(lipsumString, fSize) << std::endl;

	std::cout << "-------------------- IMAGE PROCESSING --------------------" << "\n\n";

	std::cout << "Small image processing:\n" << bench::testImages(smallImage, imageSmallDimensions.first, imageSmallDimensions.second) << '\n';
	std::cout << "Medium image processing:\n" << bench::testImages(mediumImage, imageMediumDimensions.first, imageMediumDimensions.second) << '\n';
	std::cout << "Large image processing:\n" << bench::testImages(largeImage, imageLargeDimensions.first, imageLargeDimensions.second) << '\n';
	std::cout << "Very large image processing:\n" << bench::testImages(vlargeImage, imageVeryLargeDimensions.first, imageVeryLargeDimensions.second) << '\n';

	delete vlargeImage;
	delete largeImage;
	delete mediumImage;
	delete smallImage;
	return 0;
}
#else
#include <iostream>
int main() {
	std::cerr << "Missing some libraries required for benchmarks. Error";
	return -1;
}

#endif
