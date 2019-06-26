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
#include <future>
#include <functional>
#include <tuple>

alignas(64) char lipsumString[10000000];

const char* lipsumLongName = "loremipsum-thicc.txt";
const char* imageVeryLargeName = "image-vlarge.jpg";
const char* imageLargeName = "image-large.jpg";
const char* imageMediumName = "image-medium.jpg";
const char* imageSmallName = "image-small.jpg";

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

	// Load images asynchronously
	std::packaged_task<std::vector<uint8_t>*()> vlargeLoad(std::bind(bench::loadImage, std::cref(imageVeryLargePath), 3));
	std::packaged_task<std::vector<uint8_t>*()> largeLoad(std::bind(bench::loadImage, std::cref(imageLargePath), 3));
	std::packaged_task<std::vector<uint8_t>*()> mediumLoad(std::bind(bench::loadImage, std::cref(imageMediumPath), 3));
	std::packaged_task<std::vector<uint8_t>*()> smallLoad(std::bind(bench::loadImage, std::cref(imageSmallPath), 3));

	auto vlargeImage_future = vlargeLoad.get_future();
	auto largeImage_future = largeLoad.get_future();
	auto mediumImage_future = mediumLoad.get_future();
	auto smallImage_future = smallLoad.get_future();

	std::thread t1([&](){vlargeLoad();});
	std::thread t2([&](){largeLoad();});
	std::thread t3([&](){mediumLoad();});
	std::thread t4([&](){smallLoad();});

	// Load text
	std::ifstream loremIpsum(lipsumLongPath);

	loremIpsum.seekg(0, std::ios::end);
	auto fSize = loremIpsum.tellg();

	loremIpsum.seekg(0, std::ios::beg);
	loremIpsum.read(lipsumString, fSize);
	loremIpsum.close();

	// Wait for all images
	auto vlargeImage = vlargeImage_future.get();
	auto largeImage = largeImage_future.get();
	auto mediumImage = mediumImage_future.get();
	auto smallImage = smallImage_future.get();

	// Benchmark cases

	std::this_thread::sleep_for(std::chrono::milliseconds(100));

	std::cout << "String processing:\n" << (std::string)bench::testStrings(lipsumString, fSize) << std::endl;

	std::cout << "-------------------- IMAGE PROCESSING --------------------" << "\n\n";

	std::cout << "Small image processing:\n" << (std::string)bench::testImages(smallImage, imageSmallDimensions.first, imageSmallDimensions.second) << '\n';
	std::cout << "Medium image processing:\n" << (std::string)bench::testImages(mediumImage, imageMediumDimensions.first, imageMediumDimensions.second) << '\n';
	std::cout << "Large image processing:\n" << (std::string)bench::testImages(largeImage, imageLargeDimensions.first, imageLargeDimensions.second) << '\n';
	std::cout << "Very large image processing:\n" << (std::string)bench::testImages(vlargeImage, imageVeryLargeDimensions.first, imageVeryLargeDimensions.second) << '\n';

	delete vlargeImage;
	delete largeImage;
	delete mediumImage;
	delete smallImage;
	return 0;
}
