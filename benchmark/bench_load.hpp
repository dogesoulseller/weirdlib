#pragma once
#include <future>
#include <iostream>
#include <vector>
#include <filesystem>

namespace bench
{
	std::vector<uint8_t>* loadImage(const std::filesystem::path& imagePath, int channels);
} // namespace bench
