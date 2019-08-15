#pragma once
#include <weirdlib_fileops.hpp>
#include <string>
#include <filesystem>
#include <vector>
#include <iostream>
#include <functional>
#include <unordered_map>
#include <algorithm>

enum ParameterName
{
	RECURSION_DEPTH,
	OUTPUT_DIR
};

bool StringIsInteger(const std::string& str);


std::unordered_map<ParameterName, std::string> GetParameters(std::vector<std::string>& args);
