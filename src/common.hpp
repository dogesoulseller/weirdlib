#pragma once
#include "../include/weirdlib_simdhelper.hpp"

#include <type_traits>
#include <cstdint>

#define PP_CAT(a, b) PP_CAT_I(a, b)
#define PP_CAT_I(a, b) PP_CAT_II(~, a ## b)
#define PP_CAT_II(p, res) res

#define IGNORESB PP_CAT(PP_CAT(_unused_sb_param__, __LINE__), __COUNTER__)


/// Get optimal thread count for processing image of given size
/// @param width image width
/// @param height image height
/// @return thread count
int getImagePreferredThreadCount(int width, int height);
