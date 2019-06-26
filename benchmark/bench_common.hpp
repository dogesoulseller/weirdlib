#pragma once
#include <chrono>
#include <random>
#include <limits>

namespace bench
{
	static inline std::random_device rng_device;
	static inline std::mt19937 rng(rng_device());
	static inline std::uniform_real_distribution<float> float_dist(std::numeric_limits<float>::min(), std::numeric_limits<float>::max());

	template<typename Time_T = std::chrono::nanoseconds>
	inline std::chrono::time_point<std::chrono::high_resolution_clock, Time_T> now() {
		return std::chrono::high_resolution_clock::now();
	}

	template<typename Dur_T>
	inline size_t as_us(Dur_T dur) {
		return std::chrono::duration_cast<std::chrono::microseconds>(dur).count();
	}

	volatile inline float random_float() {
		return float_dist(rng);
	}
} // bench
