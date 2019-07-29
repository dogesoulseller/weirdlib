#pragma once
#include <cstddef>
#include <chrono>

namespace wlib
{

/// Stress testing utilities
namespace anxiety
{

	void StressSquareRoot(std::chrono::milliseconds duration, size_t threadCount);
	void StressSquareRoot(size_t durationMS, size_t threadCount);

	void StressInverseSquareRoot(std::chrono::milliseconds duration, size_t threadCount);
	void StressInverseSquareRoot(size_t durationMS, size_t threadCount);

	void StressFMA(std::chrono::milliseconds duration, size_t threadCount);
	void StressFMA(size_t durationMS, size_t threadCount);

} // namespace anxiety

} // namespace wlib
