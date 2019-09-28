#pragma once
#include <cstddef>
#include <chrono>

namespace wlib
{

/// Stress testing utilities
namespace anxiety
{
	/// Stress testing via SIMD sqrt
	/// @param duration duration to perform stress test for
	/// @param threadCount number of threads to execute concurrently
	void StressSquareRoot(std::chrono::milliseconds duration, size_t threadCount);

	/// Stress testing via SIMD sqrt
	/// @param durationMS duration in milliseconds to perform stress test for
	/// @param threadCount number of threads to execute concurrently
	void StressSquareRoot(size_t durationMS, size_t threadCount);

	/// Stress testing via SIMD rsqrt
	/// @param duration duration to perform stress test for
	/// @param threadCount number of threads to execute concurrently
	void StressInverseSquareRoot(std::chrono::milliseconds duration, size_t threadCount);

	/// Stress testing via SIMD rsqrt
	/// @param durationMS duration in milliseconds to perform stress test for
	/// @param threadCount number of threads to execute concurrently
	void StressInverseSquareRoot(size_t durationMS, size_t threadCount);

	/// Stress testing via SIMD FMA
	/// @param duration duration to perform stress test for
	/// @param threadCount number of threads to execute concurrently
	void StressFMA(std::chrono::milliseconds duration, size_t threadCount);

	/// Stress testing via SIMD FMA
	/// @param durationMS duration in milliseconds to perform stress test for
	/// @param threadCount number of threads to execute concurrently
	void StressFMA(size_t durationMS, size_t threadCount);

} // namespace anxiety

} // namespace wlib
