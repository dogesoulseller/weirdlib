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
	void StressSquareRoot(const std::chrono::milliseconds duration, const size_t threadCount);

	/// Stress testing via SIMD sqrt
	/// @param durationMS duration in milliseconds to perform stress test for
	/// @param threadCount number of threads to execute concurrently
	void StressSquareRoot(const size_t durationMS, const size_t threadCount);

	/// Stress testing via SIMD rsqrt
	/// @param duration duration to perform stress test for
	/// @param threadCount number of threads to execute concurrently
	void StressInverseSquareRoot(const std::chrono::milliseconds duration, const size_t threadCount);

	/// Stress testing via SIMD rsqrt
	/// @param durationMS duration in milliseconds to perform stress test for
	/// @param threadCount number of threads to execute concurrently
	void StressInverseSquareRoot(const size_t durationMS, const size_t threadCount);

	/// Stress testing via SIMD FMA
	/// @param duration duration to perform stress test for
	/// @param threadCount number of threads to execute concurrently
	void StressFMA(const std::chrono::milliseconds duration, const size_t threadCount);

	/// Stress testing via SIMD FMA
	/// @param durationMS duration in milliseconds to perform stress test for
	/// @param threadCount number of threads to execute concurrently
	void StressFMA(const size_t durationMS, const size_t threadCount);

} // namespace anxiety

} // namespace wlib
