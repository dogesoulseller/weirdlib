#include "../../include/weirdlib_anxiety.hpp"
#include "../../include/cpu_detection.hpp"
#include "../../include/weirdlib_traits.hpp"
#include <random>
#include <cmath>
#include <array>
#include <thread>
#include <chrono>

namespace wlib::anxiety
{
	#ifdef WEIRDLIB_ENABLE_ANXIETY
	static inline std::uniform_real_distribution<double> float_dist(std::numeric_limits<double>::min(), std::numeric_limits<double>::max());
	static std::chrono::time_point<std::chrono::high_resolution_clock> startTimePoint;

	static double performFMA(std::mt19937_64& rng) {
		#if X86_SIMD_LEVEL >= 7
		[[maybe_unused]] volatile __m256d reg00 = _mm256_set1_pd(float_dist(rng));
		[[maybe_unused]] volatile __m256d reg01 = _mm256_set1_pd(float_dist(rng));
		[[maybe_unused]] volatile __m256d reg02 = _mm256_set1_pd(float_dist(rng));

		[[maybe_unused]] volatile __m256d reg10 = _mm256_set1_pd(float_dist(rng));
		[[maybe_unused]] volatile __m256d reg11 = _mm256_set1_pd(float_dist(rng));
		[[maybe_unused]] volatile __m256d reg12 = _mm256_set1_pd(float_dist(rng));

		[[maybe_unused]] volatile __m256d reg20 = _mm256_fmadd_pd(reg00, reg01, reg02);
		[[maybe_unused]] volatile __m256d reg21 = _mm256_fmadd_pd(reg10, reg11, reg12);
		#else
		[[maybe_unused]] volatile auto reg00 = std::fma(float_dist(rng), float_dist(rng), float_dist(rng));
		[[maybe_unused]] volatile auto reg01 = std::fma(float_dist(rng), float_dist(rng), float_dist(rng));
		[[maybe_unused]] volatile auto reg02 = std::fma(float_dist(rng), float_dist(rng), float_dist(rng));
		#endif
		return float_dist(rng);
	}

	void StressFMA(const std::chrono::milliseconds duration, const size_t threadCount) {
		std::vector<std::thread> hogThreads;
		hogThreads.reserve(threadCount);

		startTimePoint = std::chrono::high_resolution_clock::now() + std::chrono::milliseconds(100);

		for (size_t i = 0; i < threadCount; i++) {
			hogThreads.emplace_back([duration]{
				std::random_device rng_device;
				std::mt19937_64 rng(rng_device());

				std::this_thread::sleep_until(startTimePoint);

				auto iterStartPoint = std::chrono::high_resolution_clock::now();
				while (true) {
					[[maybe_unused]] volatile auto result = performFMA(rng);
					if (std::chrono::high_resolution_clock::now() - iterStartPoint > duration) {
						break;
					}
				}
			});
		}
		std::this_thread::sleep_for(duration);

		for (auto& t : hogThreads) {
			t.join();
		}
	}

	void StressFMA(const size_t durationMS, const size_t threadCount) {
		StressFMA(std::chrono::milliseconds(durationMS), threadCount);
	}

	#else
	constexpr const char* errMsg = "This function is a stub - stress test module was disabled for this compilation";

	void StressFMA(const std::chrono::milliseconds /*duration*/, const size_t /*threadCount*/) {
		throw wlib::module_not_built(errMsg);
	}
	void StressFMA(const size_t /*durationMS*/, const size_t /*threadCount*/) {
		throw wlib::module_not_built(errMsg);
	}
	#endif
} // namespace wlib::anxiety
