#include "../../include/weirdlib_anxiety.hpp"
#include "../../include/cpu_detection.hpp"
#include <random>
#include <cmath>
#include <array>
#include <thread>
#include <chrono>

namespace wlib::anxiety
{
	static inline std::uniform_real_distribution<double> float_dist(std::numeric_limits<double>::min(), std::numeric_limits<double>::max());
	static std::chrono::time_point<std::chrono::high_resolution_clock> startTimePoint;

	double performFMA(std::mt19937_64& rng) {
		volatile __m256d reg00 = _mm256_set1_pd(float_dist(rng));
		volatile __m256d reg01 = _mm256_set1_pd(float_dist(rng));
		volatile __m256d reg02 = _mm256_set1_pd(float_dist(rng));

		volatile __m256d reg10 = _mm256_set1_pd(float_dist(rng));
		volatile __m256d reg11 = _mm256_set1_pd(float_dist(rng));
		volatile __m256d reg12 = _mm256_set1_pd(float_dist(rng));

		volatile __m256d reg20 = _mm256_fmadd_pd(reg00, reg01, reg02);
		volatile __m256d reg21 = _mm256_fmadd_pd(reg10, reg11, reg12);
		return float_dist(rng);
	}

	void StressFMA(std::chrono::milliseconds duration, size_t threadCount) {
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

	void StressFMA(size_t durationMS, size_t threadCount) {
		StressFMA(std::chrono::milliseconds(durationMS), threadCount);
	}

} // namespace wlib::anxiety
