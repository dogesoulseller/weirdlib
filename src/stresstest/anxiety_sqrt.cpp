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

	static inline std::uniform_real_distribution<double> float_dist(1.0, std::numeric_limits<double>::max());
	static inline std::uniform_real_distribution<float> spfloat_dist(1.0, std::numeric_limits<float>::max());
	static std::chrono::time_point<std::chrono::high_resolution_clock> startTimePoint;

	static double performSqrt(std::mt19937_64& rng) noexcept {
		#if X86_SIMD_LEVEL >= 9
			alignas(64) std::array<double, 8> output;
			volatile __m512d x0 = _mm512_set1_pd(float_dist(rng));
			volatile __m512d x1 = _mm512_set1_pd(float_dist(rng));
			volatile __m512d x2 = _mm512_set1_pd(float_dist(rng));
			volatile __m512d x3 = _mm512_set1_pd(float_dist(rng));
			volatile __m512d x4 = _mm512_set1_pd(float_dist(rng));
			volatile __m512d x5 = _mm512_set1_pd(float_dist(rng));
			volatile __m512d x6 = _mm512_set1_pd(float_dist(rng));
			volatile __m512d x7 = _mm512_set1_pd(float_dist(rng));
			[[maybe_unused]] volatile __m512d tmp0 = _mm512_sqrt_pd(x0);
			[[maybe_unused]] volatile __m512d tmp1 = _mm512_sqrt_pd(x1);
			[[maybe_unused]] volatile __m512d tmp2 = _mm512_sqrt_pd(x2);
			[[maybe_unused]] volatile __m512d tmp3 = _mm512_sqrt_pd(x3);
			[[maybe_unused]] volatile __m512d tmp4 = _mm512_sqrt_pd(x4);
			[[maybe_unused]] volatile __m512d tmp5 = _mm512_sqrt_pd(x5);
			[[maybe_unused]] volatile __m512d tmp6 = _mm512_sqrt_pd(x6);
			[[maybe_unused]] volatile __m512d tmp7 = _mm512_sqrt_pd(x7);
			_mm512_store_pd(output.data(), tmp0);
		#elif X86_SIMD_LEVEL >= 7
			alignas(32) std::array<double, 4> output;
			volatile __m256d x0 = _mm256_set1_pd(float_dist(rng));
			volatile __m256d x1 = _mm256_set1_pd(float_dist(rng));
			volatile __m256d x2 = _mm256_set1_pd(float_dist(rng));
			volatile __m256d x3 = _mm256_set1_pd(float_dist(rng));
			volatile __m256d x4 = _mm256_set1_pd(float_dist(rng));
			volatile __m256d x5 = _mm256_set1_pd(float_dist(rng));
			volatile __m256d x6 = _mm256_set1_pd(float_dist(rng));
			volatile __m256d x7 = _mm256_set1_pd(float_dist(rng));
			[[maybe_unused]] volatile __m256d tmp0 = _mm256_sqrt_pd(x0);
			[[maybe_unused]] volatile __m256d tmp1 = _mm256_sqrt_pd(x1);
			[[maybe_unused]] volatile __m256d tmp2 = _mm256_sqrt_pd(x2);
			[[maybe_unused]] volatile __m256d tmp3 = _mm256_sqrt_pd(x3);
			[[maybe_unused]] volatile __m256d tmp4 = _mm256_sqrt_pd(x4);
			[[maybe_unused]] volatile __m256d tmp5 = _mm256_sqrt_pd(x5);
			[[maybe_unused]] volatile __m256d tmp6 = _mm256_sqrt_pd(x6);
			[[maybe_unused]] volatile __m256d tmp7 = _mm256_sqrt_pd(x7);
			_mm256_store_pd(output.data(), tmp0);
		#elif X86_SIMD_LEVEL >= 2
			alignas(16) std::array<double, 2> output;
			volatile __m128d x0 = _mm_set1_pd(float_dist(rng));
			volatile __m128d x1 = _mm_set1_pd(float_dist(rng));
			volatile __m128d x2 = _mm_set1_pd(float_dist(rng));
			volatile __m128d x3 = _mm_set1_pd(float_dist(rng));
			volatile __m128d x4 = _mm_set1_pd(float_dist(rng));
			volatile __m128d x5 = _mm_set1_pd(float_dist(rng));
			volatile __m128d x6 = _mm_set1_pd(float_dist(rng));
			volatile __m128d x7 = _mm_set1_pd(float_dist(rng));
			[[maybe_unused]] volatile __m128d tmp0 = _mm_sqrt_pd(x0);
			[[maybe_unused]] volatile __m128d tmp1 = _mm_sqrt_pd(x1);
			[[maybe_unused]] volatile __m128d tmp2 = _mm_sqrt_pd(x2);
			[[maybe_unused]] volatile __m128d tmp3 = _mm_sqrt_pd(x3);
			[[maybe_unused]] volatile __m128d tmp4 = _mm_sqrt_pd(x4);
			[[maybe_unused]] volatile __m128d tmp5 = _mm_sqrt_pd(x5);
			[[maybe_unused]] volatile __m128d tmp6 = _mm_sqrt_pd(x6);
			[[maybe_unused]] volatile __m128d tmp7 = _mm_sqrt_pd(x7);
			_mm_store_pd(output.data(), tmp0);
		#else
			[[maybe_unused]] volatile auto x0 = sqrtf64(float_dist(rng));
			[[maybe_unused]] volatile auto x1 = sqrtf64(float_dist(rng));
			[[maybe_unused]] volatile auto x2 = sqrtf64(float_dist(rng));
			[[maybe_unused]] volatile auto x3 = sqrtf64(float_dist(rng));
			[[maybe_unused]] volatile auto x4 = sqrtf64(float_dist(rng));
			[[maybe_unused]] volatile auto x5 = sqrtf64(float_dist(rng));
			[[maybe_unused]] volatile auto x6 = sqrtf64(float_dist(rng));
			[[maybe_unused]] volatile auto x7 = sqrtf64(float_dist(rng));
			return x7;
		#endif
		return output[0];
	}

	static float performISqrt(std::mt19937& rng) noexcept {
		#if X86_SIMD_LEVEL >= 9
			alignas(64) std::array<float, 16> output;
			volatile __m512 x0 = _mm512_set1_ps(spfloat_dist(rng));
			volatile __m512 x1 = _mm512_set1_ps(spfloat_dist(rng));
			volatile __m512 x2 = _mm512_set1_ps(spfloat_dist(rng));
			volatile __m512 x3 = _mm512_set1_ps(spfloat_dist(rng));
			volatile __m512 x4 = _mm512_set1_ps(spfloat_dist(rng));
			volatile __m512 x5 = _mm512_set1_ps(spfloat_dist(rng));
			volatile __m512 x6 = _mm512_set1_ps(spfloat_dist(rng));
			volatile __m512 x7 = _mm512_set1_ps(spfloat_dist(rng));
			[[maybe_unused]] volatile __m512 tmp0 = _mm512_rsqrt14_ps(x0);
			[[maybe_unused]] volatile __m512 tmp1 = _mm512_rsqrt14_ps(x1);
			[[maybe_unused]] volatile __m512 tmp2 = _mm512_rsqrt14_ps(x2);
			[[maybe_unused]] volatile __m512 tmp3 = _mm512_rsqrt14_ps(x3);
			[[maybe_unused]] volatile __m512 tmp4 = _mm512_rsqrt14_ps(x4);
			[[maybe_unused]] volatile __m512 tmp5 = _mm512_rsqrt14_ps(x5);
			[[maybe_unused]] volatile __m512 tmp6 = _mm512_rsqrt14_ps(x6);
			[[maybe_unused]] volatile __m512 tmp7 = _mm512_rsqrt14_ps(x7);
			_mm512_store_ps(output.data(), tmp0);
		#elif X86_SIMD_LEVEL >= 7
			alignas(32) std::array<float, 8> output;
			volatile __m256 x0 = _mm256_set1_ps(spfloat_dist(rng));
			volatile __m256 x1 = _mm256_set1_ps(spfloat_dist(rng));
			volatile __m256 x2 = _mm256_set1_ps(spfloat_dist(rng));
			volatile __m256 x3 = _mm256_set1_ps(spfloat_dist(rng));
			volatile __m256 x4 = _mm256_set1_ps(spfloat_dist(rng));
			volatile __m256 x5 = _mm256_set1_ps(spfloat_dist(rng));
			volatile __m256 x6 = _mm256_set1_ps(spfloat_dist(rng));
			volatile __m256 x7 = _mm256_set1_ps(spfloat_dist(rng));
			[[maybe_unused]] volatile __m256 tmp0 = _mm256_sqrt_ps(x0);
			[[maybe_unused]] volatile __m256 tmp1 = _mm256_sqrt_ps(x1);
			[[maybe_unused]] volatile __m256 tmp2 = _mm256_sqrt_ps(x2);
			[[maybe_unused]] volatile __m256 tmp3 = _mm256_sqrt_ps(x3);
			[[maybe_unused]] volatile __m256 tmp4 = _mm256_sqrt_ps(x4);
			[[maybe_unused]] volatile __m256 tmp5 = _mm256_sqrt_ps(x5);
			[[maybe_unused]] volatile __m256 tmp6 = _mm256_sqrt_ps(x6);
			[[maybe_unused]] volatile __m256 tmp7 = _mm256_sqrt_ps(x7);
			_mm256_store_ps(output.data(), tmp0);
		#elif X86_SIMD_LEVEL >= 2
			alignas(16) std::array<float, 4> output;
			volatile __m128 x0 = _mm_set1_ps(spfloat_dist(rng));
			volatile __m128 x1 = _mm_set1_ps(spfloat_dist(rng));
			volatile __m128 x2 = _mm_set1_ps(spfloat_dist(rng));
			volatile __m128 x3 = _mm_set1_ps(spfloat_dist(rng));
			volatile __m128 x4 = _mm_set1_ps(spfloat_dist(rng));
			volatile __m128 x5 = _mm_set1_ps(spfloat_dist(rng));
			volatile __m128 x6 = _mm_set1_ps(spfloat_dist(rng));
			volatile __m128 x7 = _mm_set1_ps(spfloat_dist(rng));
			[[maybe_unused]] volatile __m128 tmp0 = _mm_rsqrt_ps(x0);
			[[maybe_unused]] volatile __m128 tmp1 = _mm_rsqrt_ps(x1);
			[[maybe_unused]] volatile __m128 tmp2 = _mm_rsqrt_ps(x2);
			[[maybe_unused]] volatile __m128 tmp3 = _mm_rsqrt_ps(x3);
			[[maybe_unused]] volatile __m128 tmp4 = _mm_rsqrt_ps(x4);
			[[maybe_unused]] volatile __m128 tmp5 = _mm_rsqrt_ps(x5);
			[[maybe_unused]] volatile __m128 tmp6 = _mm_rsqrt_ps(x6);
			[[maybe_unused]] volatile __m128 tmp7 = _mm_rsqrt_ps(x7);
			_mm_store_ps(output.data(), tmp0);
		#else
			[[maybe_unused]] volatile auto x0 = 1.0f/sqrtf32(spfloat_dist(rng));
			[[maybe_unused]] volatile auto x1 = 1.0f/sqrtf32(spfloat_dist(rng));
			[[maybe_unused]] volatile auto x2 = 1.0f/sqrtf32(spfloat_dist(rng));
			[[maybe_unused]] volatile auto x3 = 1.0f/sqrtf32(spfloat_dist(rng));
			[[maybe_unused]] volatile auto x4 = 1.0f/sqrtf32(spfloat_dist(rng));
			[[maybe_unused]] volatile auto x5 = 1.0f/sqrtf32(spfloat_dist(rng));
			[[maybe_unused]] volatile auto x6 = 1.0f/sqrtf32(spfloat_dist(rng));
			[[maybe_unused]] volatile auto x7 = 1.0f/sqrtf32(spfloat_dist(rng));
			return x7;
		#endif
		return output[0];
	}

	void StressSquareRoot(const std::chrono::milliseconds duration, const size_t threadCount) {
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
					[[maybe_unused]] volatile auto result = performSqrt(rng);
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

	void StressSquareRoot(const size_t durationMS, const size_t threadCount) {
		StressSquareRoot(std::chrono::milliseconds(durationMS), threadCount);
	}

	void StressInverseSquareRoot(const std::chrono::milliseconds duration, const size_t threadCount) {
		std::vector<std::thread> hogThreads;
		hogThreads.reserve(threadCount);

		startTimePoint = std::chrono::high_resolution_clock::now() + std::chrono::milliseconds(100);

		for (size_t i = 0; i < threadCount; i++) {
			hogThreads.emplace_back([duration]{
				std::random_device rng_device;
				std::mt19937 rng(rng_device());

				std::this_thread::sleep_until(startTimePoint);

				auto iterStartPoint = std::chrono::high_resolution_clock::now();
				while (true) {
					[[maybe_unused]] volatile auto result = performISqrt(rng);
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

	void StressInverseSquareRoot(const size_t durationMS, const size_t threadCount) {
		StressInverseSquareRoot(std::chrono::milliseconds(durationMS), threadCount);
	}

	#else
	constexpr const char* errMsg = "This function is a stub - stress test module was disabled for this compilation";

	void StressSquareRoot(const std::chrono::milliseconds /*duration*/, const size_t /*threadCount*/) {
		throw wlib::module_not_built(errMsg);
	}
	void StressSquareRoot(const size_t /*durationMS*/, const size_t /*threadCount*/) {
		throw wlib::module_not_built(errMsg);
	}
	void StressInverseSquareRoot(const std::chrono::milliseconds /*duration*/, const size_t /*threadCount*/) {
		throw wlib::module_not_built(errMsg);
	}
	void StressInverseSquareRoot(const size_t /*durationMS*/, const size_t /*threadCount*/) {
		throw wlib::module_not_built(errMsg);
	}
	#endif

} // namespace wlib::anxiety
