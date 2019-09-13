#include "common.hpp"
#include <thread>

static const auto CPU_CORE_COUNT = static_cast<int>(std::thread::hardware_concurrency());

int getImagePreferredThreadCount(const int width, const int height) {
	int numThreadsLocal = -1;
	numThreadsLocal = width * height >= 3000*3000 ? CPU_CORE_COUNT : numThreadsLocal;
	numThreadsLocal = width * height >= 2000*2000 && CPU_CORE_COUNT >= 8 ? 8 : numThreadsLocal;
	numThreadsLocal = width * height >= 1000*1000 && CPU_CORE_COUNT >= 4 ? 4 : numThreadsLocal;
	numThreadsLocal = width * height >= 700*700 && CPU_CORE_COUNT >= 3 ? 3 : numThreadsLocal;
	numThreadsLocal = numThreadsLocal == -1 ? 2 : numThreadsLocal;

	return numThreadsLocal;
}
