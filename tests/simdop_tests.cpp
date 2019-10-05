#include <gtest/gtest.h>
#include <cstring>
#include <array>
#include <cstdint>

#include "../include/weirdlib.hpp"
#include "../include/weirdlib_simdhelper.hpp"

TEST(SIMDOps_X86, Reverse) {
#if X86_SIMD_LEVEL >= SSE2
	alignas(16) float floatArr[4];
	alignas(16) double doubleArr[2];
	alignas(16) uint8_t byteArr[16];
	alignas(16) uint16_t wordArr[8];
	alignas(16) uint32_t dwordArr[4];
	alignas(16) uint64_t qwordArr[2];

	float floatArrRev[4]    = {3.0f, 2.0f, 1.0f, 0.0f};
	double doubleArrRev[2]  = {1.0, 0.0};
	uint8_t byteArrRev[16]  = {15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0};
	uint16_t wordArrRev[8]  = {7, 6, 5, 4, 3, 2, 1, 0};
	uint32_t dwordArrRev[4] = {3, 2, 1, 0};
	uint64_t qwordArrRev[2] = {1, 0};

	__m128  floatVec  = _mm_set_ps(3.0f, 2.0f, 1.0f, 0.0f);
	__m128d doubleVec = _mm_set_pd(1.0, 0.0);
	__m128i byteVec   = _mm_set_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
	__m128i wordVec   = _mm_set_epi16(7, 6, 5, 4, 3, 2, 1, 0);
	__m128i dwordVec  = _mm_set_epi32(3, 2, 1, 0);
	__m128i qwordVec  = _mm_set_epi64x(1, 0);

	__m128  floatVecRev = wlib::simd::reverse(floatVec);
	__m128d doubleVecRev = wlib::simd::reverse(doubleVec);
	__m128i byteVecRev = wlib::simd::reverse<8>(byteVec);
	__m128i wordVecRev = wlib::simd::reverse<16>(wordVec);
	__m128i dwordVecRev = wlib::simd::reverse<32>(dwordVec);
	__m128i qwordVecRev = wlib::simd::reverse<64>(qwordVec);

	_mm_store_ps(floatArr, floatVecRev);
	_mm_store_pd(doubleArr, doubleVecRev);
	_mm_store_si128(reinterpret_cast<__m128i*>(byteArr), byteVecRev);
	_mm_store_si128(reinterpret_cast<__m128i*>(wordArr), wordVecRev);
	_mm_store_si128(reinterpret_cast<__m128i*>(dwordArr), dwordVecRev);
	_mm_store_si128(reinterpret_cast<__m128i*>(qwordArr), qwordVecRev);

	EXPECT_TRUE(std::equal(floatArr, floatArr+4, floatArrRev));
	EXPECT_TRUE(std::equal(doubleArr, doubleArr+2, doubleArrRev));
	EXPECT_TRUE(std::equal(byteArr, byteArr+16, byteArrRev));
	EXPECT_TRUE(std::equal(wordArr, wordArr+8, wordArrRev));
	EXPECT_TRUE(std::equal(dwordArr, dwordArr+4, dwordArrRev));
	EXPECT_TRUE(std::equal(qwordArr, qwordArr+2, qwordArrRev));
#endif

#if X86_SIMD_LEVEL >= LV_AVX
	alignas(32) float floatArrAVX[8];
	alignas(32) double doubleArrAVX[4];
	alignas(32) uint64_t intArrAVX[4];

	alignas(32) float floatArrAVXLRev[8] = {4.0f, 5.0f, 6.0f, 7.0f, 0.0f, 1.0f, 2.0f, 3.0f};
	alignas(32) double doubleArrAVXLRev[4] = {2.0, 3.0, 0.0, 1.0};
	alignas(32) uint64_t intArrAVXLRev[4] = {2, 3, 0, 1};

	__m256 floatVecAVX = _mm256_set_ps(7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f, 0.0f);
	__m256d doubleVecAVX = _mm256_set_pd(3.0, 2.0, 1.0, 0.0);
	__m256i intVecAVX = _mm256_set_epi64x(3, 2, 1, 0);

	__m256 floatVecAVXLRev = wlib::simd::reverseLanes(floatVecAVX);
	__m256d doubleVecAVXLRev = wlib::simd::reverseLanes(doubleVecAVX);
	__m256i intVecAVXLRev = wlib::simd::reverseLanes(intVecAVX);

	_mm256_store_ps(floatArrAVX, floatVecAVXLRev);
	_mm256_store_pd(doubleArrAVX, doubleVecAVXLRev);
	_mm256_store_si256(reinterpret_cast<__m256i*>(intArrAVX), intVecAVXLRev);

	EXPECT_TRUE(std::equal(floatArrAVX, floatArrAVX+8, floatArrAVXLRev));
	EXPECT_TRUE(std::equal(doubleArrAVX, doubleArrAVX+4, doubleArrAVXLRev));
	EXPECT_TRUE(std::equal(intArrAVX, intArrAVX+4, intArrAVXLRev));

#endif

}
