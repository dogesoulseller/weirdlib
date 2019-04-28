#include <gtest/gtest.h>
#include <cstring>
#include <random>
#include "../include/weirdlib.hpp"
#include <vector>
#include <string>

std::mt19937 random_generator;

#if defined(__RDRND__) && defined(__GNUC__) && !defined(__clang__)
TEST(WlibCrypto, RNG) {
	EXPECT_NO_THROW(wlib::crypto::SecureRandomInteger_u16());
	EXPECT_NO_THROW(wlib::crypto::SecureRandomInteger_u32());
	EXPECT_NO_THROW(wlib::crypto::SecureRandomInteger_u64());
}
#endif

#if defined(__RDSEED__) && defined(__GNUC__) && !defined(__clang__)
TEST(WlibCrypto, SEED) {
	EXPECT_NO_THROW(wlib::crypto::SecureRandomSeed_u16());
	EXPECT_NO_THROW(wlib::crypto::SecureRandomSeed_u32());
	EXPECT_NO_THROW(wlib::crypto::SecureRandomSeed_u64());
}
#endif

// Literal string - This is a test string used for testing.
std::vector<uint8_t> CRCTestVector = {0x54 ,0x68 ,0x69 ,0x73 ,0x20 ,0x69 ,0x73 ,0x20 ,0x61 ,0x20 ,0x74 ,0x65 ,0x73 ,0x74 ,0x20 ,0x73 ,0x74 ,0x72 ,0x69 ,0x6e ,0x67 ,0x20 ,0x75 ,0x73 ,0x65 ,0x64 ,0x20 ,0x66 ,0x6f ,0x72 ,0x20 ,0x74 ,0x65 ,0x73 ,0x74 ,0x69 ,0x6e ,0x67 ,0x2E};

TEST(WlibCrypto, CRC32) {
	EXPECT_EQ(wlib::crypto::CRC32(CRCTestVector.data(), CRCTestVector.data()+CRCTestVector.size()), 0xF6D84F7A);
}

TEST(WlibCrypto, CRC32C) {
	EXPECT_EQ(wlib::crypto::CRC32C(CRCTestVector.data(), CRCTestVector.data()+CRCTestVector.size()), 0x870B2648);
}

TEST(WlibCrypto, CRC32D) {
	EXPECT_EQ(wlib::crypto::CRC32D(CRCTestVector.data(), CRCTestVector.data()+CRCTestVector.size()), 0xB91E2065);
}

TEST(WlibCrypto, CRC32AUTOSAR) {
	EXPECT_NO_THROW(wlib::crypto::CRC32AUTOSAR(CRCTestVector.data(), CRCTestVector.data()+CRCTestVector.size()));
}

TEST(WlibCrypto, CRC32JAMCRC) {
	EXPECT_EQ(wlib::crypto::CRC32JAMCRC(CRCTestVector.data(), CRCTestVector.data()+CRCTestVector.size()), 0x0927B085);
}
