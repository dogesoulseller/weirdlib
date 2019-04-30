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


TEST(WlibCrypto, CRC16ARC) {
	EXPECT_EQ(wlib::crypto::CRC16ARC(CRCTestVector.data(), CRCTestVector.data()+CRCTestVector.size()), 0x7698);
}

TEST(WlibCrypto, CRC16DNP) {
	EXPECT_EQ(wlib::crypto::CRC16DNP(CRCTestVector.data(), CRCTestVector.data()+CRCTestVector.size()), 0xC7DF);
}

TEST(WlibCrypto, CRC16MAXIM) {
	EXPECT_EQ(wlib::crypto::CRC16MAXIM(CRCTestVector.data(), CRCTestVector.data()+CRCTestVector.size()), 0x8967);
}

TEST(WlibCrypto, CRC16USB) {
	EXPECT_EQ(wlib::crypto::CRC16USB(CRCTestVector.data(), CRCTestVector.data()+CRCTestVector.size()), 0x9D6B);
}

TEST(WlibCrypto, CRC16X_25) {
	EXPECT_EQ(wlib::crypto::CRC16X_25(CRCTestVector.data(), CRCTestVector.data()+CRCTestVector.size()), 0x8961);
}




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




TEST(WlibCrypto, CRC64XZ) {
	EXPECT_EQ(wlib::crypto::CRC64XZ(CRCTestVector.data(), CRCTestVector.data()+CRCTestVector.size()), 0x89E09CD81D0D68A0);
}

TEST(WlibCrypto, CRC64ISO) {
	EXPECT_NO_THROW(wlib::crypto::CRC64ISO(CRCTestVector.data(), CRCTestVector.data()+CRCTestVector.size()));
}
