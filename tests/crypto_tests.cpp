#include <gtest/gtest.h>
#include <cstring>
#include "../include/weirdlib.hpp"

#if defined (__RDRND__) && defined (__GNUC__)
TEST(WlibCrypto, RNG) {
	EXPECT_NO_THROW(wlib::crypto::SecureRandomInteger_u16());
	EXPECT_NO_THROW(wlib::crypto::SecureRandomInteger_u32());
	EXPECT_NO_THROW(wlib::crypto::SecureRandomInteger_u64());
}
#endif

#if defined (__RDSEED__) && defined (__GNUC__)
TEST(WlibCrypto, SEED) {
	EXPECT_NO_THROW(wlib::crypto::SecureRandomSeed_u16());
	EXPECT_NO_THROW(wlib::crypto::SecureRandomSeed_u32());
	EXPECT_NO_THROW(wlib::crypto::SecureRandomSeed_u64());
}
#endif
