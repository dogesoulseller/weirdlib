#ifdef WEIRDLIB_ENABLE_STRING_OPERATIONS
#include <gtest/gtest.h>
#include <cstring>
#include "../include/weirdlib.hpp"


static const char* loremIpsum = "Lorem ipsum dolor sit amet, consectetur adipiscing elit."
"Sed feugiat, orci eu varius efficitur, arcu ex condimentum leo, et auctor felis nisi ut sapien. Aliquam at rutrum ante. Vivamus nisl neque, condimentum sed tincidunt eget, pellentesque sed enim. Phasellus mollis enim nibh. Suspendisse potenti. Morbi interdum consectetur commodo. Morbi ac feugiat dolor. Vestibulum nunc erat, pharetra non eros id, pharetra pretium tortor.Curabitur elementum,"
"massa id sagittis interdum, urna metus interdum urna, eu scelerisque eros ex non ligula. Donec feugiat nisi velit, ac lobortis elit volutpat et. Suspendisse eu dui mattis, accumsan sem in, hendrerit nulla. In metus ligula, ullamcorper ut semper ut, fringilla commodo velit. Proin consectetur, mi a congue eleifend, leo ex sollicitudin ex, sit amet tincidunt libero velit ut arcu."
"Maecenas laoreet ex leo, quis finibus nisl dictum quis. Nullam et consequat nunc. Donec tempus nisi vitae tortor blandit, quis pulvinar magna feugiat. Donec sodales eu urna in suscipit. Vivamus commodo efficitur urna, id varius velit cursus a. Sed consequat arcu vitae dui sagittis consequat. Vestibulum ante ipsum primis in faucibus orci luctus et ultrices posuere cubilia Curae; Vivamus volutpat dictum neque."
" Proin congue metus sit amet elit tempus iaculis.Praesent sodales neque auctor velit pulvinar, et dictum orci luctus. Ut vulputate maximus luctus. Aenean tristique lobortis risus a vulputate. Nam quis augue sit amet metus fringilla sodales eu non massa. Ut vel nibh a nulla dapibus venenatis. Nam vel interdum magna. In posuere ligula non tortor efficitur lobortis. Praesent et ex semper massa mattis malesuada at quis felis. Ut hendrerit at turpis vel pretium. Nulla sit amet neque tortor."
"Vestibulum et nulla ac justo posuere cursus ut eget purus. Maecenas ornare euismod odio eget maximus. Duis lacinia, est quis egestas eleifend, arcu orci lobortis tortor, dictum tristique dui dui ut neque. Etiam semper leo ac nisl vehicula, et fringilla arcu porta. Cras sit amet sem quis quam finibus scelerisque.Duis suscipit tellus eu mi rhoncus, in imperdiet nisl pretium. Quisque ultricies congue cursus."
" Integer sit amet ex enim. Sed elit turpis, "
"congue at auctor sit amet, luctus eget nibh. In tincidunt nisi ac feugiat tempus. In elementum mi sit amet tincidunt feugiat. Proin lacinia, neque vitae suscipit venenatis, mauris ante mollis sem, at ultrices quam nulla ac elit. In at risus id mauris viverra maximus. Ut imperdiet lectus aliquet risus auctor ultricies. Etiam varius imperdiet neque, ac convallis neque congue in. Class aptent taciti sociosqu ad "
"litora torquent per conubia nostra, per inceptos himenaeos. Proin pharetra justo eu turpis iaculis, ut condimentum enim pretium. Quisque lacinia diam felis, auctor venenatis ipsum commodo vel. Vestibulum congue efficitur lectus, sit amet vehicula ipsum suscipit vel. In pellentesque magna sit amet iaculis pharetra.Duis efficitur nunc porta orci mollis posuere. Duis eu ante sit amet odio pretium scelerisque. "
"Donec imperdiet mauris non sem auctor, in tempor erat ornare. Fusce quam ante, placerat eget neque vitae, consequat imperdiet massa. Mauris quis augue nisi. Aenean tristique fringilla viverra. Integer feugiat imperdiet velit non semper. Aenean quis fringilla nibh. Duis tempus neque turpis, vel rutrum nulla tincidunt quis. "
"In fringilla nec dolor vel semper.Nam ullamcorper a diam vitae consectetur. Praesent leo tortor, semper vel orci sed, auctor mattis arcu. Integer at gravida dui, id varius nisl. Duis eu ultrices est, non eleifend mauris. Donec fringilla suscipit egestas. Praesent bibendum lectus leo, sed ornare arcu iaculis sed. Sed vulputate nunc nec augue luctus, quis mollis ipsum ullamcorper."
"Fusce molestie aliquet ipsum at placerat.Cras eu varius justo. Fusce ac malesuada nisi. Aliquam fringilla bibendum magna et hendrerit. Nunc quis vestibulum felis, sed auctor odio. Quisque in justo et lectus iaculis pretium nec non quam. Proin ligula ligula, accumsan sed quam tincidunt, vehicula sodales massa. Integer tempus vulputate leo, vel aliquet ipsum elementum vel. Nulla egestas dolor non diam eleifend,"
" vel dapibus odio tincidunt. Fusce pretium nulla vel gravida dapibus. Sed sit amet viverra orci. Aliquam erat volutpat. Proin viverra orci efficitur, consectetur neque a, vestibulum justo.In sit amet mollis dolor. Nulla at felis at tellus suscipit luctus. Donec risus neque, pulvinar id ex id, elementum rutrum leo. Morbi libero ex, elementum et tellus sit amet, tempus iaculis lectus. Pellentesque pretium nisi nisl,"
" eu blandit justo commodo ac. Morbi id elit sagittis, aliquam arcu vel, hendrerit felis. Vivamus elementum pulvinar congue. Fusce tincidunt consequat aliquam.Mauris lacinia massa massa, id lacinia orci volutpat at. Vestibulum tellus ante, dictum ut lectus nec, tempor porttitor erat. Donec at felis eget turpis cursus tempus sit amet id erat. Aliquam varius viverra urna in elementum. Sed eu odio lacus. "
"Proin interdum consequat fringilla. Ut placerat erat eu dignissim pellentesque. Nam accumsan mi non lacus congue venenatis. Etiam et dictum sem. Praesent justo neque, egestas id maximus quis, malesuada at eros. Curabitur vitae tortor aliquam, ornare urna in, accumsan tortor. In non volutpat orci, sed pulvinar tellus. In tincidunt ac orci id scelerisque.Suspendisse eleifend tellus vel nisi pretium, id faucibus turpis "
"pulvinar. Quisque sagittis justo in dapibus dapibus. Quisque id nibh eu dui sagittis venenatis vehicula vel enim. Sed non commodo augue. Donec eu justo sit amet tellus bibendum commodo vitae at nunc. Maecenas dui est, eleifend vitae metus vitae, pretium sodales lectus. Mauris rutrum, nibh ac tempor ultricies, mauris turpis semper metus, sed auctor ipsum ex quis tellus. In ultricies dapibus dui, condimentum imperdiet nulla tincidunt ac."
" Aenean commodo et tortor ac efficitur. Curabitur tempor lacinia ex aliquet varius. Mauris dapibus diam nunc, vitae ornare urna accumsan a. Pellentesque aliquam leo tellus, ut finibus enim elementum quis. Sed tincidunt tortor eu lacinia scelerisque. Nam lacinia nibh dui, vitae suscipit dolor scelerisque sit amet. Phasellus sit amet lobortis odio, eu lacinia lectus. Proin mattis sit amet felis nec vulputate.";

static const char* testringMatches = "This is a comparison string that matches the other string perfectly";
static const char* teststringNotMatches0 = "This is a comparison string that does not match the other string perfectly";
static const char* teststringNotMatches1 = "This is a comparison string that seod not match the other string perfectly";
static const char* teststringNotMatchesShort = "This is a comparison string that doesn't match the other string perfectly";

static const char* teststringEmpty = "";


TEST(StringOps, Strlen) {
	// Matching
	const size_t lenReference = std::strlen(testringMatches);
	const size_t lenActual = wlib::str::strlen(testringMatches);
	EXPECT_EQ(lenReference, lenActual) << "String lengths do not match";

	// Matching large size
	const size_t lenReference_vlong = std::strlen(loremIpsum);
	const size_t lenActual_vlong = wlib::str::strlen(loremIpsum);
	EXPECT_EQ(lenReference_vlong, lenActual_vlong) << "String lengths do not match";

	// Zero-length matches
	const size_t lenReference_zeroLen = std::strlen(teststringEmpty);
	const size_t lenActual_zeroLen = wlib::str::strlen(teststringEmpty);
	EXPECT_EQ(lenReference_zeroLen, lenActual_zeroLen) << "String lengths do not match";
}


TEST(StringOps, Strcmp) {
	// Matching
	const char* str0 = testringMatches;
	const char* str1 = testringMatches;
	EXPECT_TRUE(wlib::str::strcmp(str0, str1)) << "The string comparison failed";

	// Not matching
	const char* str0_ne = teststringNotMatches0;
	const char* str1_ne = teststringNotMatches1;
	EXPECT_FALSE(wlib::str::strcmp(str0_ne, str1_ne)) << "The string comparison did not fail";

	// Different lengths
	const char* str0_difflen = teststringNotMatches0;
	const char* str1_difflen = teststringNotMatchesShort;
	EXPECT_FALSE(wlib::str::strcmp(str0_difflen, str1_difflen)) << "The string comparison did not fail";

	// Left empty
	const char* str0_empty0 = teststringEmpty;
	const char* str1_empty0 = testringMatches;
	EXPECT_FALSE(wlib::str::strcmp(str0_empty0, str1_empty0)) << "The string comparison did not fail";

	// Right empty
	const char* str0_empty1 = testringMatches;
	const char* str1_empty1 = teststringEmpty;
	EXPECT_FALSE(wlib::str::strcmp(str0_empty1, str1_empty1)) << "The string comparison did not fail";

	// Both empty
	const char* str0_empty2 = teststringEmpty;
	const char* str1_empty2 = teststringEmpty;
	EXPECT_TRUE(wlib::str::strcmp(str0_empty2, str1_empty2)) << "The string comparison failed";

	// Matching strncmp
	const char* str0_ncmpeq = testringMatches;
	const char* str1_ncmpeq = testringMatches;
	EXPECT_TRUE(wlib::str::strncmp(str0_ncmpeq, str1_ncmpeq, strlen(str0_ncmpeq))) << "The string comparison failed";

	// Matching strncmp with different length
	const char* str0_eqdiff = teststringNotMatches0;
	const char* str1_eqdiff = teststringNotMatches1;
	EXPECT_TRUE(wlib::str::strncmp(str0_eqdiff, str1_eqdiff, 32u)) << "The string comparison failed";

	// Not matching
	const char* str0_neq = teststringNotMatches0;
	const char* str1_neq = teststringNotMatches1;
	EXPECT_FALSE(wlib::str::strncmp(str0_neq, str1_neq, 40u)) << "The string comparison did not fail";

	// From constchar
	EXPECT_TRUE(wlib::str::strncmp(testringMatches, testringMatches, wlib::str::strlen(testringMatches))) << "The string comparison failed";
}

TEST(StringOps, Strstr) {
	// No match
	const auto reference_nomatch = std::strstr(loremIpsum, "nomatchneedle");
	const auto actual_nomatch = wlib::str::strstr(loremIpsum, "nomatchneedle");
	EXPECT_EQ(reference_nomatch, actual_nomatch) << "Substring search results do not match";

	// Matching large size
	const auto reference_vlong = std::strstr(loremIpsum, "varius");
	const auto actual_vlong = wlib::str::strstr(loremIpsum, "varius");
	EXPECT_EQ(reference_vlong, actual_vlong) << "Substring search results do not match";

	// Zero-length matches
	const auto reference_zeroLen = std::strstr(teststringEmpty, "");
	const auto actual_zeroLen = wlib::str::strstr(teststringEmpty, "");
	EXPECT_EQ(reference_zeroLen, actual_zeroLen) << "Substring search results do not match";
}

// TODO: Mark as false if had unexpected char
TEST(StringOps, Parse_Int) {
	std::string sIntNegStr = "-32451";
	std::string sIntPosStr = "32451";
	std::string uIntStr = "51228";
	std::string invalidStrStr = "test";
	std::string earlyStopStr = "3245e4r1yst0p";

	int32_t sIntNeg;
	uint32_t sIntInUInt;
	int32_t sIntPos;
	uint32_t uInt;
	uint32_t invalidStr;
	int32_t earlyStopSInt;
	uint32_t earlyStopUInt;

	EXPECT_TRUE(wlib::str::ParseString(sIntNegStr, sIntNeg));
	EXPECT_TRUE(wlib::str::ParseString(sIntPosStr, sIntPos));
	EXPECT_TRUE(wlib::str::ParseString(uIntStr, uInt));
	EXPECT_TRUE(wlib::str::ParseString(sIntNegStr, sIntNeg));
	EXPECT_TRUE(wlib::str::ParseString(earlyStopStr, earlyStopUInt));
	EXPECT_TRUE(wlib::str::ParseString(earlyStopStr, earlyStopSInt));

	EXPECT_FALSE(wlib::str::ParseString(sIntNegStr, sIntInUInt));
	EXPECT_FALSE(wlib::str::ParseString(invalidStrStr, invalidStr));

	EXPECT_EQ(sIntNeg, -32451);
	EXPECT_EQ(sIntPos, 32451);
	EXPECT_EQ(uInt, 51228);
	EXPECT_EQ(earlyStopSInt, 3245);
	EXPECT_EQ(earlyStopUInt, 3245);
}

TEST(StringOps, Parse_Bool) {
	std::string yesStr = "Yes";
	std::string truStr = "True";
	std::string tStr = "t";
	std::string yStr = "y";

	EXPECT_TRUE(wlib::str::ParseBool(yesStr));
	EXPECT_TRUE(wlib::str::ParseBool(truStr));
	EXPECT_TRUE(wlib::str::ParseBool(tStr));
	EXPECT_TRUE(wlib::str::ParseBool(yStr));

	EXPECT_TRUE(wlib::str::ParseBool(" Yes "));
	EXPECT_FALSE(wlib::str::ParseBool("random"));
	EXPECT_FALSE(wlib::str::ParseBool("no"));
	EXPECT_FALSE(wlib::str::ParseBool("false"));
}

TEST(StringOps, Parse_Float) {
	std::string fltStr = "3.14";
	std::string fltNegStr = "-3.14";

	float flt;
	double dbl;
	long double ldbl;

	float fltNeg;
	double dblNeg;
	long double ldblNeg;

	wlib::str::ParseString(fltStr, flt);
	wlib::str::ParseString(fltStr, dbl);
	wlib::str::ParseString(fltStr, ldbl);

	wlib::str::ParseString(fltNegStr, fltNeg);
	wlib::str::ParseString(fltNegStr, dblNeg);
	wlib::str::ParseString(fltNegStr, ldblNeg);

	EXPECT_FLOAT_EQ(flt, 3.14f);
	EXPECT_FLOAT_EQ(dbl, 3.14);
	EXPECT_FLOAT_EQ(ldbl, 3.14L);

	EXPECT_FLOAT_EQ(fltNeg, -3.14f);
	EXPECT_FLOAT_EQ(dblNeg, -3.14);
	EXPECT_FLOAT_EQ(ldblNeg, -3.14L);
}

#endif //WEIRDLIB_ENABLE_STRING_OPERATIONS
