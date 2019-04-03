#include <gtest/gtest.h>
#include <cstring>
#include "../include/weirdlib.hpp"

const char* lorem_ipsum = "Lorem ipsum dolor sit amet, consectetur adipiscing elit."
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

const char* teststring_matches = "This is a comparison string that matches the other string perfectly";
const char* teststring_not_matches0 = "This is a comparison string that does not match the other string perfectly";
const char* teststring_not_matches1 = "This is a comparison string that seod not match the other string perfectly";
const char* teststring_not_matches_short = "This is a comparison string that doesn't match the other string perfectly";

const char* teststring_zerolen = "";

TEST(StringOps, strlen) {
	const size_t len_reference = std::strlen(teststring_matches);
	const size_t len_actual = wlib::strlen(teststring_matches);

	EXPECT_EQ(len_reference, len_actual) << "String lengths do not match";
}

TEST(StringOps, strlen_verylong) {
	const size_t len_reference = std::strlen(lorem_ipsum);
	const size_t len_actual = wlib::strlen(lorem_ipsum);

	EXPECT_EQ(len_reference, len_actual) << "String lengths do not match";
}

TEST(StringOps, strlen_empty) {
	const size_t len_reference = std::strlen(teststring_zerolen);
	const size_t len_actual = wlib::strlen(teststring_zerolen);

	EXPECT_EQ(len_reference, len_actual) << "String lengths do not match";
}

TEST(StringOps, strcmp_equal) {
	const char* str0 = teststring_matches;
	const char* str1 = teststring_matches;
	EXPECT_TRUE(wlib::strcmp(str0, str1)) << "The string comparison failed";
}

TEST(StringOps, strcmp_not_equal) {
	const char* str0 = teststring_not_matches0;
	const char* str1 = teststring_not_matches1;
	EXPECT_FALSE(wlib::strcmp(str0, str1)) << "The string comparison did not fail";
}

TEST(StringOps, strcmp_difflen) {
	const char* str0 = teststring_not_matches0;
	const char* str1 = teststring_not_matches_short;
	EXPECT_FALSE(wlib::strcmp(str0, str1)) << "The string comparison did not fail";
}

TEST(StringOps, strcmp_empty_0) {
	const char* str0 = teststring_zerolen;
	const char* str1 = teststring_matches;
	EXPECT_FALSE(wlib::strcmp(str0, str1)) << "The string comparison did not fail";
}

TEST(StringOps, strcmp_empty_1) {
	const char* str0 = teststring_matches;
	const char* str1 = teststring_zerolen;
	EXPECT_FALSE(wlib::strcmp(str0, str1)) << "The string comparison did not fail";
}

TEST(StringOps, strcmp_empty_both) {
	const char* str0 = teststring_zerolen;
	const char* str1 = teststring_zerolen;
	EXPECT_TRUE(wlib::strcmp(str0, str1)) << "The string comparison failed";
}

TEST(StringOps, strncmp_equal) {
	const char* str0 = teststring_matches;
	const char* str1 = teststring_matches;
	EXPECT_TRUE(wlib::strncmp(str0, str1, strlen(str0))) << "The string comparison failed";
}

TEST(StringOps, strncmp_equal_different) {
	const char* str0 = teststring_not_matches0;
	const char* str1 = teststring_not_matches1;
	EXPECT_TRUE(wlib::strncmp(str0, str1, 32u)) << "The string comparison failed";
}

TEST(StringOps, strncmp_not_equal) {
	const char* str0 = teststring_not_matches0;
	const char* str1 = teststring_not_matches1;
	EXPECT_FALSE(wlib::strncmp(str0, str1, 40u)) << "The string comparison did not fail";
}

TEST(StringOps, strncmp_from_constchar) {
	EXPECT_TRUE(wlib::strncmp(teststring_matches, teststring_matches, wlib::strlen(teststring_matches))) << "The string comparison failed";
}
