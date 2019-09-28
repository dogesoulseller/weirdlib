#include <gtest/gtest.h>
#include "../include/weirdlib.hpp"
#include <iostream>

//TODO: Test erase
TEST(Containers, UnorderedFlatMap) {
	wlib::unordered_flat_map<std::string, std::string> flatmap;
	wlib::unordered_flat_map<uint32_t, std::string> flatmap_simple;

	// Insertion
	EXPECT_NO_FATAL_FAILURE(flatmap.insert("test", "also_test"));
	EXPECT_NO_FATAL_FAILURE(flatmap_simple.insert(3, "also_test"));

	EXPECT_EQ(flatmap["test"], "also_test");
	EXPECT_EQ(flatmap_simple[3], "also_test");

	// Size check
	EXPECT_EQ(flatmap.size(), 1);
	EXPECT_EQ(flatmap_simple.size(), 1);

	// Capacity check
	EXPECT_EQ(flatmap.capacity(), 1);
	EXPECT_EQ(flatmap_simple.capacity(), 1);

	// Repeated insertion should not change value or size
	EXPECT_NO_FATAL_FAILURE(flatmap.insert("test", "no_work"));
	EXPECT_NO_FATAL_FAILURE(flatmap_simple.insert(3, "no_work"));

	EXPECT_EQ(flatmap["test"], "also_test");
	EXPECT_EQ(flatmap_simple[3], "also_test");

	EXPECT_EQ(flatmap.size(), 1);
	EXPECT_EQ(flatmap_simple.size(), 1);

	EXPECT_EQ(flatmap.capacity(), 1);
	EXPECT_EQ(flatmap_simple.capacity(), 1);

	// Insertion / assignment should change value, but not size
	EXPECT_NO_FATAL_FAILURE(flatmap.insert_or_assign("test", "different_test"));
	EXPECT_NO_FATAL_FAILURE(flatmap_simple.insert_or_assign(3, "different_test"));

	EXPECT_EQ(flatmap["test"], "different_test");
	EXPECT_EQ(flatmap_simple[3], "different_test");

	EXPECT_EQ(flatmap.size(), 1);
	EXPECT_EQ(flatmap_simple.size(), 1);

	EXPECT_EQ(flatmap.capacity(), 1);
	EXPECT_EQ(flatmap_simple.capacity(), 1);

	// Accessing nonexistent element should throw out_of_range exception
	EXPECT_THROW(flatmap["test_noexist"], std::out_of_range);
	EXPECT_THROW(flatmap_simple[4], std::out_of_range);

	// Clear all should erase all elements and set new size, but not reset capacity
	EXPECT_NO_FATAL_FAILURE(flatmap.clear());
	EXPECT_NO_FATAL_FAILURE(flatmap_simple.clear());

	EXPECT_THROW(flatmap["test"], std::out_of_range);
	EXPECT_THROW(flatmap_simple[3], std::out_of_range);

	EXPECT_EQ(flatmap.size(), 0);
	EXPECT_EQ(flatmap_simple.size(), 0);

	EXPECT_EQ(flatmap.capacity(), 1);
	EXPECT_EQ(flatmap_simple.capacity(), 1);

	// Reserve should set new capacity, but not size
	EXPECT_NO_FATAL_FAILURE(flatmap.reserve(3));
	EXPECT_NO_FATAL_FAILURE(flatmap_simple.reserve(3));

	EXPECT_EQ(flatmap.size(), 0);
	EXPECT_EQ(flatmap_simple.size(), 0);

	EXPECT_EQ(flatmap.capacity(), 3);
	EXPECT_EQ(flatmap_simple.capacity(), 3);

	// Shrink to fit should merge size and capacity
	EXPECT_NO_FATAL_FAILURE(flatmap.insert("test", "also_test"));
	EXPECT_NO_FATAL_FAILURE(flatmap_simple.insert(3, "also_test"));
	EXPECT_NO_FATAL_FAILURE(flatmap.insert("test0", "also_test0"));
	EXPECT_NO_FATAL_FAILURE(flatmap_simple.insert(5, "also_test0"));

	ASSERT_EQ(flatmap.size(), 2);
	ASSERT_EQ(flatmap_simple.size(), 2);

	ASSERT_EQ(flatmap.capacity(), 3);
	ASSERT_EQ(flatmap_simple.capacity(), 3);

	EXPECT_NO_FATAL_FAILURE(flatmap.shrink_to_fit());
	EXPECT_NO_FATAL_FAILURE(flatmap_simple.shrink_to_fit());

	EXPECT_EQ(flatmap.size(), 2);
	EXPECT_EQ(flatmap_simple.size(), 2);

	EXPECT_EQ(flatmap.capacity(), 2);
	EXPECT_EQ(flatmap_simple.capacity(), 2);

	// Erase
	EXPECT_NO_FATAL_FAILURE(flatmap.erase("test0"));
	EXPECT_NO_FATAL_FAILURE(flatmap_simple.erase(5));

	EXPECT_EQ(flatmap.size(), 1);
	EXPECT_EQ(flatmap_simple.size(), 1);

	EXPECT_EQ(flatmap.capacity(), 2);
	EXPECT_EQ(flatmap_simple.capacity(), 2);
}

TEST(Containers, UnorderedFlatMap_initlist) {
	// Should work for any type that looks like a 2-tuple
	wlib::unordered_flat_map<std::string, std::string> flatmap_initlist_pair = {std::pair{"test", "test"}, std::pair{"test", "test"}, std::pair{"test0", "test0"}};
	wlib::unordered_flat_map<std::string, std::string> flatmap_initlist_tuple = {std::make_tuple("test", "test"), std::make_tuple("test", "test"), std::make_tuple("test0", "test0")};

	EXPECT_EQ(flatmap_initlist_pair["test"], "test");
	EXPECT_EQ(flatmap_initlist_tuple["test"], "test");

	EXPECT_EQ(flatmap_initlist_pair["test0"], "test0");
	EXPECT_EQ(flatmap_initlist_tuple["test0"], "test0");

	EXPECT_EQ(flatmap_initlist_pair.size(), 2);
	EXPECT_EQ(flatmap_initlist_tuple.size(), 2);

	EXPECT_EQ(flatmap_initlist_pair.capacity(), 2);
	EXPECT_EQ(flatmap_initlist_tuple.capacity(), 2);

}
