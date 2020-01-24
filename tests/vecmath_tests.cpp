#include <gtest/gtest.h>
#include "../include/weirdlib.hpp"

using namespace wlib::vecmath;

TEST(Vecmath, VectorInitialize) {
	Vector2 vec2(1.0f, 2.0f);
	Vector2 vec2d(1.0, 2.0);

	EXPECT_FLOAT_EQ(vec2.x, 1.0f);
	EXPECT_FLOAT_EQ(vec2.y, 2.0f);
	EXPECT_DOUBLE_EQ(vec2d.x, 1.0);
	EXPECT_DOUBLE_EQ(vec2d.y, 2.0);


	Vector3 vec3{1.0f, 2.0f, 3.0f};
	Vector3 vec3d{1.0, 2.0, 3.0};

	EXPECT_FLOAT_EQ(vec3.x, 1.0f);
	EXPECT_FLOAT_EQ(vec3.y, 2.0f);
	EXPECT_FLOAT_EQ(vec3.z, 3.0f);
	EXPECT_DOUBLE_EQ(vec3d.x, 1.0);
	EXPECT_DOUBLE_EQ(vec3d.y, 2.0);
	EXPECT_DOUBLE_EQ(vec3d.z, 3.0);


	Vector4 vec4{1.0f, 2.0f, 3.0f, 4.0f};
	Vector4 vec4d{1.0, 2.0, 3.0, 4.0};

	EXPECT_FLOAT_EQ(vec4.x, 1.0f);
	EXPECT_FLOAT_EQ(vec4.y, 2.0f);
	EXPECT_FLOAT_EQ(vec4.z, 3.0f);
	EXPECT_FLOAT_EQ(vec4.w, 4.0f);
	EXPECT_DOUBLE_EQ(vec4d.x, 1.0);
	EXPECT_DOUBLE_EQ(vec4d.y, 2.0);
	EXPECT_DOUBLE_EQ(vec4d.z, 3.0);
	EXPECT_DOUBLE_EQ(vec4d.w, 4.0);
}

TEST(Vecmath, DotProduct) {
	Vector2 vec2(1.0f, 2.0f);
	Vector3 vec3(1.0f, 2.0f, 3.0f);
	Vector4 vec4(1.0f, 2.0f, 3.0f, 4.0f);

	Vector2 vec2d(1.0, 2.0);
	Vector3 vec3d(1.0, 2.0, 3.0);
	Vector4 vec4d(1.0, 2.0, 3.0, 4.0);


	EXPECT_FLOAT_EQ(DotProduct(vec2, vec2), 5.0f);
	EXPECT_FLOAT_EQ(DotProduct(vec3, vec3), 14.0f);
	EXPECT_FLOAT_EQ(DotProduct(vec4, vec4), 30.0f);

	EXPECT_DOUBLE_EQ(DotProduct(vec2d, vec2d), 5.0);
	EXPECT_DOUBLE_EQ(DotProduct(vec3d, vec3d), 14.0);
	EXPECT_DOUBLE_EQ(DotProduct(vec4d, vec4d), 30.0);
}

TEST(Vecmath, CrossProduct) {
	Vector3 vec3_0(6.0f, 9.0f, 12.0f);
	Vector3 vec3_1(8.0f, 10.0f, 13.0f);

	Vector3 vec3d_0(6.0, 9.0, 12.0);
	Vector3 vec3d_1(8.0, 10.0, 13.0);

	auto resultFlt = CrossProduct(vec3_0, vec3_1);
	auto resultDbl = CrossProduct(vec3d_0, vec3d_1);

	EXPECT_FLOAT_EQ(resultFlt.x, -3.0f);
	EXPECT_FLOAT_EQ(resultFlt.y, 18.0f);
	EXPECT_FLOAT_EQ(resultFlt.z, -12.0f);

	EXPECT_DOUBLE_EQ(resultDbl.x, -3.0);
	EXPECT_DOUBLE_EQ(resultDbl.y, 18.0);
	EXPECT_DOUBLE_EQ(resultDbl.z, -12.0);
}

TEST(Vecmath, Distance) {
	Vector4 vec4_0(6.0f, 9.0f, 12.0f, 15.0f);
	Vector4 vec4_1(8.0f, 10.0f, 12.0f, 18.0f);

	Vector4 vec4d_0(6.0, 9.0, 12.0, 15.0);
	Vector4 vec4d_1(8.0, 10.0, 12.0, 18.0);

	auto resultFlt4 = Distance(vec4_0, vec4_1);
	auto resultDbl4 = Distance(vec4d_0, vec4d_1);

	EXPECT_FLOAT_EQ(resultFlt4, 3.7416575f);
	EXPECT_DOUBLE_EQ(resultDbl4, 3.7416573867739413);



	Vector3 vec3_0(6.0f, 9.0f, 12.0f);
	Vector3 vec3_1(8.0f, 10.0f, 12.0f);

	Vector3 vec3d_0(6.0, 9.0, 12.0);
	Vector3 vec3d_1(8.0, 10.0, 12.0);

	auto resultFlt3 = Distance(vec3_0, vec3_1);
	auto resultDbl3 = Distance(vec3d_0, vec3d_1);

	EXPECT_FLOAT_EQ(resultFlt3, 2.236068f);
	EXPECT_DOUBLE_EQ(resultDbl3, 2.2360679774997898);



	Vector2 vec2_0(6.0f, 9.0f);
	Vector2 vec2_1(8.0f, 10.0f);

	Vector2 vec2d_0(6.0, 9.0);
	Vector2 vec2d_1(8.0, 10.0);

	auto resultFlt2 = Distance(vec2_0, vec2_1);
	auto resultDbl2 = Distance(vec2d_0, vec2d_1);

	EXPECT_FLOAT_EQ(resultFlt2, 2.236068f);
	EXPECT_DOUBLE_EQ(resultDbl2, 2.2360679774997898);
}

TEST(Vecmath, Length) {
	Vector2 vec2(1.0f, 2.0f);
	Vector2 vec2d(1.0, 2.0);

	Vector3 vec3(1.0f, 2.0f, 3.0f);
	Vector3 vec3d(1.0, 2.0, 3.0);

	Vector4 vec4(1.0f, 2.0f, 3.0f, 4.0f);
	Vector4 vec4d(1.0, 2.0, 3.0, 4.0);

	EXPECT_FLOAT_EQ(Length(vec2), 2.23606797749978969640917366f);
	EXPECT_DOUBLE_EQ(Length(vec2d), 2.23606797749978969640917366);

	EXPECT_FLOAT_EQ(Length(vec3), 3.74165738677394138558374872f);
	EXPECT_DOUBLE_EQ(Length(vec3d), 3.74165738677394138558374832);

	EXPECT_FLOAT_EQ(Length(vec4), 5.47722557505166113456969728f);
	EXPECT_DOUBLE_EQ(Length(vec4d), 5.47722557505166113456997828);
}