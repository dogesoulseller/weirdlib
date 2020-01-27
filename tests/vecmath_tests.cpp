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

TEST(Vecmath, Normalize) {
	Vector2 vec2(1.0f, 2.0f);
	Vector2 vec2d(1.0, 2.0);

	Vector3 vec3(1.0f, 2.0f, 3.0f);
	Vector3 vec3d(1.0, 2.0, 3.0);

	Vector4 vec4(1.0f, 2.0f, 3.0f, 4.0f);
	Vector4 vec4d(1.0, 2.0, 3.0, 4.0);

	auto vec2_norm = Normalize(vec2);
	auto vec2d_norm = Normalize(vec2d);
	auto vec3_norm = Normalize(vec3);
	auto vec3d_norm = Normalize(vec3d);
	auto vec4_norm = Normalize(vec4);
	auto vec4d_norm = Normalize(vec4d);

	auto vec2_norm_approx = NormalizeApprox(vec2);
	auto vec3_norm_approx = NormalizeApprox(vec3);
	auto vec4_norm_approx = NormalizeApprox(vec4);

	EXPECT_FLOAT_EQ(Length(vec2_norm), 1.0f);
	EXPECT_DOUBLE_EQ(Length(vec2d_norm), 1.0);
	EXPECT_FLOAT_EQ(Length(vec3_norm), 1.0f);
	EXPECT_DOUBLE_EQ(Length(vec3d_norm), 1.0);
	EXPECT_FLOAT_EQ(Length(vec4_norm), 1.0f);
	EXPECT_DOUBLE_EQ(Length(vec4d_norm), 1.0);

	EXPECT_FLOAT_EQ(Length(vec2_norm_approx), 0.99997985f);
	EXPECT_FLOAT_EQ(Length(vec3_norm_approx), 1.0000437f);
	EXPECT_FLOAT_EQ(Length(vec4_norm_approx), 1.0000683f);
}

TEST(Vecmath, Reflect) {
	Vector2 vec2i(1.0f, 2.0f);
	Vector2 vec2n(432.0f, 832.0f);

	Vector2 vec2id(1.0, 2.0);
	Vector2 vec2nd(432.0, 832.0);

	Vector3 vec3i(1.0f, 2.0f, 3.0f);
	Vector3 vec3n(432.0f, 832.0f, 1232.0f);

	Vector3 vec3id(1.0, 2.0, 3.0);
	Vector3 vec3nd(432.0, 832.0, 1232.0);

	Vector4 vec4i(1.0f, 2.0f, 3.0f, 4.0f);
	Vector4 vec4n(432.0f, 832.0f, 1232.0f, 1632.0f);

	Vector4 vec4id(1.0, 2.0, 3.0, 4.0);
	Vector4 vec4nd(432.0, 832.0, 1232.0, 1632.0);

	auto vec2reflection = Reflect(vec2i, Normalize(vec2n));
	auto vec2reflectiond = Reflect(vec2id, Normalize(vec2nd));

	auto vec3reflection = Reflect(vec3i, Normalize(vec3n));
	auto vec3reflectiond = Reflect(vec3id, Normalize(vec3nd));

	auto vec4reflection = Reflect(vec4i, Normalize(vec4n));
	auto vec4reflectiond = Reflect(vec4id, Normalize(vec4nd));

	EXPECT_FLOAT_EQ(vec2reflection.x, -1.0605884f);
	EXPECT_FLOAT_EQ(vec2reflection.y, -1.9685404f);

	EXPECT_DOUBLE_EQ(vec2reflectiond.x, -1.0605884066414211);
	EXPECT_DOUBLE_EQ(vec2reflectiond.y, -1.9685406350131078);

	EXPECT_FLOAT_EQ(vec3reflection.x, -1.0880151f);
	EXPECT_FLOAT_EQ(vec3reflection.y, -2.0213628f);
	EXPECT_FLOAT_EQ(vec3reflection.z, -2.95471f);

	EXPECT_DOUBLE_EQ(vec3reflectiond.x, -1.0880153813287756);
	EXPECT_DOUBLE_EQ(vec3reflectiond.y, -2.0213629566331983);
	EXPECT_DOUBLE_EQ(vec3reflectiond.z, -2.9547105319376197);

	EXPECT_FLOAT_EQ(vec4reflection.x, -1.1036122f);
	EXPECT_FLOAT_EQ(vec4reflection.y, -2.0514016f);
	EXPECT_FLOAT_EQ(vec4reflection.z, -2.9991903f);
	EXPECT_FLOAT_EQ(vec4reflection.w, -3.9469795f);

	EXPECT_DOUBLE_EQ(vec4reflectiond.x, -1.1036122634827477);
	EXPECT_DOUBLE_EQ(vec4reflectiond.y, -2.0514013963371438);
	EXPECT_DOUBLE_EQ(vec4reflectiond.z, -2.9991905291915391);
	EXPECT_DOUBLE_EQ(vec4reflectiond.w, -3.9469796620459352);
}

TEST(Vecmath, Refract) {
	float eta = 3.0f;
	double etad = 3.0;

	Vector2 vec2i(1.0f, 2.0f);
	Vector2 vec2n(432.0f, 832.0f);

	Vector2 vec2id(1.0, 2.0);
	Vector2 vec2nd(432.0, 832.0);

	Vector3 vec3i(1.0f, 2.0f, 3.0f);
	Vector3 vec3n(432.0f, 832.0f, 1232.0f);

	Vector3 vec3id(1.0, 2.0, 3.0);
	Vector3 vec3nd(432.0, 832.0, 1232.0);

	Vector4 vec4i(1.0f, 2.0f, 3.0f, 4.0f);
	Vector4 vec4n(432.0f, 832.0f, 1232.0f, 1632.0f);

	Vector4 vec4id(1.0, 2.0, 3.0, 4.0);
	Vector4 vec4nd(432.0, 832.0, 1232.0, 1632.0);

	auto vec2refraction = Refract(Normalize(vec2i), Normalize(vec2n), eta);
	auto vec2refractiond = Refract(Normalize(vec2id), Normalize(vec2nd), etad);

	auto vec3refraction = Refract(Normalize(vec3i), Normalize(vec3n), eta);
	auto vec3refractiond = Refract(Normalize(vec3id), Normalize(vec3nd), etad);

	auto vec4refraction = Refract(Normalize(vec4i), Normalize(vec4n), eta);
	auto vec4refractiond = Refract(Normalize(vec4id), Normalize(vec4nd), etad);

	EXPECT_FLOAT_EQ(vec2refraction.x, -0.50097549f);
	EXPECT_FLOAT_EQ(vec2refraction.y, -0.86546063f);

	EXPECT_DOUBLE_EQ(vec2refractiond.x, -0.50097572392144785);
	EXPECT_DOUBLE_EQ(vec2refractiond.y, -0.86546133595983488);

	EXPECT_FLOAT_EQ(vec3refraction.x, -0.31410253f);
	EXPECT_FLOAT_EQ(vec3refraction.y, -0.54554677f);
	EXPECT_FLOAT_EQ(vec3refraction.z, -0.77699113f);

	EXPECT_DOUBLE_EQ(vec3refractiond.x, -0.3141029611661329);
	EXPECT_DOUBLE_EQ(vec3refractiond.y, -0.54554764922831001);
	EXPECT_LE(vec3refractiond.z, -0.7769923372903);
	EXPECT_GE(vec3refractiond.z, -0.7769923372906);

	EXPECT_FLOAT_EQ(vec4refraction.x, -0.22030413f);
	EXPECT_FLOAT_EQ(vec4refraction.y, -0.38371742f);
	EXPECT_FLOAT_EQ(vec4refraction.z, -0.54713082f);
	EXPECT_FLOAT_EQ(vec4refraction.w, -0.71054387f);

	EXPECT_DOUBLE_EQ(vec4refractiond.x, -0.22030436328164349);
	EXPECT_DOUBLE_EQ(vec4refractiond.y, -0.38371784354204186);
	EXPECT_DOUBLE_EQ(vec4refractiond.z, -0.54713132380244001);
	EXPECT_DOUBLE_EQ(vec4refractiond.w, -0.7105448040628386);
}