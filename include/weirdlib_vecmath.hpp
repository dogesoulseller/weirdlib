#pragma once
#include <type_traits>

namespace wlib::vecmath
{
	template<typename FltT = float, typename = std::enable_if_t<std::is_floating_point_v<FltT>>>
	class Vector2
	{
	  public:
		FltT x;
		FltT y;
		Vector2() = default;
		Vector2(const Vector2&) = default;
		Vector2(Vector2&&) = default;

		inline Vector2(FltT val) : x{val}, y{val} {}
		inline Vector2(FltT _x, FltT _y) : x{_x}, y{_y} {}

		Vector2& operator=(const Vector2&) = default;
		Vector2& operator=(Vector2&&) = default;

		friend inline Vector2 operator*(FltT lhs, const Vector2& rhs) {
			return Vector2(rhs.x * lhs, rhs.y * lhs);
		}

		friend inline Vector2 operator*(const Vector2& lhs, FltT rhs) {
			return Vector2(lhs.x * rhs, lhs.y * rhs);
		}

		inline Vector2& operator*=(FltT rhs) {
			x *= rhs; y *= rhs;
			return *this;
		}

		inline Vector2<double> ToDoubles() {return Vector2<double>(static_cast<double>(x), static_cast<double>(y));};
		inline Vector2<float> ToFloats() {return Vector2<float>(static_cast<float>(x), static_cast<float>(y));};
	};

	template<typename FltT = float, typename = std::enable_if_t<std::is_floating_point_v<FltT>>>
	class Vector3
	{
	  public:
		FltT x;
		FltT y;
		FltT z;
		Vector3() = default;
		Vector3(const Vector3&) = default;
		Vector3(Vector3&&) = default;

		inline Vector3(FltT val) : x{val}, y{val}, z{val} {}
		inline Vector3(FltT _x, FltT _y, FltT _z) : x{_x}, y{_y}, z{_z} {}

		Vector3& operator=(const Vector3&) = default;
		Vector3& operator=(Vector3&&) = default;

		friend inline Vector3 operator*(FltT lhs, const Vector3& rhs) {
			return Vector3(rhs.x * lhs, rhs.y * lhs, rhs.z * lhs);
		}

		friend inline Vector3 operator*(const Vector3& lhs, FltT rhs) {
			return Vector3(lhs.x * rhs, lhs.y * rhs, lhs.z * rhs);
		}

		inline Vector3& operator*=(FltT rhs) {
			x *= rhs; y *= rhs; z *= rhs;
			return *this;
		}

		inline Vector3<double> ToDoubles() {return Vector3<double>(static_cast<double>(x), static_cast<double>(y), static_cast<double>(z));};
		inline Vector3<float> ToFloats() {return Vector3<float>(static_cast<float>(x), static_cast<float>(y), static_cast<float>(z));};
	};

	template<typename FltT = float, typename = std::enable_if_t<std::is_floating_point_v<FltT>>>
	class Vector4
	{
	  public:
		FltT x;
		FltT y;
		FltT z;
		FltT w;
		Vector4() = default;
		Vector4(const Vector4&) = default;
		Vector4(Vector4&&) = default;

		inline Vector4(FltT val) : x{val}, y{val}, z{val}, w{val} {}
		inline Vector4(FltT _x, FltT _y, FltT _z, FltT _w) : x{_x}, y{_y}, z{_z}, w{_w} {}

		Vector4& operator=(const Vector4&) = default;
		Vector4& operator=(Vector4&&) = default;

		friend inline Vector4 operator*(FltT lhs, const Vector4& rhs) {
			return Vector4(rhs.x * lhs, rhs.y * lhs, rhs.z * lhs, rhs.w * lhs);
		}

		friend inline Vector4 operator*(const Vector4& lhs, FltT rhs) {
			return Vector4(lhs.x * rhs, lhs.y * rhs, lhs.z * rhs, lhs.w * rhs);
		}

		inline Vector4& operator*=(FltT rhs) {
			x *= rhs; y *= rhs; z *= rhs; w *= rhs;
			return *this;
		}

		inline Vector4<double> ToDoubles() {return Vector4<double>(static_cast<double>(x), static_cast<double>(y), static_cast<double>(z), static_cast<double>(w));};
		inline Vector4<float> ToFloats() {return Vector4<float>(static_cast<float>(x), static_cast<float>(y), static_cast<float>(z), static_cast<float>(w));};
	};

	namespace detail
	{

	}

	float DotProduct(const Vector2<float>& lhs, const Vector2<float>& rhs);
	double DotProduct(const Vector2<double>& lhs, const Vector2<double>& rhs);
	float DotProduct(const Vector3<float>& lhs, const Vector3<float>& rhs);
	double DotProduct(const Vector3<double>& lhs, const Vector3<double>& rhs);
	float DotProduct(const Vector4<float>& lhs, const Vector4<float>& rhs);
	double DotProduct(const Vector4<double>& lhs, const Vector4<double>& rhs);

	float Distance(const Vector2<float>& lhs, const Vector2<float>& rhs);
	double Distance(const Vector2<double>& lhs, const Vector2<double>& rhs);
	float Distance(const Vector3<float>& lhs, const Vector3<float>& rhs);
	double Distance(const Vector3<double>& lhs, const Vector3<double>& rhs);
	float Distance(const Vector4<float>& lhs, const Vector4<float>& rhs);
	double Distance(const Vector4<double>& lhs, const Vector4<double>& rhs);

	float Length(const Vector2<float>& vec);
	double Length(const Vector2<double>& vec);
	float Length(const Vector3<float>& vec);
	double Length(const Vector3<double>& vec);
	float Length(const Vector4<float>& vec);
	double Length(const Vector4<double>& vec);

	Vector2<double> Normalize(const Vector2<double>& vec);
	Vector3<double> Normalize(const Vector3<double>& vec);
	Vector4<double> Normalize(const Vector4<double>& vec);
	Vector2<float> Normalize(const Vector2<float>& vec);
	Vector3<float> Normalize(const Vector3<float>& vec);
	Vector4<float> Normalize(const Vector4<float>& vec);
	Vector2<float> NormalizeApprox(const Vector2<float>& vec);
	Vector3<float> NormalizeApprox(const Vector3<float>& vec);
	Vector4<float> NormalizeApprox(const Vector4<float>& vec);

	Vector2<double> Reflect(const Vector2<double>& incident, const Vector2<double>& surfaceNormal);
	Vector3<double> Reflect(const Vector3<double>& incident, const Vector3<double>& surfaceNormal);
	Vector4<double> Reflect(const Vector4<double>& incident, const Vector4<double>& surfaceNormal);
	Vector2<float> Reflect(const Vector2<float>& incident, const Vector2<float>& surfaceNormal);
	Vector3<float> Reflect(const Vector3<float>& incident, const Vector3<float>& surfaceNormal);
	Vector4<float> Reflect(const Vector4<float>& incident, const Vector4<float>& surfaceNormal);

	Vector2<double> Refract(const Vector2<double>& incident, const Vector2<double>& surfaceNormal, double eta);
	Vector3<double> Refract(const Vector3<double>& incident, const Vector3<double>& surfaceNormal, double eta);
	Vector4<double> Refract(const Vector4<double>& incident, const Vector4<double>& surfaceNormal, double eta);
	Vector2<float> Refract(const Vector2<float>& incident, const Vector2<float>& surfaceNormal, float eta);
	Vector3<float> Refract(const Vector3<float>& incident, const Vector3<float>& surfaceNormal, float eta);
	Vector4<float> Refract(const Vector4<float>& incident, const Vector4<float>& surfaceNormal, float eta);

	Vector3<float> CrossProduct(const Vector3<float>& lhs, const Vector3<float>& rhs);
	Vector3<double> CrossProduct(const Vector3<double>& lhs, const Vector3<double>& rhs);

} // namespace wlib::vecmath
