#pragma once
#include <cstddef>

namespace wlib::util
{
	/// Check if *elem* is equal to one of the elements in [start, end)
	template<typename ElemT, typename IterT>
	inline bool EqualToOneOf(const ElemT& elem, IterT start, IterT end) noexcept {
		while (start != end) {
			if (elem == *start) {
				return true;
			}
			++start;
		}
		return false;
	}

	void DenormalizeData(float* inout, size_t count, float maxval);
	void DenormalizeData(double* inout, size_t count, double maxval);
	void DenormalizeData(long double* inout, size_t count, long double maxval);

} // namespace wlib::util
