#pragma once

namespace wlib::util
{
	template<typename ElemT, typename IterT>
	inline static bool EqualToOneOf(const ElemT& elem, IterT start, IterT end) noexcept {
		while (start != end) {
			if (elem == *start) {
				return true;
			}
			++start;
		}
		return false;
	}
} // namespace wlib::util
