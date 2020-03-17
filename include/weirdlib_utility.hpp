#pragma once

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
} // namespace wlib::util
