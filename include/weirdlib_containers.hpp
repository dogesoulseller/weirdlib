#pragma once
#include <vector>
#include <utility>
#include <stdexcept>
#include <type_traits>
#include <algorithm>
#include <initializer_list>

#include "weirdlib_traits.hpp"

namespace wlib
{
	//NOTE: This serves as a placeholder so doxygen generates documentation

	/// Simple, vector-based associative container for use in rarely changing maps which benefit from fast access <br>
	/// Two specializations exist:
	///     - for simple data that can be easily compared
	///     - for complex data that can be hashed
	/// @tparam KeyT type of key
	/// @tparam ValueT type of value
	/// @tparam Enable SFINAE helper
	template<typename KeyT, typename ValueT, typename Enable = void>
	class unordered_flat_map
	{
		private:

		/// Based on type's size, first pair type is `std::pair<KeyT, size_t>` or `KeyT`
		std::vector<std::pair<KeyT, ValueT>> m_kvpairs;

		size_t findByKey(const KeyT& key) const noexcept;

		public:
		unordered_flat_map() = default;
		unordered_flat_map(const unordered_flat_map&) = default;
		unordered_flat_map(unordered_flat_map&&) noexcept = default;

		template<typename PairT>
		unordered_flat_map(std::initializer_list<PairT> l);

		unordered_flat_map& operator=(const unordered_flat_map&) = default;
		unordered_flat_map& operator=(unordered_flat_map&&) noexcept = default;

		/// Get count of key-value pairs
		[[nodiscard]] size_t size() const noexcept;

		/// Get preallocated element count
		[[nodiscard]] size_t capacity() const noexcept;

		/// Free excess preallocated memory
		void shrink_to_fit();

		/// Remove all key-value pairs without freeing preallocated memory
		void clear() noexcept;

		/// Preallocate memory for `size` elements
		void reserve(const size_t size);

		/// Get max element count possible for container
		[[nodiscard]] size_t max_size() const noexcept;

		/// Look up reference to value with `key`
		const ValueT& at(const KeyT& key) const;

		/// Look up reference to value with `key` <br>
		/// @see at
		const ValueT& operator[](const KeyT& key) const;

		/// Check if value with `key` exists in container
		bool exists(const KeyT& key) const noexcept;

		/// Check if `value` exists in container
		bool exists_val(const ValueT& value) const noexcept;

		/// Insert key-value pair into map <br>
		/// If value already exists, it is not overwritten
		/// @see insert_or_assign
		void insert(const KeyT& key, const ValueT& value);

		/// Insert key-value pair into map <br>
		/// If value already exists, it is overwritten
		/// @see insert
		void insert_or_assign(const KeyT& key, const ValueT& value);

		/// Remove item under `key`
		void erase(const KeyT& key);

		auto begin() noexcept;
		auto end() noexcept;
		auto cbegin() const noexcept;
		auto cend() const noexcept;
		auto rbegin() noexcept;
		auto rend() noexcept;
		auto crbegin() const noexcept;
		auto crend() const noexcept;

		bool empty();
	};

	/// Simple version
	template<typename KeyT, typename ValueT>
	class unordered_flat_map<KeyT, ValueT, typename std::enable_if_t<sizeof(KeyT) <= 8>>
	{
		private:
		std::vector<std::pair<KeyT, ValueT>> m_kvpairs;

		size_t findByKey(const KeyT& key) const noexcept {
			size_t i;
			for (i = 0; i < m_kvpairs.size(); i++) {
				if (m_kvpairs[i].first == key) {
					return i;
				}
			}

			return size_t(-1);
		}

		public:
		unordered_flat_map() = default;
		unordered_flat_map(const unordered_flat_map&) = default;
		unordered_flat_map(unordered_flat_map&&) noexcept = default;

		template<typename PairT>
		unordered_flat_map(std::initializer_list<PairT> l) : m_kvpairs{} {
			m_kvpairs.reserve(l.size());
			for (const auto& [key, val]: l) {
				insert(key, val);
			}

			shrink_to_fit();
		}

		unordered_flat_map& operator=(const unordered_flat_map&) = default;
		unordered_flat_map& operator=(unordered_flat_map&&) noexcept = default;

		[[nodiscard]] size_t size() const noexcept {return m_kvpairs.size();}

		[[nodiscard]] size_t capacity() const noexcept {return m_kvpairs.capacity();}

		void shrink_to_fit() {m_kvpairs.shrink_to_fit();}

		void clear() noexcept {m_kvpairs.clear();}

		void reserve(size_t size) {m_kvpairs.reserve(size);}

		[[nodiscard]] size_t max_size() const noexcept {return m_kvpairs.max_size();}

		bool empty() {return m_kvpairs.empty();}

		const ValueT& at(const KeyT& key) const {
			size_t i;
			for (i = 0; i < m_kvpairs.size(); i++) {
				if (m_kvpairs[i].first == key) {
					return m_kvpairs[i].second;
				}
			}

			throw std::out_of_range("");
		}

		const ValueT& operator[](const KeyT& key) const {
			return at(key);
		}

		bool exists(const KeyT& key) const noexcept {
			return findByKey(key) != size_t(-1);
		}

		bool exists_val(const ValueT& value) const noexcept {
			for (const auto& [_, c_val]: m_kvpairs) {
				if (c_val == value) {
					return true;
				}
			}

			return false;
		}

		void insert(const KeyT& key, const ValueT& value) {
			if (exists(key)) {
				return;
			} else {
				m_kvpairs.push_back(std::pair(key, value));
			}
		}

		void insert_or_assign(const KeyT& key, const ValueT& value) {
			if (const size_t location = findByKey(key); location != size_t(-1)) {
				m_kvpairs[location] = std::pair(key, value);
			} else {
				insert(key, value);
			}
		}

		void erase(const KeyT& key) {
			m_kvpairs.erase(std::find_if(m_kvpairs.cbegin(), m_kvpairs.cend(), [&key](const auto& elem){
				return elem.first == key;
			}));
		}

		auto begin() noexcept {return m_kvpairs.begin();}
		auto end() noexcept {return m_kvpairs.end();}
		auto cbegin() const noexcept {return m_kvpairs.cbegin();}
		auto cend() const noexcept {return m_kvpairs.cend();}
		auto rbegin() noexcept {return m_kvpairs.rbegin();}
		auto rend() noexcept {return m_kvpairs.rend();}
		auto crbegin() const noexcept {return m_kvpairs.crbegin();}
		auto crend() const noexcept {return m_kvpairs.crend();}
	};

	/// Complex version
	template<typename KeyT, typename ValueT>
	class unordered_flat_map<KeyT, ValueT, typename std::enable_if_t<wlib::traits::is_hashable_v<KeyT> && (sizeof(KeyT) > 8)>>
	{
		private:
		std::vector<std::pair<std::pair<KeyT, size_t>, ValueT>> m_kvpairs;

		size_t findByKey(const KeyT& key) const noexcept {
			size_t i;
			for (i = 0; i < m_kvpairs.size(); i++) {
				if (m_kvpairs[i].first.second == std::hash<KeyT>{}(key)) {
					return i;
				}
			}

			return size_t(-1);
		}

		public:
		unordered_flat_map() = default;
		unordered_flat_map(const unordered_flat_map&) = default;
		unordered_flat_map(unordered_flat_map&&) noexcept = default;
		unordered_flat_map& operator=(const unordered_flat_map&) = default;
		unordered_flat_map& operator=(unordered_flat_map&&) noexcept = default;


		template<typename PairT>
		unordered_flat_map(std::initializer_list<PairT> l) : m_kvpairs{} {
			m_kvpairs.reserve(l.size());
			for (const auto& [key, val]: l) {
				insert(key, val);
			}

			shrink_to_fit();
		}


		[[nodiscard]] size_t size() const noexcept {return m_kvpairs.size();}

		[[nodiscard]] size_t capacity() const noexcept {return m_kvpairs.capacity();}

		void shrink_to_fit() {m_kvpairs.shrink_to_fit();}

		void clear() noexcept {m_kvpairs.clear();}

		void reserve(const size_t size) {m_kvpairs.reserve(size);}

		[[nodiscard]] size_t max_size() const noexcept {return m_kvpairs.max_size();}

		bool empty() {return m_kvpairs.empty();}

		const ValueT& at(const KeyT& key) const {
			size_t i;
			for (i = 0; i < m_kvpairs.size(); i++) {
				if (m_kvpairs[i].first.second == std::hash<KeyT>{}(key)) {
					return m_kvpairs[i].second;
				}
			}

			throw std::out_of_range("");
		}

		const ValueT& operator[](const KeyT& key) const {
			return at(key);
		}

		bool exists(const KeyT& key) const noexcept {
			return findByKey(key) != size_t(-1);
		}

		bool exists_val(const ValueT& value) const noexcept {
			for (const auto& [_, c_val]: m_kvpairs) {
				if (c_val == value) {
					return true;
				}
			}
			return false;
		}

		void insert(const KeyT& key, const ValueT& value) {
			if (exists(key)) {
				return;
			} else {
				m_kvpairs.push_back(std::pair(std::pair(key, std::hash<KeyT>{}(key)), value));
			}
		}

		void insert_or_assign(const KeyT& key, const ValueT& value) {
			if (const size_t location = findByKey(key); location != size_t(-1)) {
				m_kvpairs[location] = std::pair(std::pair(key, std::hash<KeyT>{}(key)), value);
			} else {
				insert(key, value);
			}
		}

		void erase(const KeyT& key) {
			m_kvpairs.erase(std::find_if(m_kvpairs.cbegin(), m_kvpairs.cend(), [&key](const auto& elem){
				return elem.first.second == std::hash<KeyT>{}(key);
			}));
		}

		auto begin() noexcept {return m_kvpairs.begin();}
		auto end() noexcept {return m_kvpairs.end();}
		auto cbegin() const noexcept {return m_kvpairs.cbegin();}
		auto cend() const noexcept {return m_kvpairs.cend();}
		auto rbegin() noexcept {return m_kvpairs.rbegin();}
		auto rend() noexcept {return m_kvpairs.rend();}
		auto crbegin() const noexcept {return m_kvpairs.crbegin();}
		auto crend() const noexcept {return m_kvpairs.crend();}
	};

} // namespace wlib
