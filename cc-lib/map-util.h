
#ifndef _CC_LIB_MAP_UTIL_H
#define _CC_LIB_MAP_UTIL_H

#include <concepts>
#include <algorithm>
#include <utility>
#include <vector>

// e.g. std::map or std::unordered_map
template<typename M>
concept MapContainer = requires(M m) {
  typename M::key_type;
  typename M::mapped_type;
  { m.size() } -> std::convertible_to<std::size_t>;
  { *m.begin() } -> std::convertible_to<
    std::pair<const typename M::key_type, typename M::mapped_type>>;
};


// This returns a copy of the value (to avoid danging references to
// the default) so it should generally be used when def is a small
// value.
template<class M>
requires MapContainer<M>
inline auto FindOrDefault(const M &m, const typename M::key_type &k,
                          typename M::mapped_type def) ->
  typename M::mapped_type {
  auto it = m.find(k);
  if (it == m.end()) return def;
  else return it->second;
}

// Sorts the key-value pairs by the key.
template<class M>
requires MapContainer<M> && std::totally_ordered<typename M::key_type>
inline auto MapToSortedVec(const M &m) ->
  std::vector<std::pair<typename M::key_type, typename M::mapped_type>> {
  using K = typename M::key_type;
  using V = typename M::mapped_type;
  std::vector<std::pair<K, V>> vec;
  vec.reserve(m.size());
  for (const auto &[k, v] : m) vec.emplace_back(k, v);
  std::sort(vec.begin(), vec.end(),
            [](const std::pair<K, V> &a,
               const std::pair<K, V> &b) {
              return a.first < b.first;
            });
  return vec;
}

// Sorts the key-value pairs by the value, in descending order.
// This is a common thing to do when counting how often some
// items (the keys) appear.
template<class M>
requires MapContainer<M> && std::totally_ordered<typename M::mapped_type>
inline auto CountMapToDescendingVector(const M &m) ->
  std::vector<std::pair<typename M::key_type, typename M::mapped_type>> {
  using K = typename M::key_type;
  using V = typename M::mapped_type;
  std::vector<std::pair<K, V>> vec;
  vec.reserve(m.size());
  for (const auto &[k, v] : m) vec.emplace_back(k, v);
  std::sort(vec.begin(), vec.end(),
            [](const std::pair<K, V> &a,
               const std::pair<K, V> &b) {
              return b.second < a.second;
            });
  return vec;
}

#endif
