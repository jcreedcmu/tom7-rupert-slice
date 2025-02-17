
#ifndef _CC_LIB_ARRAY_UTIL_H
#define _CC_LIB_ARRAY_UTIL_H

#include <array>
#include <cstddef>
#include <functional>
#include <utility>

template<class F, class I, size_t N>
auto MapArray(const F &f,
              const std::array<I, N> &a) ->
  std::array<decltype(f(a[0])), N> {
  using O = decltype(f(a[0]));
  std::array<O, N> out;
  for (size_t i = 0; i < N; i++) {
    out[i] = f(a[i]);
  }
  return out;
}

// SubArray<1, 3>(arr) returns an array of (references to) three
// consecutive elements of the array, starting at index 1. Intended
// for extracting parts of an array with structured binding; should
// not have any overhead.
template<size_t Start, size_t Count, typename T, size_t N>
inline std::array<std::reference_wrapper<const T>, Count>
SubArray(const std::array<T, N>& arr);


// Template implementations follow.

namespace internal {
template<size_t Start, size_t Count, typename T, size_t N, std::size_t... Idx>
inline auto SubArrayImpl(const std::array<T, N>& arr,
                         std::index_sequence<Idx...>) {
  static_assert(Start + Count <= N, "SubArray: range out of bounds");
  return std::array<std::reference_wrapper<const T>, Count>({
      std::ref(arr[Start + Idx])...});
}
}  // internal

template<size_t Start, size_t Count, typename T, size_t N>
inline std::array<std::reference_wrapper<const T>, Count>
SubArray(const std::array<T, N>& arr) {
  return internal::SubArrayImpl<Start, Count>(
      arr, std::make_index_sequence<Count>{});
}

#endif
