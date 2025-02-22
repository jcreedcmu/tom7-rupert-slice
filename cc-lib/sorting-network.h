
// Sorting networks for fixed-size inputs.
// Takes a tuple or array of N orderable elements (e.g. ints) and
// performs a fixed sequence of comparisons and swaps that put
// them in sorted order.
//
// Call FixedSort<N>(&arr).
//
// TODO: Especially when the argument is a tuple, it is useful
// to have a functional version that returns the sorted tuple.

#include <array>
#include <cstddef>
#include <tuple>
#include <utility>

#define CAS(a, b) do {                              \
    if (!cmp(std::get<a>(*c), std::get<b>(*c)))     \
      std::swap(std::get<a>(*c), std::get<b>(*c));  \
  } while (0)

template<class Cont, class Cmp>
requires requires (Cont *c) {
  std::get<0>(*c);
  std::get<1>(*c);
}
inline void FixedSort2(Cont *c, const Cmp &cmp) {
  CAS(0, 1);
}

template<class Cont, class Cmp>
requires requires (Cont *c) {
  std::get<0>(*c);
  // ...
  std::get<2>(*c);
}
inline void FixedSort3(Cont *c, const Cmp &cmp) {
  CAS(0, 2);
  CAS(0, 1);
  CAS(1, 2);
}

// TODO: More sizes from here
// https://bertdobbelaere.github.io/sorting_networks.html

#undef CAS

template<size_t N, class Cont, class Cmp>
void FixedSort(Cont *c, const Cmp &cmp) {
  if constexpr (N <= 1) {
    return;
  } else if constexpr (N == 2) {
    FixedSort2(c, cmp);
  } else if constexpr (N == 3) {
    FixedSort3(c, cmp);
  } else {
    static_assert("Size not supported.");
  }
}

template<size_t N, class Cont>
void FixedSort(Cont *c) {
  using T = decltype(std::get<0>(*c));
  auto F = [](const T &a, const T &b) {
      return a < b;
    };
  return FixedSort<N, Cont, decltype(F)>(c, F);
}
