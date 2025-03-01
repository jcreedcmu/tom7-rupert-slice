
// Sorting networks for fixed-size inputs.
// Takes a tuple or array of N orderable elements (e.g. ints) and
// performs a fixed sequence of comparisons and swaps that put
// them in sorted order.
//
// These are optimal in the number of comparisons, but note that
// there are many cases where there is a known network with
// fewer layers at the expense of more (parallel) comparisons.
// e.g. for 10, we use the one with 29 compare-and-swaps in 8
// parallel layers, but there also exists one with 7 layers and
// 31 CASes.
//
//
// Call FixedSort<N>(&arr).
//
// TODO: Especially when the argument is a tuple, it is useful
// to have a functional version that returns the sorted tuple.
//
// TODO: Sometimes this produces code I like (e.g. cmov), but other
// times it just makes a huge series of branches. We should see
// if there ways to coax the compiler.

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
  CAS(0, 2); CAS(0, 1); CAS(1, 2);
}

template<class Cont, class Cmp>
requires requires (Cont *c) {
  std::get<0>(*c);
  // ...
  std::get<3>(*c);
}
inline void FixedSort4(Cont *c, const Cmp &cmp) {
  CAS(0,2); CAS(1,3);
  // --
  CAS(0,1); CAS(2,3);
  // --
  CAS(1,2);
}

template<class Cont, class Cmp>
requires requires (Cont *c) {
  std::get<0>(*c);
  // ...
  std::get<4>(*c);
}
inline void FixedSort5(Cont *c, const Cmp &cmp) {
  CAS(0,3); CAS(1,4);
  // --
  CAS(0,2); CAS(1,3);
  // --
  CAS(0,1); CAS(2,4);
  // --
  CAS(1,2); CAS(3,4);
  // --
  CAS(2,3);
}

template<class Cont, class Cmp>
requires requires (Cont *c) {
  std::get<0>(*c);
  // ...
  std::get<5>(*c);
}
inline void FixedSort6(Cont *c, const Cmp &cmp) {
  CAS(0,5); CAS(1,3); CAS(2,4);
  // --
  CAS(1,2); CAS(3,4);
  // --
  CAS(0,3); CAS(2,5);
  // --
  CAS(0,1); CAS(2,3); CAS(4,5);
  // --
  CAS(1,2); CAS(3,4);
}

template<class Cont, class Cmp>
requires requires (Cont *c) {
  std::get<0>(*c);
  // ...
  std::get<6>(*c);
}
inline void FixedSort7(Cont *c, const Cmp &cmp) {
  CAS(0,6); CAS(2,3); CAS(4,5);
  // --
  CAS(0,2); CAS(1,4); CAS(3,6);
  // --
  CAS(0,1); CAS(2,5); CAS(3,4);
  // --
  CAS(1,2); CAS(4,6);
  // --
  CAS(2,3); CAS(4,5);
  // --
  CAS(1,2); CAS(3,4); CAS(5,6);
}

template<class Cont, class Cmp>
requires requires (Cont *c) {
  std::get<0>(*c);
  // ...
  std::get<7>(*c);
}
inline void FixedSort8(Cont *c, const Cmp &cmp) {
  CAS(0,2); CAS(1,3); CAS(4,6); CAS(5,7);
  // --
  CAS(0,4); CAS(1,5); CAS(2,6); CAS(3,7);
  // --
  CAS(0,1); CAS(2,3); CAS(4,5); CAS(6,7);
  // --
  CAS(2,4); CAS(3,5);
  // --
  CAS(1,4); CAS(3,6);
  // --
  CAS(1,2); CAS(3,4); CAS(5,6);
}

template<class Cont, class Cmp>
requires requires (Cont *c) {
  std::get<0>(*c);
  // ...
  std::get<8>(*c);
}
inline void FixedSort9(Cont *c, const Cmp &cmp) {
  CAS(0,3); CAS(1,7); CAS(2,5); CAS(4,8);
  // --
  CAS(0,7); CAS(2,4); CAS(3,8); CAS(5,6);
  // --
  CAS(0,2); CAS(1,3); CAS(4,5); CAS(7,8);
  // --
  CAS(1,4); CAS(3,6); CAS(5,7);
  // --
  CAS(0,1); CAS(2,4); CAS(3,5); CAS(6,8);
  // --
  CAS(2,3); CAS(4,5); CAS(6,7);
  // --
  CAS(1,2); CAS(3,4); CAS(5,6);
}

template<class Cont, class Cmp>
requires requires (Cont *c) {
  std::get<0>(*c);
  // ...
  std::get<9>(*c);
}
inline void FixedSort10(Cont *c, const Cmp &cmp) {
  CAS(0,8); CAS(1,9); CAS(2,7); CAS(3,5); CAS(4,6);
  // --
  CAS(0,2); CAS(1,4); CAS(5,8); CAS(7,9);
  // --
  CAS(0,3); CAS(2,4); CAS(5,7); CAS(6,9);
  // --
  CAS(0,1); CAS(3,6); CAS(8,9);
  // --
  CAS(1,5); CAS(2,3); CAS(4,8); CAS(6,7);
  // --
  CAS(1,2); CAS(3,5); CAS(4,6); CAS(7,8);
  // --
  CAS(2,3); CAS(4,5); CAS(6,7);
  // --
  CAS(3,4); CAS(5,6);
}

template<class Cont, class Cmp>
requires requires (Cont *c) {
  std::get<0>(*c);
  // ...
  std::get<10>(*c);
}
inline void FixedSort11(Cont *c, const Cmp &cmp) {
  CAS(0,9); CAS(1,6); CAS(2,4); CAS(3,7); CAS(5,8);
  // --
  CAS(0,1); CAS(3,5); CAS(4,10); CAS(6,9); CAS(7,8);
  // --
  CAS(1,3); CAS(2,5); CAS(4,7); CAS(8,10);
  // --
  CAS(0,4); CAS(1,2); CAS(3,7); CAS(5,9); CAS(6,8);
  // --
  CAS(0,1); CAS(2,6); CAS(4,5); CAS(7,8); CAS(9,10);
  // --
  CAS(2,4); CAS(3,6); CAS(5,7); CAS(8,9);
  // --
  CAS(1,2); CAS(3,4); CAS(5,6); CAS(7,8);
  // --
  CAS(2,3); CAS(4,5); CAS(6,7);
}

// With 39 CAS, 9 layers.
// There also exists one with 40 CAS, 8 layers.
template<class Cont, class Cmp>
requires requires (Cont *c) {
  std::get<0>(*c);
  // ...
  std::get<11>(*c);
}
inline void FixedSort12(Cont *c, const Cmp &cmp) {
  CAS(0,8); CAS(1,7); CAS(2,6); CAS(3,11); CAS(4,10); CAS(5,9);
  // --
  CAS(0,1); CAS(2,5); CAS(3,4); CAS(6,9); CAS(7,8); CAS(10,11);
  // --
  CAS(0,2); CAS(1,6); CAS(5,10); CAS(9,11);
  // --
  CAS(0,3); CAS(1,2); CAS(4,6); CAS(5,7); CAS(8,11); CAS(9,10);
  // --
  CAS(1,4); CAS(3,5); CAS(6,8); CAS(7,10);
  // --
  CAS(1,3); CAS(2,5); CAS(6,9); CAS(8,10);
  // --
  CAS(2,3); CAS(4,5); CAS(6,7); CAS(8,9);
  // --
  CAS(4,6); CAS(5,7);
  // --
  CAS(3,4); CAS(5,6); CAS(7,8);
}

// 45 CAS, 10 layers.
// There also exists 46 CAS, 9 layers.
template<class Cont, class Cmp>
requires requires (Cont *c) {
  std::get<0>(*c);
  // ...
  std::get<12>(*c);
}
inline void FixedSort13(Cont *c, const Cmp &cmp) {
  CAS(0,12); CAS(1,10); CAS(2,9); CAS(3,7); CAS(5,11); CAS(6,8);
  // --
  CAS(1,6); CAS(2,3); CAS(4,11); CAS(7,9); CAS(8,10);
  // --
  CAS(0,4); CAS(1,2); CAS(3,6); CAS(7,8); CAS(9,10); CAS(11,12);
  // --
  CAS(4,6); CAS(5,9); CAS(8,11); CAS(10,12);
  // --
  CAS(0,5); CAS(3,8); CAS(4,7); CAS(6,11); CAS(9,10);
  // --
  CAS(0,1); CAS(2,5); CAS(6,9); CAS(7,8); CAS(10,11);
  // --
  CAS(1,3); CAS(2,4); CAS(5,6); CAS(9,10);
  // --
  CAS(1,2); CAS(3,4); CAS(5,7); CAS(6,8);
  // --
  CAS(2,3); CAS(4,5); CAS(6,7); CAS(8,9);
  // --
  CAS(3,4); CAS(5,6);
}

// 51 CAS, 10 layers.
// Also have 52 CAS, 9 layers.
template<class Cont, class Cmp>
requires requires (Cont *c) {
  std::get<0>(*c);
  // ...
  std::get<13>(*c);
}
inline void FixedSort14(Cont *c, const Cmp &cmp) {
  CAS(0,1); CAS(2,3); CAS(4,5); CAS(6,7); CAS(8,9); CAS(10,11); CAS(12,13);
  // --
  CAS(0,2); CAS(1,3); CAS(4,8); CAS(5,9); CAS(10,12); CAS(11,13);
  // --
  CAS(0,4); CAS(1,2); CAS(3,7); CAS(5,8); CAS(6,10); CAS(9,13); CAS(11,12);
  // --
  CAS(0,6); CAS(1,5); CAS(3,9); CAS(4,10); CAS(7,13); CAS(8,12);
  // --
  CAS(2,10); CAS(3,11); CAS(4,6); CAS(7,9);
  // --
  CAS(1,3); CAS(2,8); CAS(5,11); CAS(6,7); CAS(10,12);
  // --
  CAS(1,4); CAS(2,6); CAS(3,5); CAS(7,11); CAS(8,10); CAS(9,12);
  // --
  CAS(2,4); CAS(3,6); CAS(5,8); CAS(7,10); CAS(9,11);
  // --
  CAS(3,4); CAS(5,6); CAS(7,8); CAS(9,10);
  // --
  CAS(6,7);
}

// 56 CAS, 10 layers.
// Also exists: 57 CAS, 9 layers.
template<class Cont, class Cmp>
requires requires (Cont *c) {
  std::get<0>(*c);
  // ...
  std::get<14>(*c);
}
inline void FixedSort15(Cont *c, const Cmp &cmp) {
  CAS(1,2); CAS(3,10); CAS(4,14); CAS(5,8); CAS(6,13); CAS(7,12); CAS(9,11);
  // --
  CAS(0,14); CAS(1,5); CAS(2,8); CAS(3,7); CAS(6,9); CAS(10,12); CAS(11,13);
  // --
  CAS(0,7); CAS(1,6); CAS(2,9); CAS(4,10); CAS(5,11); CAS(8,13); CAS(12,14);
  // --
  CAS(0,6); CAS(2,4); CAS(3,5); CAS(7,11); CAS(8,10); CAS(9,12); CAS(13,14);
  // --
  CAS(0,3); CAS(1,2); CAS(4,7); CAS(5,9); CAS(6,8); CAS(10,11); CAS(12,13);
  // --
  CAS(0,1); CAS(2,3); CAS(4,6); CAS(7,9); CAS(10,12); CAS(11,13);
  // --
  CAS(1,2); CAS(3,5); CAS(8,10); CAS(11,12);
  // --
  CAS(3,4); CAS(5,6); CAS(7,8); CAS(9,10);
  // --
  CAS(2,3); CAS(4,5); CAS(6,7); CAS(8,9); CAS(10,11);
  // --
  CAS(5,6); CAS(7,8);
}

template<class Cont, class Cmp>
requires requires (Cont *c) {
  std::get<0>(*c);
  // ...
  std::get<15>(*c);
}
inline void FixedSort16(Cont *c, const Cmp &cmp) {
  CAS(0,13); CAS(1,12); CAS(2,15); CAS(3,14); CAS(4,8); CAS(5,6);
  CAS(7,11); CAS(9,10);
  // --
  CAS(0,5); CAS(1,7); CAS(2,9); CAS(3,4); CAS(6,13); CAS(8,14);
  CAS(10,15); CAS(11,12);
  // --
  CAS(0,1); CAS(2,3); CAS(4,5); CAS(6,8); CAS(7,9); CAS(10,11);
  CAS(12,13); CAS(14,15);
  // --
  CAS(0,2); CAS(1,3); CAS(4,10); CAS(5,11); CAS(6,7); CAS(8,9);
  CAS(12,14); CAS(13,15);
  // --
  CAS(1,2); CAS(3,12); CAS(4,6); CAS(5,7); CAS(8,10); CAS(9,11); CAS(13,14);
  // --
  CAS(1,4); CAS(2,6); CAS(5,8); CAS(7,10); CAS(9,13); CAS(11,14);
  // --
  CAS(2,4); CAS(3,6); CAS(9,12); CAS(11,13);
  // --
  CAS(3,5); CAS(6,8); CAS(7,9); CAS(10,12);
  // --
  CAS(3,4); CAS(5,6); CAS(7,8); CAS(9,10); CAS(11,12);
  // --
  CAS(6,7); CAS(8,9);
}

// TODO: More sizes from here
// https://bertdobbelaere.github.io/sorting_networks.html

#undef CAS

template<size_t N, class Cont, class Cmp>
void FixedSort(Cont *c, const Cmp &cmp) {
  if constexpr (N <= 1) { return; }
  else if constexpr (N == 2) { FixedSort2(c, cmp); }
  else if constexpr (N == 3) { FixedSort3(c, cmp); }
  else if constexpr (N == 4) { FixedSort4(c, cmp); }
  else if constexpr (N == 5) { FixedSort5(c, cmp); }
  else if constexpr (N == 6) { FixedSort6(c, cmp); }
  else if constexpr (N == 7) { FixedSort7(c, cmp); }
  else if constexpr (N == 8) { FixedSort8(c, cmp); }
  else if constexpr (N == 9) { FixedSort9(c, cmp); }
  else if constexpr (N == 10) { FixedSort10(c, cmp); }
  else if constexpr (N == 11) { FixedSort11(c, cmp); }
  else if constexpr (N == 12) { FixedSort12(c, cmp); }
  else if constexpr (N == 13) { FixedSort13(c, cmp); }
  else if constexpr (N == 14) { FixedSort14(c, cmp); }
  else if constexpr (N == 15) { FixedSort15(c, cmp); }
  else if constexpr (N == 16) { FixedSort16(c, cmp); }
  else {
    static_assert(false, "Size not supported.");
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

inline constexpr size_t MAX_FIXED_SORT_SUPPORTED = 16;
