
#ifndef _CC_LIB_SET_UTIL_H
#define _CC_LIB_SET_UTIL_H

#include <algorithm>
#include <functional>
#include <type_traits>
#include <vector>

// e.g.
// std::unordered_set<int64_t> s;
// std::vector<int64_t> v = ToSortedVec(s);
//
// or provide a custom comparator:
//
// std::unordered_set<std::string> s;
// std::vector<std::string> v =
//   ToSortedVec(s, [](const auto &a, const auto &b) {
//                     return a.size() < b.size();
//                  });
template<class S>
auto SetToSortedVec(const S &s,
                    const std::function<
                    bool(const typename S::key_type &a,
                         const typename S::key_type &b)> &cmp =
                    [](const typename S::key_type &a,
                       const typename S::key_type &b) {
                      return a < b;
                    }) ->
  std::vector<std::remove_cvref_t<decltype(*s.begin())>> {
  using T = std::remove_cvref_t<decltype(*s.begin())>;
  std::vector<T> ret;
  ret.reserve(s.size());
  for (const auto &elt : s)
    ret.push_back(elt);
  std::sort(ret.begin(), ret.end(), cmp);
  return ret;
}

#endif
