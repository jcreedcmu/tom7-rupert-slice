
#ifndef _CC_LIB_UNION_FIND_H
#define _CC_LIB_UNION_FIND_H

#include <vector>

struct UnionFind {
  explicit UnionFind(int size) : arr(size, -1) {}

  int Find(int a) {
    if (arr[a] == -1) return a;
    else return arr[a] = Find(arr[a]);
  }

  void Union(int a, int b) {
    if (Find(a) != Find(b)) arr[Find(a)] = b;
  }

  int Size() const { return arr.size(); }

 private:
  std::vector<int> arr;
};

#endif
