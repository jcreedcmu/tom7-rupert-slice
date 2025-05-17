
#ifndef _CC_LIB_GEOM_TREE_ND_H
#define _CC_LIB_GEOM_TREE_ND_H

#include <algorithm>
#include <cstdio>
#include <functional>
#include <iostream>
#include <memory>
#include <queue>
#include <span>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include "base/logging.h"

#define SPLIT_MEDIAN 1

// Maps 3D points to values of type T.
// This is a classic kd tree without any rebalancing.
template<class Num, class T>
requires std::is_arithmetic_v<Num>
struct TreeND {
  using Pos = std::vector<Num>;

  static constexpr bool SELF_CHECK = false;
  static constexpr bool VERBOSE = false;

  // Positive number of dimensions d.
  TreeND(int d);
  // Move-only.
  TreeND(TreeND &&other) = default;
  TreeND &operator =(TreeND &&other) = default;
  // Degenerate, but sometimes useful for default-initialization, etc.
  TreeND() : TreeND(1) {}

  // Initialize the tree (discarding existing contents) with
  // a batch of data. Takes ownership of the argument, because
  // it modifies it in place.
  void InitBatch(std::vector<std::pair<std::span<const Num>, T>> &&items);

  // Insert a single point.
  void Insert(std::span<const Num> pos, T t);

  // Returns true if a matching point was found (and thus removed).
  bool Remove(std::span<const Num> pos);

  size_t Size() const { return count; }
  bool Empty() const { return count == 0; }
  void Clear();

  template<class F>
  void App(const F &f) const;

  // Return all points within the given radius, with their data
  // and the distance (Euclidean).
  std::vector<std::tuple<Pos, T, double>>
  LookUp(std::span<const Num> pos, double radius) const;

  // Aborts if the tree is empty, so check first.
  std::tuple<Pos, T, double>
  Closest(std::span<const Num> pos) const;

  // Debugging and advanced things.

  void DebugPrint() const;

  struct DebugClosestResult {
    std::tuple<Pos, T, double> res;
    int64_t leaves_searched = 0;
    int64_t new_best = 0;
    int64_t max_depth = 0;
    int64_t heap_pops = 0;
    int64_t max_heap_size = 0;
  };

  DebugClosestResult DebugClosest(std::span<const Num> pos) const;


 private:
  static constexpr int MAX_LEAF = 128;
  struct Leaf {
    int Size() const { return values.size(); }
    void Add(std::span<const Num> pos, T t) {
      for (int i = 0; i < (int)pos.size(); i++) {
        positions.push_back(pos[i]);
      }
      values.push_back(t);
    }

   private:
    // Flattened: size * d positions
    std::vector<Num> positions;
    std::vector<T> values;
    friend struct TreeND;
  };

  struct Split {
    // Which axis we are splitting on, in [0, d).
    int axis = 0;
    // The location of the split.
    Num value = 0;
    std::unique_ptr<std::variant<Leaf, Split>> lesseq, greater;
  };
  using Node = std::variant<Leaf, Split>;

  inline bool Classify(std::span<const Num> pos,
                       int axis, double value) const {
    DCHECK(axis >= 0 && axis < d);
    return pos[axis] <= value;
  }

  void InitBatchTo(
      std::unique_ptr<Node> &cursor,
      std::span<std::pair<std::span<const Num>, T>> items);

  double SqDist(std::span<const Num> a,
                std::span<const Num> b) const {
    double sqdist = 0.0;
    for (int i = 0; i < d; i++) {
      double dist = b[i] - a[i];
      sqdist += dist * dist;
    }
    return sqdist;
  }

  inline double Dist(std::span<const Num> a,
                     std::span<const Num> b) const {
    return sqrt(SqDist(a, b));
  }

  inline bool SamePos(std::span<const Num> a,
                      std::span<const Num> b) const {
    for (int i = 0; i < d; i++) {
      if (a[i] != b[i]) return false;
    }
    return true;
  }

  double SplitPoint(Leaf &leaf, int axis) const {
    #if SPLIT_MEDIAN
    std::vector<int> indices;
    indices.reserve(leaf.Size());
    for (int i = 0; i < leaf.Size(); i++) {
      indices.push_back(i);
    }

    const size_t median_idx = (leaf.Size() - 1) >> 1;

    // This is like the pivot step in QuickSort. Expected O(n).
    std::nth_element(indices.begin(),
                     indices.begin() + median_idx,
                     indices.end(),
                     [this, axis, &leaf](size_t a, size_t b) {
                       return
                         leaf.positions[a * d + axis] <
                         leaf.positions[b * d + axis];
                     });

    return leaf.positions[indices[median_idx] * d + axis];

    #else
    // PERF: Median is likely a better choice.
    double sum = 0.0;
    int size = leaf.Size();
    for (int i = 0; i < size; i++) {
      sum += leaf.positions[i * d + axis];
    }
    return sum / size;
    #endif
  }

  std::span<const Num> LeafPos(const Leaf &leaf, int i) const {
    return std::span<const Num>(leaf.positions.data() + (d * i), d);
  }

  T LeafData(const Leaf &leaf, int i) const {
    return leaf.values[i];
  }

  void LeafEraseAt(Leaf *leaf, int idx) const {
    if (idx != (int)leaf->values.size() - 1) {
      std::swap(leaf->values[idx],
                leaf->values[leaf->values.size() - 1]);
    }
    leaf->values.pop_back();

    // And the d positions.
    if (idx * d != (int)leaf->positions.size() - d) {
      for (int i = 0; i < d; i++) {
        leaf->positions[idx * d + i] =
          leaf->positions[leaf->positions.size() - d + i];
      }
    }
    leaf->positions.resize(leaf->positions.size() - d);
  }


  void InsertTo(Node *node, int axis,
                std::span<const Num> pos, T t);

  static std::string AxisName(int axis) {
    return std::format("axis[{}]", axis);
  }

  inline int NextAxis(int axis) {
    return (axis + 1) % d;
  }

  int d = 0;
  size_t count = 0;
  std::unique_ptr<Node> root;
};


// Implementations follow.

template<class Num, class T>
requires std::is_arithmetic_v<Num>
TreeND<Num, T>::TreeND(int d) : d(d) {
  // An empty leaf.
  // root = std::make_unique<Node>(std::in_place_type<Leaf>);
}

template<class Num, class T>
requires std::is_arithmetic_v<Num>
void TreeND<Num, T>::Insert(std::span<const Num> pos, T t) {
  // Insertion always increases the size by one; there can be multiple
  // items at the same point.
  count++;
  if (root.get() == nullptr) {
    root = std::make_unique<Node>(std::in_place_type<Leaf>);
  }

  // Arbitrary preference for first split we create.
  InsertTo(root.get(), 0, pos, t);
}

template<class Num, class T>
requires std::is_arithmetic_v<Num>
bool TreeND<Num, T>::Remove(std::span<const Num> pos) {
  // This is removing without a radius, which means we can find the
  // exact node that has it. But we want to be able to replace that
  // node. So we have a reference to the node.

  std::unique_ptr<Node> &cursor = root;
  if (cursor.get() == nullptr) return false;

  int removed = 0;
  std::function<void(std::unique_ptr<Node> &)> Rec =
    [this, &Rec, &removed, pos](std::unique_ptr<Node> &cursor) {
      CHECK(cursor.get() != nullptr);
      if (Split *split = std::get_if<Split>(cursor.get())) {
        const bool lesseq = Classify(pos, split->axis, split->value);
        if (lesseq) {
          if (split->lesseq.get() == nullptr) {
            // Then the point cannot exist.
            return;
          }

          Rec(split->lesseq);
        } else {
          if (split->greater.get() == nullptr) {
            // Then the point cannot exist.
            return;
          }
          Rec(split->greater);
        }

        // Delete empty splits on the way back.
        if (split->lesseq.get() == nullptr &&
            split->greater.get() == nullptr) {
          cursor.reset(nullptr);
        }

      } else {
        Leaf *leaf = &std::get<Leaf>(*cursor.get());
        for (int idx = 0; idx < (int)leaf->Size(); /* in loop */) {
          if (SamePos(LeafPos(*leaf, idx), pos)) {
            // Erase it. We do this by swapping with the last
            // element (if any) and then reducing the size by one.
            LeafEraseAt(leaf, idx);
            removed++;

            // Keep index where it is, since we haven't looked
            // at the swapped element yet (or if it was the last
            // one, we'll exit the loop).
          } else {
            idx++;
          }
        }

        // Clean up empty vectors.
        if (leaf->Size() == 0) {
          cursor.reset(nullptr);
        }
      }
    };

  Rec(root);
  count -= removed;
  return removed != 0;
}

// Calls f(pos, t) for every point in the tree in
// arbitrary order. The callback should not modify an
// alias of the tree!
template<class Num, class T>
requires std::is_arithmetic_v<Num>
template<class F>
void TreeND<Num, T>::App(const F &f) const {
  std::vector<Node *> q;
  if (root.get() != nullptr) q.push_back(root.get());
  while (!q.empty()) {
    Node *node = q.back();
    q.pop_back();

    if (Split *split = std::get_if<Split>(node)) {
      if (split->lesseq.get() != nullptr)
        q.push_back(split->lesseq.get());
      if (split->greater.get() != nullptr)
        q.push_back(split->greater.get());

    } else {
      const Leaf *leaf = &std::get<Leaf>(*node);
      int size = leaf->Size();
      for (int i = 0; i < size; i++) {
        f(LeafPos(*leaf, i), LeafData(*leaf, i));
      }
    }
  }
}

template<class Num, class T>
requires std::is_arithmetic_v<Num>
std::vector<std::tuple<typename TreeND<Num, T>::Pos, T, double>>
TreeND<Num, T>::LookUp(std::span<const Num> pos, double radius) const {

  // PERF: q should be a pair with the reduced radius (distance to axis).
  // (But the leaf search uses the original point.)
  // This is pretty easy. Just do it!
  // PERF: We can start by using code from Closest below.

  std::vector<Node *> q;
  if (root.get() != nullptr) q.push_back(root.get());

  std::vector<std::tuple<Pos, T, double>> out;
  while (!q.empty()) {
    Node *node = q.back();
    q.pop_back();

    if (Split *split = std::get_if<Split>(node)) {

      // PERF! No need to copy the axis point. Just
      // compute the distance along that one dimension.
      // Project the lookup point to the split axis.
      std::vector<Num> axis_pt(pos.begin(), pos.end());
      axis_pt[split->axis] = split->value;

      // If it is within the radius, we need to check both.
      // PERF the distance is a straight line!
      const double axisdist = Dist(pos, axis_pt);
      const bool both = axisdist <= radius;
      const bool lesseq = Classify(pos, split->axis, split->value);

      if (both || lesseq) {
        if (split->lesseq.get() != nullptr) {
          q.push_back(split->lesseq.get());
        }
      }

      if (both || !lesseq) {
        if (split->greater.get() != nullptr) {
          q.push_back(split->greater.get());
        }
      }

    } else {
      const Leaf *leaf = &std::get<Leaf>(*node);
      int num = leaf->Size();
      for (int i = 0; i < num; i++) {
        std::span<const Num> ppos = LeafPos(*leaf, i);
        const double dist = Dist(pos, ppos);
        if (dist <= radius) {
          out.emplace_back(std::vector<Num>(ppos.begin(), ppos.end()),
                           LeafData(*leaf, i), dist);
        }
      }
    }
  }

  return out;
}

template <class Num, class T>
requires std::is_arithmetic_v<Num>
void TreeND<Num, T>::InsertTo(Node *node, int axis,
                              std::span<const Num> pos, T t) {
  for (;;) {
    CHECK(node != nullptr);
    if (Split *split = std::get_if<Split>(node)) {
      // Next split should use the next axis.
      axis = NextAxis(split->axis);
      bool lesseq = Classify(pos, split->axis, split->value);
      // Leaves are created lazily, so add an empty one if
      // null (but do the actual insertion on the next pass).
      if (lesseq) {
        if (split->lesseq.get() == nullptr) {
          split->lesseq = std::make_unique<Node>(std::in_place_type<Leaf>);
        }
        node = split->lesseq.get();
      } else {
        if (split->greater.get() == nullptr) {
          split->greater = std::make_unique<Node>(std::in_place_type<Leaf>);
        }
        node = split->greater.get();
      }
    } else {
      Leaf *leaf = &std::get<Leaf>(*node);
      CHECK((int)pos.size() == d);
      leaf->Add(pos, t);

      // May have exceeded max leaf size.
      const int num = (int)leaf->Size();
      if (num > MAX_LEAF) {
        // We'll replace the node in place, but we'll need
        // the old leaves to do it.
        Leaf old = std::move(*leaf);

        const double split_pt = SplitPoint(old, axis);

        CHECK(!std::holds_alternative<Split>(*node));

        // Replace contents of the node with a Split.
        node->template emplace<Split>(axis, split_pt, nullptr, nullptr);

        CHECK(std::holds_alternative<Split>(*node));

        // Now insert the old contents.
        for (int i = 0; i < num; i++) {
          std::span<const Num> pos = LeafPos(old, i);
          T t = LeafData(old, i);
          InsertTo(node, axis, pos, t);
        }
      }
      // Inserted, so we are done.
      return;
    }
  }
}

template <class Num, class T>
requires std::is_arithmetic_v<Num>
void TreeND<Num, T>::Clear() {
  count = 0;
  root.reset(nullptr);
}


template <class Num, class T>
requires std::is_arithmetic_v<Num>
void TreeND<Num, T>::InitBatch(
    std::vector<std::pair<std::span<const Num>, T>> &&items_in) {
  Clear();

  std::vector<
    std::pair<std::span<const Num>, T>
    > items = std::move(items_in);

  InitBatchTo(root, items);
}

template <class Num, class T>
requires std::is_arithmetic_v<Num>
void TreeND<Num, T>::InitBatchTo(
    std::unique_ptr<Node> &cursor,
    std::span<std::pair<std::span<const Num>, T>> items) {

  if (VERBOSE) {
    printf("InitBatchTo (%lld items)\n", items.size());
  }

  CHECK(cursor.get() == nullptr);
  if (items.empty()) return;

  if (items.size() <= MAX_LEAF) {
    cursor = std::make_unique<Node>(std::in_place_type<Leaf>);
    Leaf *leaf = &std::get<Leaf>(*cursor);
    for (auto &[pos, t] : items) {
      leaf->Add(pos, t);
    }
    count += items.size();
    return;
  }

  // Enough points for a split.
  // Choose the axis as the one with the largest variance.
  int best_axis = 0;
  double best_ssq = 0.0;
  for (int a = 0; a < d; a++) {
    double avg = 0.0;
    for (const auto &[pos, t_] : items) {
      avg += pos[a];
    }

    avg /= items.size();

    double ssq = 0.0;
    for (const auto &[pos, t_] : items) {
      double da = pos[a] - avg;
      ssq += da * da;
    }
    if (ssq > best_ssq) {
      best_ssq = ssq;
      best_axis = a;
    }
  }

  if (VERBOSE) {
    printf("%lld pts. Best axis is %d with variance %.11g\n",
           items.size(),
           best_axis, best_ssq / items.size());
  }

  // FIXME: If all of the points are the same, we have no choice
  // but to leave this as a leaf (or else we'll just get into an
  // endless loop here). This is plausible for integer data sets
  // especially. We would also want to detect such leaves later.

  // Then compute the split point. Here we compute the
  // median and partition in one go.
  const size_t median_idx = (items.size() - 1) >> 1;


  // This is like the pivot step in QuickSort. Expected O(n).
  std::nth_element(items.begin(),
                   items.begin() + median_idx,
                   items.end(),
                   [best_axis](const auto &a, const auto &b) {
                     return a.first[best_axis] < b.first[best_axis];
                   });

  if (SELF_CHECK) {
    #if 0
    // too slow.
    for (size_t i = 0; i < median_idx; i++) {
      for (int j = median_idx + 1; j < items.size(); j++) {
        CHECK(items[i].first[best_axis] <= items[j].first[best_axis]) <<
          "Median: " << median_idx << " For i=" << i << " and j=" << j;
      }
    }
    #endif

    CHECK(median_idx != 0 && median_idx <= items.size() - 1);

    Num max_left = items[0].first[best_axis];
    for (size_t i = 1; i < median_idx; i++) {
      max_left = std::max(max_left, items[i].first[best_axis]);
    }

    Num min_right = items[median_idx].first[best_axis];
    for (size_t j = median_idx; j < items.size(); j++) {
      min_right = std::min(min_right, items[j].first[best_axis]);
    }
    CHECK(max_left <= min_right);

    if (VERBOSE) {
      printf("median Self check OK. %.11g < %.11g\n", max_left, min_right);
    }
  }


  double split_pt = items[median_idx].first[best_axis];

  // Still need to partition the data, because there might be values
  // equal to the median on each side.
  const size_t partition_idx =
    std::partition(items.begin(),
                   items.end(),
                   [split_pt, best_axis](const auto &item) {
                     return item.first[best_axis] <= split_pt;
                   }) - items.begin();

  if (VERBOSE) {
    printf("Median idx was %lld. Median value = Split pt is %.11g\n"
           "Partition idx: %lld\n", median_idx, split_pt, partition_idx);
  }

  // XXX make sure the split is not degenerate. Maybe combined
  // with the code that checks for all-equal elements (we can
  // compute the median and partitions first, then detect
  // that they are all equal that way).
  if (SELF_CHECK) {
    for (size_t i = 0; i < partition_idx; i++) {
      CHECK(items[i].first[best_axis] <= split_pt);
    }

    for (size_t i = partition_idx; i < items.size(); i++) {
      CHECK(split_pt < items[i].first[best_axis]);
    }
  }

  CHECK(partition_idx != 0 &&
        partition_idx != items.size()) << "Need to handle degenerate "
    "splits (where all the data are the same)";

  cursor = std::make_unique<Node>(std::in_place_type<Split>);
  Split *split = &std::get<Split>(*cursor);

  split->axis = best_axis;
  split->value = split_pt;

  InitBatchTo(split->lesseq, std::span(items.data(), partition_idx));
  InitBatchTo(split->greater, std::span(items.data() + partition_idx,
                                        items.size() - partition_idx));
}

template <class Num, class T>
requires std::is_arithmetic_v<Num>
void TreeND<Num, T>::DebugPrint() const {
  std::function<void(const Node*, int)> Rec =
    [this, &Rec](const Node *node, int pad) {
      std::string p(pad, ' ');
      if (const Split *split = std::get_if<Split>(node)) {
        std::cout << p
                  << AxisName(split->axis) << " @"
                  << split->value << "\n";
        if (split->lesseq.get() != nullptr) {
          std::cout << p << "LESSEQ:\n";
          Rec(split->lesseq.get(), pad + 2);
        }
        if (split->greater.get() != nullptr) {
          std::cout << p << "GREATER:\n";
          Rec(split->greater.get(), pad + 2);
        }

      } else {
        const Leaf *leaf = std::get_if<Leaf>(node);
        CHECK(leaf != nullptr);
        for (int i = 0; i < leaf->Size(); i++) {
          std::span<const Num> pos = LeafPos(*leaf, i);
          T t = LeafData(*leaf, i);
          std::cout << p << "(";
          for (const Num &a : pos) {
            std::cout << a << ", ";
          }
          std::cout << "): " << t << "\n";
        }
      }
    };

  Rec(root.get(), 0);
}

// Aborts if the tree is empty, so check first.
template <class Num, class T>
requires std::is_arithmetic_v<Num>
std::tuple<typename TreeND<Num, T>::Pos, T, double>
TreeND<Num, T>::Closest(std::span<const Num> pos) const {
  CHECK(count != 0) << "Closest can only be called on a non-empty "
    "tree.";

  const Leaf *best_leaf = nullptr;
  int best_leaf_idx = 0;
  double best_sq_dist = std::numeric_limits<double>::infinity();

  // lower bound on the squared distance to the region (from pos), node
  //
  // We use a heap because we want to check closer regions first;
  // whenever we find a point this gives us a new lower bound.
  //
  // PERF: We could actually keep the distance to the corner,
  // rather than the axis.
  using Elt = std::pair<double, const Node *>;
  std::priority_queue<Elt, std::vector<Elt>, std::greater<Elt>> q;

  q.push({0.0, root.get()});

  while (!q.empty()) {
    const double node_sqdist = q.top().first;
    const Node *node = q.top().second;

    q.pop();

    // Since this is the node with the smallest lower bound remaining
    // in the heap, we are done.
    if (node_sqdist >= best_sq_dist) {
      break;
    }

    CHECK(node != nullptr);

    if (const Split *split = std::get_if<Split>(node)) {
      // Get the minimum distance between the lookup point and
      // the split axis.
      double sdist = split->value - pos[split->axis];
      double sq_dist = sdist * sdist;

      // Note we may be close to the split plane, but far from
      // the parent node (different axis); take the max.
      const double other_dist = std::max(sq_dist, node_sqdist);

      // We always search the one we're in. But we can also search
      // the other one if it is within our search radius.
      const bool both = best_leaf == nullptr || other_dist <= best_sq_dist;
      const bool lesseq = Classify(pos, split->axis, split->value);

      if (both) {
        // Insert the other one (if non-empty), as it is close enough.
        if (const Node *other =
            lesseq ? split->greater.get() : split->lesseq.get()) {
          q.emplace(other_dist, other);
        }
      }

      if (lesseq && split->lesseq.get() != nullptr) {
        q.emplace(node_sqdist, split->lesseq.get());
      } else if (!lesseq && split->greater.get() != nullptr) {
        q.emplace(node_sqdist, split->greater.get());
      }
    } else {

      const Leaf *leaf = &std::get<Leaf>(*node);
      int num = leaf->Size();
      for (int i = 0; i < num; i++) {
        std::span<const Num> ppos = LeafPos(*leaf, i);
        double sq_dist = SqDist(pos, ppos);
        if (best_leaf == nullptr || sq_dist < best_sq_dist) {
          best_leaf = leaf;
          best_leaf_idx = i;
          best_sq_dist = sq_dist;
        }
      }
    }
  }

  CHECK(best_leaf != nullptr) << "This should find something since count > 0.";

  std::span<const Num> ppos = LeafPos(*best_leaf, best_leaf_idx);
  return std::make_tuple(std::vector<Num>(ppos.begin(), ppos.end()),
                         LeafData(*best_leaf, best_leaf_idx),
                         sqrt(best_sq_dist));
}


// XXX Probably should not keep two copies of this code around.
template <class Num, class T>
requires std::is_arithmetic_v<Num>
typename TreeND<Num, T>::DebugClosestResult
TreeND<Num, T>::DebugClosest(std::span<const Num> pos) const {
  CHECK(count != 0) << "Closest can only be called on a non-empty "
    "tree.";

  DebugClosestResult ret;

  const Leaf *best_leaf = nullptr;
  int best_leaf_idx = 0;
  double best_sq_dist = std::numeric_limits<double>::infinity();

  // lower bound on the squared distance to the region (from pos), node
  //
  // We use a heap because we want to check closer regions first;
  // whenever we find a point this gives us a new lower bound.
  //
  // PERF: We could actually keep the distance to the corner,
  // rather than the axis.
  // distance, node, depth
  using Elt = std::tuple<double, const Node *, int64_t>;
  std::priority_queue<Elt, std::vector<Elt>, std::greater<Elt>> q;

  q.push({0.0, root.get(), 0});

  while (!q.empty()) {
    const double node_sqdist = std::get<0>(q.top());
    const Node *node = std::get<1>(q.top());
    const int64_t depth = std::get<2>(q.top());

    ret.heap_pops++;
    ret.max_heap_size = std::max((int64_t)q.size(), ret.max_heap_size);
    ret.max_depth = std::max(depth, ret.max_depth);

    q.pop();

    // Since this is the node with the smallest lower bound remaining
    // in the heap, we are done.
    if (node_sqdist >= best_sq_dist) {
      break;
    }

    CHECK(node != nullptr);

    if (const Split *split = std::get_if<Split>(node)) {
      // Get the minimum distance between the lookup point and
      // the split axis.
      double sdist = split->value - pos[split->axis];
      double sq_dist = sdist * sdist;

      // Note we may be close to the split plane, but far from
      // the parent node (different axis); take the max.
      const double other_dist = std::max(sq_dist, node_sqdist);

      // We always search the one we're in. But we can also search
      // the other one if it is within our search radius.
      const bool both = best_leaf == nullptr || other_dist <= best_sq_dist;
      const bool lesseq = Classify(pos, split->axis, split->value);

      if (both) {
        // Insert the other one (if non-empty), as it is close enough.
        if (const Node *other =
            lesseq ? split->greater.get() : split->lesseq.get()) {
          q.emplace(other_dist, other, depth + 1);
        }
      }

      if (lesseq && split->lesseq.get() != nullptr) {
        q.emplace(node_sqdist, split->lesseq.get(), depth + 1);
      } else if (!lesseq && split->greater.get() != nullptr) {
        q.emplace(node_sqdist, split->greater.get(), depth + 1);
      }
    } else {

      const Leaf *leaf = &std::get<Leaf>(*node);
      int num = leaf->Size();
      for (int i = 0; i < num; i++) {
        ret.leaves_searched++;
        std::span<const Num> ppos = LeafPos(*leaf, i);
        double sq_dist = SqDist(pos, ppos);
        if (best_leaf == nullptr || sq_dist < best_sq_dist) {
          ret.new_best++;
          best_leaf = leaf;
          best_leaf_idx = i;
          best_sq_dist = sq_dist;
        }
      }
    }
  }

  CHECK(best_leaf != nullptr) << "This should find something since count > 0.";

  std::span<const Num> ppos = LeafPos(*best_leaf, best_leaf_idx);
  ret.res = std::make_tuple(std::vector<Num>(ppos.begin(), ppos.end()),
                            LeafData(*best_leaf, best_leaf_idx),
                            sqrt(best_sq_dist));
  return ret;
}


#endif

