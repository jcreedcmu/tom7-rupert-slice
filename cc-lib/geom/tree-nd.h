
#ifndef _CC_LIB_GEOM_TREE_ND_H
#define _CC_LIB_GEOM_TREE_ND_H

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

// Maps 3D points to values of type T.
// This is a classic kd tree without any rebalancing.
template<class Num, class T>
requires std::is_arithmetic_v<Num>
struct TreeND {
  using Pos = std::vector<Num>;

  // Positive number of dimensions d.
  TreeND(int d);

  void Insert(std::span<const Num> pos, T t);

  // Returns true if a matching point was found (and thus removed).
  bool Remove(std::span<const Num> pos);

  size_t Size() const { return count; }
  bool Empty() const { return count == 0; }

  template<class F>
  void App(const F &f) const;

  // Return all points within the given radius, with their data
  // and the distance (Euclidean).
  std::vector<std::tuple<Pos, T, double>>
  LookUp(std::span<const Num> pos, double radius) const;

  // Aborts if the tree is empty, so check first.
  std::tuple<Pos, T, double>
  Closest(std::span<const Num> pos) const;

  void DebugPrint() const;

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

  double SplitPoint(const Leaf &leaf, int axis) const {
    // PERF: Median is likely a better choice.
    double sum = 0.0;
    int size = leaf.Size();
    for (int i = 0; i < size; i++) {
      sum += leaf.positions[i * d + axis];
    }
    return sum / size;
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
  root = std::make_unique<Node>(std::in_place_type<Leaf>);
}

template<class Num, class T>
requires std::is_arithmetic_v<Num>
void TreeND<Num, T>::Insert(std::span<const Num> pos, T t) {
  // Insertion always increases the size by one; there can be multiple
  // items at the same point.
  count++;
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
  std::vector<Node *> q = {root.get()};
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
  std::vector<Node *> q = {root.get()};
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


#endif

