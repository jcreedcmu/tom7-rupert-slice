
#ifndef _CC_LIB_GEOM_TREE_3D_H
#define _CC_LIB_GEOM_TREE_3D_H

#include <cstdio>
#include <iostream>
#include <memory>
#include <variant>
#include <utility>
#include <vector>
#include <tuple>
#include <functional>

#include "base/logging.h"

// Maps 3D points to values of type T.
// Classic octree with no rebalancing.
template<class Num, class T>
requires std::is_arithmetic_v<Num>
struct Tree3D {
  using Pos = std::tuple<Num, Num, Num>;

  Tree3D();

  void Insert(Num x, Num y, Num z, T t);
  void Insert(Pos pos, T t);

  // Returns true if a matching point was found (and thus removed).
  bool Remove(Num x, Num y, Num z);
  bool Remove(Pos pos);

  size_t Size() const { return count; }
  bool Empty() const { return count == 0; }

  template<class F>
  void App(const F &f) const;

  // Return all points within the given radius, with their data
  // and the distance (Euclidean).
  std::vector<std::tuple<Pos, T, double>>
  LookUp(Pos pos, double radius) const;

  // Aborts if the tree is empty, so check first.
  std::tuple<Pos, T, double>
  Closest(Pos pos) const;
  std::tuple<Pos, T, double>
  Closest(Num x, Num y, Num z) const;

  void DebugPrint() const;

 private:
  int d = 0;
  enum class Axis { X, Y, Z, };
  static constexpr size_t MAX_LEAF = 8;
  using Leaf = std::vector<std::pair<Pos, T>>;
  struct Split {
    // Which axis we are splitting on. An X split means
    // separating points on either side of the YZ plane, i.e.,
    // by the value of their x coordinates only.
    Axis axis = Axis::X;
    // The location of the split.
    Num value = 0;
    std::unique_ptr<std::variant<Leaf, Split>> lesseq, greater;
  };
  using Node = std::variant<Leaf, Split>;

  static inline bool Classify(Pos pos, Axis axis, double value) {
    const auto &[x, y, z] = pos;
    switch (axis) {
    case Axis::X: return x <= value;
    case Axis::Y: return y <= value;
    case Axis::Z: return z <= value;
    }
  }

  static inline double SqDist(Pos a, Pos b) {
    const auto &[ax, ay, az] = a;
    const auto &[bx, by, bz] = b;
    double dx = ax - bx;
    double dy = ay - by;
    double dz = az - bz;
    return dx * dx + dy * dy + dz * dz;
  }

  static inline double Dist(Pos a, Pos b) {
    return sqrt(SqDist(a, b));
  }

  void InsertTo(Node *node, Axis axis, Pos pos, T t);

  static const char *AxisName(Axis axis) {
    switch (axis) {
    case Axis::X: return "X";
    case Axis::Y: return "Y";
    case Axis::Z: return "Z";
    default: return "?";
    }
  }

  static inline Axis NextAxis(Axis axis) {
    switch (axis) {
    default:
    case Axis::X: return Axis::Y;
    case Axis::Y: return Axis::Z;
    case Axis::Z: return Axis::X;
    }
  }

  size_t count = 0;
  std::unique_ptr<Node> root;
};


// Implementations follow.

template<class Num, class T>
requires std::is_arithmetic_v<Num>
Tree3D<Num, T>::Tree3D() {
  // An empty leaf.
  root = std::make_unique<Node>(std::in_place_type<Leaf>);
}

template<class Num, class T>
requires std::is_arithmetic_v<Num>
void Tree3D<Num, T>::Insert(Pos pos, T t) {
  // Insertion always increases the size by one; there can be multiple
  // items at the same point.
  count++;
  // Arbitrary preference for first split we create.
  InsertTo(root.get(), Axis::X, pos, t);
}

template<class Num, class T>
requires std::is_arithmetic_v<Num>
void Tree3D<Num, T>::Insert(Num x, Num y, Num z, T t) {
  Insert(std::make_tuple(x, y, z), t);
}

template<class Num, class T>
requires std::is_arithmetic_v<Num>
bool Tree3D<Num, T>::Remove(Num x, Num y, Num z) {
  return Remove(std::make_tuple(x, y, z));
}

template<class Num, class T>
requires std::is_arithmetic_v<Num>
bool Tree3D<Num, T>::Remove(Pos pos) {
  // This is removing without a radius, which means we can find the
  // exact node that has it. But we want to be able to replace that
  // node. So we have a reference to the node.

  std::unique_ptr<Node> &cursor = root;
  if (cursor.get() == nullptr) return false;

  int removed = 0;
  std::function<void(std::unique_ptr<Node> &)> Rec = [&Rec, &removed, pos](
      std::unique_ptr<Node> &cursor) {
      CHECK(cursor.get() != nullptr);
      if (Split *split = std::get_if<Split>(cursor.get())) {
        bool lesseq = Classify(pos, split->axis, split->value);
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
        for (int idx = 0; idx < (int)leaf->size(); /* in loop */) {
          if ((*leaf)[idx].first == pos) {
            // Erase it. We do this by swapping with the last
            // element (if any) and then reducing the size by one.
            if (idx != (int)leaf->size() - 1) {
              std::swap((*leaf)[idx], (*leaf)[leaf->size() - 1]);
            }
            leaf->pop_back();
            removed++;

            // Keep index where it is, since we haven't looked
            // at the swapped element yet (or if it was the last
            // one, we'll exit the loop).
          } else {
            idx++;
          }
        }

        // Clean up empty vectors.
        if (leaf->empty()) {
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
void Tree3D<Num, T>::App(const F &f) const {
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
      for (const auto &[ll, t] : std::get<Leaf>(*node)) {
        f(ll, t);
      }
    }
  }
}

template<class Num, class T>
requires std::is_arithmetic_v<Num>
std::vector<std::tuple<typename Tree3D<Num, T>::Pos, T, double>>
Tree3D<Num, T>::LookUp(Pos pos, double radius) const {
  const auto &[x, y, z] = pos;

  // PERF: q should have reduced radius (distance to axis).
  // PERF: We can start by using code from Closest below.
  std::vector<Node *> q = {root.get()};
  std::vector<std::tuple<Pos, T, double>> out;
  while (!q.empty()) {
    Node *node = q.back();
    q.pop_back();

    if (Split *split = std::get_if<Split>(node)) {
      // Project the lookup point to the split axis.
      const Pos axispt = [&]{
          switch (split->axis) {
          case Axis::X : return std::make_tuple(split->value, y, z);
          case Axis::Y : return std::make_tuple(x, split->value, z);
          case Axis::Z : return std::make_tuple(x, y, split->value);
          }
        }();

      // If it is within the radius, we need to check both.
      // PERF the distance is a straight line!
      const double axisdist = Dist(pos, axispt);
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
      for (const auto &[ppos, t] : std::get<Leaf>(*node)) {
        const double dist = Dist(pos, ppos);
        /*
        printf("Leaf (%d,%d) dist %.3f vs radius %.3f\n",
               ppos.first, ppos.second,
               dist, radius);
        */
        if (dist <= radius) {
          out.emplace_back(ppos, t, dist);
        }
      }
    }
  }

  return out;
}

template <class Num, class T>
requires std::is_arithmetic_v<Num>
void Tree3D<Num, T>::InsertTo(Node *node, Axis axis, Pos pos, T t) {
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
      leaf->emplace_back(pos, t);

      // May have exceeded max leaf size.
      const size_t num = leaf->size();
      if (num > MAX_LEAF) {

        // We'll replace the node in place, but we'll need
        // the old leaves to do it.
        std::vector<std::pair<Pos, T>> old = std::move(*leaf);

        // PERF: Median is likely a better choice.
        double avg = 0.0;
        for (const auto &[pos, t_] : old) {
          const auto [x, y, z] = pos;
          switch (axis) {
          case Axis::X: avg += x; break;
          case Axis::Y: avg += y; break;
          case Axis::Z: avg += z; break;
          }
        }
        avg /= num;

        CHECK(!std::holds_alternative<Split>(*node));

        // Replace contents of the node with a Split.
        node->template emplace<Split>(axis, avg, nullptr, nullptr);

        CHECK(std::holds_alternative<Split>(*node));

        // Now insert the old contents.
        for (const auto &[pos, t] : old) {
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
void Tree3D<Num, T>::DebugPrint() const {
  std::function<void(const Node*, int)> Rec =
    [&Rec](const Node *node, int pad) {
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
        for (const auto &[pos, t] : *leaf) {
          const auto &[x, y, z] = pos;
          std::cout << p << "(" << x << "," << y << "," << z
                    << "): " << t << "\n";
        }
      }
    };

  Rec(root.get(), 0);
}

template <class Num, class T>
requires std::is_arithmetic_v<Num>
std::tuple<typename Tree3D<Num, T>::Pos, T, double>
Tree3D<Num, T>::Closest(Num x, Num y, Num z) const {
  return Closest(std::make_tuple(x, y, z));
}

// Aborts if the tree is empty, so check first.
template <class Num, class T>
requires std::is_arithmetic_v<Num>
std::tuple<typename Tree3D<Num, T>::Pos, T, double>
Tree3D<Num, T>::Closest(Pos pos) const {
  const auto &[x, y, z] = pos;

  CHECK(count != 0);

  const std::pair<Pos, T> *best = nullptr;
  double best_sq_dist = 0.0;

  // minimum squared distance to the region (from pos), node
  //
  // PERF: We could actually keep the distance to the corner,
  // rather than the axis.
  //
  // PERF: This would be better as a heap so that we can check
  // the closest nodes first. Since we reject regions that are
  // further than our best, checking close ones first is much
  // faster.
  std::vector<std::pair<double, const Node *>> q = {{0.0, root.get()}};

  while (!q.empty()) {
    double node_sqdist = q.back().first;
    const Node *node = q.back().second;
    q.pop_back();
    CHECK(node != nullptr);
    // No need to search if every point in there is further than
    // our current best.
    if (best != nullptr && node_sqdist > best_sq_dist)
      continue;

    if (const Split *split = std::get_if<Split>(node)) {
      // Get the minimum distance between the lookup point and
      // the split axis.
      double sdist = [&]{
          switch (split->axis) {
          case Axis::X: return split->value - x;
          case Axis::Y: return split->value - y;
          case Axis::Z: return split->value - z;
          }
        }();
      double sq_dist = sdist * sdist;

      // We always search the one we're in. But we can also search
      // the other one if it is within our search radius.
      const bool both = best == nullptr || sq_dist <= best_sq_dist;
      const bool lesseq = Classify(pos, split->axis, split->value);

      // Since we pop from the end, put the node we're in on the queue
      // last, so that it's searched first.
      if (both) {
        // Insert the other one (if non-empty), as it is close enough.
        if (const Node *other =
            lesseq ? split->greater.get() : split->lesseq.get()) {
          q.emplace_back(sq_dist, other);
        }
      }

      if (lesseq && split->lesseq.get() != nullptr) {
        q.emplace_back(0.0, split->lesseq.get());
      } else if (!lesseq && split->greater.get() != nullptr) {
        q.emplace_back(0.0, split->greater.get());
      }
    } else {
      for (const auto &elt : std::get<Leaf>(*node)) {
        const auto &[pp, tt] = elt;
        double sq_dist = SqDist(pos, pp);
        if (best == nullptr || sq_dist < best_sq_dist) {
          best = &elt;
          best_sq_dist = sq_dist;
        }
      }
    }
  }

  CHECK(best != nullptr) << "This should find something since count > 0.";
  return std::make_tuple(std::get<0>(*best), std::get<1>(*best),
                         sqrt(best_sq_dist));

}


#endif

