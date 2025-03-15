#ifndef _RUPERTS_DIRTY_H
#define _RUPERTS_DIRTY_H

#include <cmath>
#include <utility>
#include <numbers>
#include <algorithm>

#include "image.h"

// XXX To cc-lib?
// TODO: Allow a scale parameter to reduce the minimum feature
// size; we rarely need pixel precision.
struct Dirty {
  Dirty(int width, int height) : used(width, height) {
    used.Clear(false);
  }

  void MarkUsed(int x, int y, int w, int h) {
    used.SetRect(x, y, w, h, true);
  }

  bool IsUsed(int x, int y, int w, int h) const {
    for (int yy = 0; yy < h; yy++) {
      for (int xx = 0; xx < w; xx++) {
        if (used.GetPixel(x + xx, y + yy)) return true;
      }
    }
    return false;
  }

  // Find a place near (x, y) to place a rectangle of size w x h.
  // If we can't find one, just return (x, y). Does not claim the space.
  std::pair<int, int> PlaceNearby(int x, int y, int w, int h,
                                  int max_dist) const {
    if (!IsUsed(x, y, w, h)) {
      // printf("Initial spot is free\n");
      return std::make_pair(x, y);
    }

    // The smallest circle that fits in the thing.
    const float dia = std::min(w, h);

    // XXX improve this!
    for (int r = 1; r < max_dist; r += std::min(1, (int)dia)) {
      // Number of circles we could place at this distance without overlapping.
      // 2π / (2πr / dia) =
      // 1 / (r / dia) =
      // dia / r
      float span = dia / r;
      for (float theta = 0.0f; theta < 2.0 * std::numbers::pi; theta += span) {
        int xpos = std::round(x + std::cos(theta) * r);
        int ypos = std::round(y + std::sin(theta) * r);
        if (!IsUsed(xpos, ypos, w, h)) {
          return std::make_pair(xpos, ypos);
        }
      }
    }

    // printf("Couldn't find anywhere.\n");
    return std::make_pair(x, y);
  }

  Image1 used;
};

#endif
