/*
 * An implementation of a Edgar Dijkstra's Shortest Path Algorithm.
 * An absolute classic.
 * 
 * E. W. Dijkstra.
 * "A Note on Two Problems in Connexion with Graphs"
 * Numerische Mathematik 1. pp. 269-271. (1959)
 *
 * Of course, I use a priority queue.
 *
 * Author: William Silversmith
 * Affiliation: Seung Lab, Princeton University
 * Date: August 2018
 */

#include <cmath>
#include <vector>
#include <queue> // probably a binomial queue
#include <cstdio>

#include "pairing_heap.hpp"

#ifndef DIJKSTRA3D_HPP
#define DIJKSTRA3D_HPP

#define NHOOD_SIZE 6


void print(float *f, int n) {
  for (int i = 0; i < n; i++) {
    printf("%.1f, ", f[i]);
  }
  printf("\n");
}


void print(int *f, int n) {
  for (int i = 0; i < n; i++) {
    printf("%d, ", f[i]);
  }
  printf("\n");
}

void print(float* dest, int x, int y, int z) {
  for (int i = 0; i < x*y*z; i++) {
    if (i % x == 0 && i > 0) {
      printf("\n");
    }
    if ((i % (x*y)) == 0 && i > 0) {
      printf("\n");
    }

    printf("%.1f, ", dest[i]);
  }

  printf("\n\n");
}


inline float* fill(float *arr, const float value, const size_t size) {
  for (size_t i = 0; i < size; i++) {
    arr[i] = value;
  }
  return arr;
}

inline void compute_neighborhood(int *neighborhood, size_t loc, size_t sx, size_t sy, size_t sz) {
  for (int i = 0; i < NHOOD_SIZE; i++) {
    neighborhood[i] = 0;
  }

  const int sxy = sx * sy;
  const int x = loc % sx;
  const int y = (int)(loc / sx);
  const int z = (int)(loc / sxy);

  if (x > 0) {
    neighborhood[0] = -1;
  }
  if (x < sx - 1) {
    neighborhood[1] = 1;
  }
  if (y > 0) {
    neighborhood[2] = -(int)sx;
  }
  if (y < sy - 1) {
    neighborhood[3] = (int)sx;
  }
  if (z > 0) {
    neighborhood[4] = -sxy;
  }
  if (z < sz - 1) {
    neighborhood[5] = sxy;
  }
}

// works for non-negative weights
// Shortest ST path
// sx, sy, sz are the dimensions of the graph
// s = index of source
// t = index of target
// field is a 3D field
float dijkstra3d(
    float* field, 
    const size_t sx, const size_t sy, const size_t sz, 
    const size_t source, const size_t target
  ) {

  const size_t voxels = sx * sy * sz;
  const size_t sxy = sx * sy;

  float *dist = new float[voxels]();
  fill(dist, +INFINITY, voxels);
  dist[source] = -0;

  // Lets start with a 6-neighborhood. Move to a 26-hood when 
  // this is working.
  int *neighborhood = new int[NHOOD_SIZE]();

  auto *heap = new MinPairingHeap();
  heap->insert(0.0, source);

  size_t loc;
  float delta;
  size_t neighboridx;
  while (!heap->empty()) {
    loc = heap->root->value;
    heap->delete_min();

    compute_neighborhood(neighborhood, loc, sx, sy, sz);

    for (int i = 0; i < NHOOD_SIZE; i++) {
      if (neighborhood[i] == 0) {
        continue;
      }

      neighboridx = loc + neighborhood[i];
      delta = field[neighboridx];

      // visited nodes are marked as negative
      // in the distance field
      if (std::signbit(dist[neighboridx])) {
        continue;
      }
      else if (dist[loc] + delta < dist[neighboridx]) {
        dist[neighboridx] = dist[loc] + delta;

        if (neighboridx == target) {
          break;
        }

        heap->insert(dist[neighboridx], neighboridx);
      }
    }

    dist[loc] *= -1;
  }

  float result = dist[target];

  delete []dist;
  delete []neighborhood;
  delete heap;

  return result;
}


int main () {
  const size_t sx = 512;
  const size_t sy = 512;
  const size_t sz = 512;
  const size_t voxels = sx * sy * sz;

  float* field = new float[voxels]();
  fill(field, 1.0, voxels);

  float min = dijkstra3d(field, sx, sy, sz, 0, voxels - 1);

  printf("min: %.2f\n", min);

  return 0;
}


#endif