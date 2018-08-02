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

#ifndef DIJKSTRA3D_HPP
#define DIJKSTRA3D_HPP

#define NHOOD_SIZE 6


void print(float *f, int n) {
  for (int i = 0; i < n; i++) {
    printf("%.1f, ", f[i]);
  }
}

void print(float* dest, int x, int y, int z) {
  for (int i = 0; i < x*y*z; i++) {
    if (i % x == 0 && i > 0) {
      printf("\n");
    }
    else if ((i % x*y) == 0 && i > 0) {
      printf("\n\n");
    }

    printf("%.1f, ", dest[i]);
  }

  printf("\n\n");
}



inline float* fill(float *arr, const size_t size, const float value) {
  for (size_t i = 0; i < size; i++) {
    arr[i] = value;
  }
  return arr;
}

float search3d(
    float* field, float* dist, 
  //  std::priority_queue<size_t, std::std::vector<size_t>, std::greater> &pq,
    const size_t sx, const size_t sxy, const size_t voxels,
    const size_t loc, const size_t target
  ) {

  if (loc == target) {
    return dist[loc];
  }

  // Lets start with a 6-neighborhood. Move to a 26-hood when 
  // this is working.
  const int neighborhood[NHOOD_SIZE] = { -1, 1, sx, -sx, sxy, -sxy };

  float delta;
  size_t neighboridx;
  for (int i = 0; i < NHOOD_SIZE; i++) {
    neighboridx = loc + neighborhood[i];
    delta = field[neighboridx];

    if (neighboridx < 0 || neighboridx >= voxels) {
      continue;
    }
    // visited nodes are marked as negative
    // in the distance field
    else if (std::signbit(delta)) {
      continue;
    }
    else if (dist[loc] + delta < dist[neighboridx]) {
      dist[neighboridx] = dist[loc] + delta;
    }
  }

  field[loc] *= -1;

  // O(V^2) version
  // replace with heap when you have it
  float min_val = +INFINITY;
  size_t next_loc = loc + 1;
  for (int i = 0; i < voxels; i++) {
    if (std::signbit(field[i])) {
      continue;
    }
    else if (dist[i] < min_val) {
      min_val = dist[i];
      next_loc = i;
    }
  }
  
  // printf("next loc: %d\n", next_loc);

  return search3d(field, dist, sx, sxy, voxels, next_loc, target);
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
  const size_t source, const size_t target) {

  const size_t voxels = sx * sy * sz;

  float* dist = new float[voxels](); 
  fill(dist, voxels, +INFINITY);
  dist[source] = 0;

  // printf("FIELD\n");
  // print(field, sx, sy, sz);

  // printf("DIST\n");
  // print(dist, sx, sy, sz);

  // std::priority_queue<size_t, std::vector<size_t>, std::greater<size_t> priority_queue;

  float shortest_distance = search3d(
    field, dist, //priority_queue,
    sx, sx * sy, voxels, 
    source, target
  );

  // printf("FIELD\n");
  // print(field, sx, sy, sz);

  // mark all nodes unvisited 
  // i.e. restore to previous state
  // so our negative number trick is invisible
  // lol so not threadsafe
  for (int i = 0; i < voxels; i++) {
    field[i] = fabs(field[i]);
  }

  // printf("DIST\n");
  // print(dist, sx, sy, sz);

  delete []dist;

  return shortest_distance;
}


int main () {
  const size_t sx = 32;
  const size_t sy = 32;
  const size_t sz = 32;
  const size_t voxels = sx * sy * sz;

  float* field = new float[voxels]();
  fill(field, voxels, 1.0);

  float min = dijkstra3d(field, sx, sy, sz, 0, voxels - 1);

  printf("min: %.2f\n", min);

  return 0;
}


#endif