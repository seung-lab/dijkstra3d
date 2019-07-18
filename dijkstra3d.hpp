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

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <queue>
#include <vector>

#include "./libdivide.h"

#ifndef DIJKSTRA3D_HPP
#define DIJKSTRA3D_HPP

#define NHOOD_SIZE 26

namespace dijkstra {

inline float* fill(float *arr, const float value, const size_t size) {
  for (size_t i = 0; i < size; i++) {
    arr[i] = value;
  }
  return arr;
}

inline std::vector<uint32_t> query_shortest_path(uint32_t* parents, const uint32_t target) {
  std::vector<uint32_t> path;
  uint32_t loc = target;
  while (parents[loc]) {
    path.push_back(loc);
    loc = parents[loc] - 1; // offset by 1 to disambiguate the 0th index
  }
  path.push_back(loc);

  return path;
}

inline void compute_neighborhood(
  int *neighborhood, 
  const int x, const int y, const int z,
  const size_t sx, const size_t sy, const size_t sz) {

  for (int i = 0; i < NHOOD_SIZE; i++) {
    neighborhood[i] = 0;
  }

  const int sxy = sx * sy;

  // 6-hood

  if (x > 0) {
    neighborhood[0] = -1;
  }
  if (x < (int)sx - 1) {
    neighborhood[1] = 1;
  }
  if (y > 0) {
    neighborhood[2] = -(int)sx;
  }
  if (y < (int)sy - 1) {
    neighborhood[3] = (int)sx;
  }
  if (z > 0) {
    neighborhood[4] = -sxy;
  }
  if (z < (int)sz - 1) {
    neighborhood[5] = sxy;
  }

  // 18-hood

  // xy diagonals
  neighborhood[6] = (neighborhood[0] + neighborhood[2]) * (neighborhood[0] && neighborhood[2]); // up-left
  neighborhood[7] = (neighborhood[0] + neighborhood[3]) * (neighborhood[0] && neighborhood[3]); // up-right
  neighborhood[8] = (neighborhood[1] + neighborhood[2]) * (neighborhood[1] && neighborhood[2]); // down-left
  neighborhood[9] = (neighborhood[1] + neighborhood[3]) * (neighborhood[1] && neighborhood[3]); // down-right

  // yz diagonals
  neighborhood[10] = (neighborhood[2] + neighborhood[4]) * (neighborhood[2] && neighborhood[4]); // up-left
  neighborhood[11] = (neighborhood[2] + neighborhood[5]) * (neighborhood[2] && neighborhood[5]); // up-right
  neighborhood[12] = (neighborhood[3] + neighborhood[4]) * (neighborhood[3] && neighborhood[4]); // down-left
  neighborhood[13] = (neighborhood[3] + neighborhood[5]) * (neighborhood[3] && neighborhood[5]); // down-right

  // xz diagonals
  neighborhood[14] = (neighborhood[0] + neighborhood[4]) * (neighborhood[0] && neighborhood[4]); // up-left
  neighborhood[15] = (neighborhood[0] + neighborhood[5]) * (neighborhood[0] && neighborhood[5]); // up-right
  neighborhood[16] = (neighborhood[1] + neighborhood[4]) * (neighborhood[1] && neighborhood[4]); // down-left
  neighborhood[17] = (neighborhood[1] + neighborhood[5]) * (neighborhood[1] && neighborhood[5]); // down-right

  // 26-hood

  // Now the eight corners of the cube
  neighborhood[18] = (neighborhood[0] + neighborhood[2] + neighborhood[4]) * (neighborhood[2] && neighborhood[4]);
  neighborhood[19] = (neighborhood[1] + neighborhood[2] + neighborhood[4]) * (neighborhood[2] && neighborhood[4]);
  neighborhood[20] = (neighborhood[0] + neighborhood[3] + neighborhood[4]) * (neighborhood[3] && neighborhood[4]);
  neighborhood[21] = (neighborhood[0] + neighborhood[2] + neighborhood[5]) * (neighborhood[2] && neighborhood[5]);
  neighborhood[22] = (neighborhood[1] + neighborhood[3] + neighborhood[4]) * (neighborhood[3] && neighborhood[4]);
  neighborhood[23] = (neighborhood[1] + neighborhood[2] + neighborhood[5]) * (neighborhood[2] && neighborhood[5]);
  neighborhood[24] = (neighborhood[0] + neighborhood[3] + neighborhood[5]) * (neighborhood[3] && neighborhood[5]);
  neighborhood[25] = (neighborhood[1] + neighborhood[3] + neighborhood[5]) * (neighborhood[3] && neighborhood[5]);
}

class HeapNode {
public:
  float key; 
  uint32_t value;

  HeapNode() {
    key = 0;
    value = 0;
  }

  HeapNode(float k, uint32_t val) {
    key = k;
    value = val;
  }

  HeapNode (const HeapNode &h) {
    key = h.key;
    value = h.value;
  }
};

struct HeapNodeCompare {
  bool operator()(const HeapNode &t1, const HeapNode &t2) const {
    return t1.key >= t2.key;
  }
};

/* Perform dijkstra's shortest path algorithm
 * on a 3D image grid. Vertices are voxels and
 * edges are the 26 nearest neighbors (except
 * for the edges of the image where the number
 * of edges is reduced).
 *
 * For given input voxels A and B, the edge
 * weight from A to B is B and from B to A is
 * A. All weights must be non-negative (incl. 
 * negative zero).
 *
 * I take advantage of negative weights to mean
 * "visited".
 *
 * Parameters:
 *  T* field: Input weights. T can be be a floating or 
 *     signed integer type, but not an unsigned int.
 *  sx, sy, sz: size of the volume along x,y,z axes in voxels.
 *  source: 1D index of starting voxel
 *  target: 1D index of target voxel
 *
 * Returns: vector containing 1D indices of the path from
 *   source to target including source and target.
 */
template <typename T>
std::vector<uint32_t> dijkstra3d(
    T* field, 
    const size_t sx, const size_t sy, const size_t sz, 
    const size_t source, const size_t target
  ) {

  if (source == target) {
    return std::vector<uint32_t>{ static_cast<uint32_t>(source) };
  }

  const size_t voxels = sx * sy * sz;
  const size_t sxy = sx * sy;
  
  const libdivide::divider<size_t> fast_sx(sx); 
  const libdivide::divider<size_t> fast_sxy(sxy); 

  const bool power_of_two = !((sx & (sx - 1)) || (sy & (sy - 1))); 
  const int xshift = std::log2(sx); // must use log2 here, not lg/lg2 to avoid fp errors
  const int yshift = std::log2(sy);

  float *dist = new float[voxels]();
  uint32_t *parents = new uint32_t[voxels]();
  fill(dist, +INFINITY, voxels);
  dist[source] = -0;

  int neighborhood[NHOOD_SIZE];

  std::priority_queue<HeapNode, std::vector<HeapNode>, HeapNodeCompare> queue;
  queue.emplace(0.0, source);

  size_t loc;
  float delta;
  size_t neighboridx;

  int x, y, z;

  while (!queue.empty()) {
    loc = queue.top().value;
    queue.pop();
    
    if (std::signbit(dist[loc])) {
      continue;
    }

    if (power_of_two) {
      z = loc >> (xshift + yshift);
      y = (loc - (z << (xshift + yshift))) >> xshift;
      x = loc - ((y + (z << yshift)) << xshift);
    }
    else {
      z = loc / fast_sxy;
      y = (loc - (z * sxy)) / fast_sx;
      x = loc - sx * (y + z * sy);
    }

    compute_neighborhood(neighborhood, x, y, z, sx, sy, sz);

    for (int i = 0; i < NHOOD_SIZE; i++) {
      if (neighborhood[i] == 0) {
        continue;
      }

      neighboridx = loc + neighborhood[i];
      delta = (float)field[neighboridx];

      // Visited nodes are negative and thus the current node
      // will always be less than as field is filled with non-negative
      // integers.
      if (dist[loc] + delta < dist[neighboridx]) { 
        dist[neighboridx] = dist[loc] + delta;
        parents[neighboridx] = loc + 1; // +1 to avoid 0 ambiguity

        // Dijkstra, Edgar. "Go To Statement Considered Harmful".
        // Communications of the ACM. Vol. 11. No. 3 March 1968. pp. 147-148
        if (neighboridx == target) {
          goto OUTSIDE;
        }

        queue.emplace(dist[neighboridx], neighboridx);
      }
    }

    dist[loc] *= -1;
  }

  OUTSIDE:
  delete []dist;

  std::vector<uint32_t> path = query_shortest_path(parents, target);
  delete [] parents;

  return path;
}

template <typename T>
std::vector<uint32_t> dijkstra2d(
    T* field, 
    const size_t sx, const size_t sy, 
    const size_t source, const size_t target
  ) {

  return dijkstra3d<T>(field, sx, sy, 1, source, target);
}

template <typename T>
uint32_t* parental_field3d(
    T* field, 
    const size_t sx, const size_t sy, const size_t sz, 
    const size_t source, uint32_t* parents = NULL
  ) {

  const size_t voxels = sx * sy * sz;
  const size_t sxy = sx * sy;

  const libdivide::divider<size_t> fast_sx(sx); 
  const libdivide::divider<size_t> fast_sxy(sxy); 

  const bool power_of_two = !((sx & (sx - 1)) || (sy & (sy - 1))); 
  const int xshift = std::log2(sx); // must use log2 here, not lg/lg2 to avoid fp errors
  const int yshift = std::log2(sy);

  float *dist = new float[voxels]();
  
  if (parents == NULL) {
    parents = new uint32_t[voxels]();
  }

  fill(dist, +INFINITY, voxels);
  dist[source] = -0;

  int neighborhood[NHOOD_SIZE];

  std::priority_queue<HeapNode, std::vector<HeapNode>, HeapNodeCompare> queue;
  queue.emplace(0.0, source);

  size_t loc;
  float delta;
  size_t neighboridx;

  size_t x, y, z;

  while (!queue.empty()) {
    loc = queue.top().value;
    queue.pop();

    if (std::signbit(dist[loc])) {
      continue;
    }

    if (power_of_two) {
      z = loc >> (xshift + yshift);
      y = (loc - (z << (xshift + yshift))) >> xshift;
      x = loc - ((y + (z << yshift)) << xshift);
    }
    else {
      z = loc / fast_sxy;
      y = (loc - (z * sxy)) / fast_sx;
      x = loc - sx * (y + z * sy);
    }

    compute_neighborhood(neighborhood, x, y, z, sx, sy, sz);

    for (int i = 0; i < NHOOD_SIZE; i++) {
      if (neighborhood[i] == 0) {
        continue;
      }

      neighboridx = loc + neighborhood[i];
      delta = (float)field[neighboridx];

      // Visited nodes are negative and thus the current node
      // will always be less than as field is filled with non-negative
      // integers.
      if (dist[loc] + delta < dist[neighboridx]) { 
        dist[neighboridx] = dist[loc] + delta;
        parents[neighboridx] = loc + 1; // +1 to avoid 0 ambiguity
        queue.emplace(dist[neighboridx], neighboridx);
      }
    }

    dist[loc] *= -1;
  }

  delete [] dist;

  return parents;
}


template <typename T>
float* distance_field3d(
    T* field, 
    const size_t sx, const size_t sy, const size_t sz, 
    const size_t source
  ) {

  const size_t voxels = sx * sy * sz;
  const size_t sxy = sx * sy;

  const libdivide::divider<size_t> fast_sx(sx); 
  const libdivide::divider<size_t> fast_sxy(sxy); 

  const bool power_of_two = !((sx & (sx - 1)) || (sy & (sy - 1))); 
  const int xshift = std::log2(sx); // must use log2 here, not lg/lg2 to avoid fp errors
  const int yshift = std::log2(sy);

  float *dist = new float[voxels]();
  fill(dist, +INFINITY, voxels);
  dist[source] = -0;

  int neighborhood[NHOOD_SIZE];

  std::priority_queue<HeapNode, std::vector<HeapNode>, HeapNodeCompare> queue;
  queue.emplace(0.0, source);

  size_t loc;
  float delta;
  size_t neighboridx;

  size_t x, y, z;

  while (!queue.empty()) {
    loc = queue.top().value;
    queue.pop();

    if (std::signbit(dist[loc])) {
      continue;
    }

    if (power_of_two) {
      z = loc >> (xshift + yshift);
      y = (loc - (z << (xshift + yshift))) >> xshift;
      x = loc - ((y + (z << yshift)) << xshift);
    }
    else {
      z = loc / fast_sxy;
      y = (loc - (z * sxy)) / fast_sx;
      x = loc - sx * (y + z * sy);
    }

    compute_neighborhood(neighborhood, x, y, z, sx, sy, sz);

    for (int i = 0; i < NHOOD_SIZE; i++) {
      if (neighborhood[i] == 0) {
        continue;
      }

      neighboridx = loc + neighborhood[i];
      delta = (float)field[neighboridx];

      // Visited nodes are negative and thus the current node
      // will always be less than as field is filled with non-negative
      // integers.
      if (dist[loc] + delta < dist[neighboridx]) { 
        dist[neighboridx] = dist[loc] + delta;
        queue.emplace(dist[neighboridx], neighboridx);
      }
    }

    dist[loc] *= -1;
  }

  for (unsigned int i = 0; i < voxels; i++) {
    dist[i] = std::fabs(dist[i]);
  }

  return dist;
}

// helper function to compute 2D anisotropy ("_s" = "square")
inline float _s(const float wa, const float wb) {
  return std::sqrt(wa * wa + wb * wb);
}

// helper function to compute 3D anisotropy ("_c" = "cube")
inline float _c(const float wa, const float wb, const float wc) {
  return std::sqrt(wa * wa + wb * wb + wc * wc);
}

float* euclidean_distance_field3d(
    uint8_t* field, // really a boolean field
    const size_t sx, const size_t sy, const size_t sz, 
    const float wx, const float wy, const float wz, 
    const size_t source, float* dist = NULL
  ) {

  const size_t voxels = sx * sy * sz;
  const size_t sxy = sx * sy;

  const libdivide::divider<size_t> fast_sx(sx); 
  const libdivide::divider<size_t> fast_sxy(sxy); 

  const bool power_of_two = !((sx & (sx - 1)) || (sy & (sy - 1))); 
  const int xshift = std::log2(sx); // must use log2 here, not lg/lg2 to avoid fp errors
  const int yshift = std::log2(sy);

  if (dist == NULL) {
    dist = new float[voxels]();
  }

  fill(dist, +INFINITY, voxels);
  dist[source] = -0;

  int neighborhood[NHOOD_SIZE];

  float neighbor_multiplier[NHOOD_SIZE] = { 
    wx, wx, wy, wy, wz, wz, // axial directions (6)
    
    // square diagonals (12)
    _s(wx, wy), _s(wx, wy), _s(wx, wy), _s(wx, wy),  
    _s(wy, wz), _s(wy, wz), _s(wy, wz), _s(wy, wz),
    _s(wx, wz), _s(wx, wz), _s(wx, wz), _s(wx, wz),

    // cube diagonals (8)
    _c(wx, wy, wz), _c(wx, wy, wz), _c(wx, wy, wz), _c(wx, wy, wz), 
    _c(wx, wy, wz), _c(wx, wy, wz), _c(wx, wy, wz), _c(wx, wy, wz)
  };

  std::priority_queue<HeapNode, std::vector<HeapNode>, HeapNodeCompare> queue;
  queue.emplace(0.0, source);

  size_t loc;
  float new_dist;
  size_t neighboridx;

  size_t x, y, z;

  while (!queue.empty()) {
    loc = queue.top().value;
    queue.pop();

    if (std::signbit(dist[loc])) {
      continue;
    }

    if (power_of_two) {
      z = loc >> (xshift + yshift);
      y = (loc - (z << (xshift + yshift))) >> xshift;
      x = loc - ((y + (z << yshift)) << xshift);
    }
    else {
      z = loc / fast_sxy;
      y = (loc - (z * sxy)) / fast_sx;
      x = loc - sx * (y + z * sy);
    }

    compute_neighborhood(neighborhood, x, y, z, sx, sy, sz);

    for (int i = 0; i < NHOOD_SIZE; i++) {
      if (neighborhood[i] == 0) {
        continue;
      }

      neighboridx = loc + neighborhood[i];
      if (field[neighboridx] == 0) {
        continue;
      }

      new_dist = dist[loc] + neighbor_multiplier[i];
      
      // Visited nodes are negative and thus the current node
      // will always be less than as field is filled with non-negative
      // integers.
      if (new_dist < dist[neighboridx]) { 
        dist[neighboridx] = new_dist;
        queue.emplace(new_dist, neighboridx);
      }
    }

    dist[loc] *= -1;
  }

  for (unsigned int i = 0; i < voxels; i++) {
    dist[i] = std::fabs(dist[i]);
  }

  return dist;
}



}; // namespace dijkstra3d

#endif