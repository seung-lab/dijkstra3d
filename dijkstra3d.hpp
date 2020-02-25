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
#include <functional>
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

void connectivity_check(int connectivity) {
  if (connectivity != 6 && connectivity != 18 && connectivity != 26) {
    throw std::runtime_error("Only 6, 18, and 26 connectivities are supported.");
  }
}

template <typename OUT = uint32_t>
inline std::vector<OUT> query_shortest_path(const OUT* parents, const OUT target) {
  std::vector<OUT> path;
  OUT loc = target;
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
  const uint64_t sx, const uint64_t sy, const uint64_t sz,
  const int connectivity = 26) {

  const int sxy = sx * sy;

  // 6-hood
  neighborhood[0] = -1 * (x > 0); // -x
  neighborhood[1] = (x < (static_cast<int>(sx) - 1)); // +x
  neighborhood[2] = -static_cast<int>(sx) * (y > 0); // -y
  neighborhood[3] = static_cast<int>(sx) * (y < static_cast<int>(sy) - 1); // +y
  neighborhood[4] = -sxy * static_cast<int>(z > 0); // -z
  neighborhood[5] = sxy * (z < static_cast<int>(sz) - 1); // +z

  // 18-hood

  // xy diagonals
  neighborhood[6] = (connectivity > 6) * (neighborhood[0] + neighborhood[2]) * (neighborhood[0] && neighborhood[2]); // up-left
  neighborhood[7] = (connectivity > 6) * (neighborhood[0] + neighborhood[3]) * (neighborhood[0] && neighborhood[3]); // up-right
  neighborhood[8] = (connectivity > 6) * (neighborhood[1] + neighborhood[2]) * (neighborhood[1] && neighborhood[2]); // down-left
  neighborhood[9] = (connectivity > 6) * (neighborhood[1] + neighborhood[3]) * (neighborhood[1] && neighborhood[3]); // down-right

  // yz diagonals
  neighborhood[10] = (connectivity > 6) * (neighborhood[2] + neighborhood[4]) * (neighborhood[2] && neighborhood[4]); // up-left
  neighborhood[11] = (connectivity > 6) * (neighborhood[2] + neighborhood[5]) * (neighborhood[2] && neighborhood[5]); // up-right
  neighborhood[12] = (connectivity > 6) * (neighborhood[3] + neighborhood[4]) * (neighborhood[3] && neighborhood[4]); // down-left
  neighborhood[13] = (connectivity > 6) * (neighborhood[3] + neighborhood[5]) * (neighborhood[3] && neighborhood[5]); // down-right

  // xz diagonals
  neighborhood[14] = (connectivity > 6) * (neighborhood[0] + neighborhood[4]) * (neighborhood[0] && neighborhood[4]); // up-left
  neighborhood[15] = (connectivity > 6) * (neighborhood[0] + neighborhood[5]) * (neighborhood[0] && neighborhood[5]); // up-right
  neighborhood[16] = (connectivity > 6) * (neighborhood[1] + neighborhood[4]) * (neighborhood[1] && neighborhood[4]); // down-left
  neighborhood[17] = (connectivity > 6) * (neighborhood[1] + neighborhood[5]) * (neighborhood[1] && neighborhood[5]); // down-right

  // 26-hood

  // Now the eight corners of the cube
  neighborhood[18] = (connectivity > 18) * (neighborhood[0] + neighborhood[2] + neighborhood[4]) * (neighborhood[2] && neighborhood[4]);
  neighborhood[19] = (connectivity > 18) * (neighborhood[1] + neighborhood[2] + neighborhood[4]) * (neighborhood[2] && neighborhood[4]);
  neighborhood[20] = (connectivity > 18) * (neighborhood[0] + neighborhood[3] + neighborhood[4]) * (neighborhood[3] && neighborhood[4]);
  neighborhood[21] = (connectivity > 18) * (neighborhood[0] + neighborhood[2] + neighborhood[5]) * (neighborhood[2] && neighborhood[5]);
  neighborhood[22] = (connectivity > 18) * (neighborhood[1] + neighborhood[3] + neighborhood[4]) * (neighborhood[3] && neighborhood[4]);
  neighborhood[23] = (connectivity > 18) * (neighborhood[1] + neighborhood[2] + neighborhood[5]) * (neighborhood[2] && neighborhood[5]);
  neighborhood[24] = (connectivity > 18) * (neighborhood[0] + neighborhood[3] + neighborhood[5]) * (neighborhood[3] && neighborhood[5]);
  neighborhood[25] = (connectivity > 18) * (neighborhood[1] + neighborhood[3] + neighborhood[5]) * (neighborhood[3] && neighborhood[5]);
}

template <typename T = uint32_t>
class HeapNode {
public:
  float key; 
  T value;

  HeapNode() {
    key = 0;
    value = 0;
  }

  HeapNode (float k, T val) {
    key = k;
    value = val;
  }

  HeapNode (const HeapNode<T> &h) {
    key = h.key;
    value = h.value;
  }
};

template <typename T = uint32_t>
struct HeapNodeCompare {
  bool operator()(const HeapNode<T> &t1, const HeapNode<T> &t2) const {
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
template <typename T, typename OUT = uint32_t>
std::vector<OUT> dijkstra3d(
    T* field, 
    const size_t sx, const size_t sy, const size_t sz, 
    const size_t source, const size_t target,
    const int connectivity = 26
  ) {

  connectivity_check(connectivity);

  if (source == target) {
    return std::vector<OUT>{ static_cast<OUT>(source) };
  }

  const size_t voxels = sx * sy * sz;
  const size_t sxy = sx * sy;
  
  const libdivide::divider<size_t> fast_sx(sx); 
  const libdivide::divider<size_t> fast_sxy(sxy); 

  const bool power_of_two = !((sx & (sx - 1)) || (sy & (sy - 1))); 
  const int xshift = std::log2(sx); // must use log2 here, not lg/lg2 to avoid fp errors
  const int yshift = std::log2(sy);

  float *dist = new float[voxels]();
  OUT *parents = new OUT[voxels]();
  fill(dist, +INFINITY, voxels);
  dist[source] = -0;

  int neighborhood[NHOOD_SIZE];

  std::priority_queue<HeapNode<OUT>, std::vector<HeapNode<OUT>>, HeapNodeCompare<OUT>> queue;
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

    compute_neighborhood(neighborhood, x, y, z, sx, sy, sz, connectivity);

    for (int i = 0; i < NHOOD_SIZE; i++) {
      if (neighborhood[i] == 0) {
        continue;
      }

      neighboridx = loc + neighborhood[i];
      delta = static_cast<float>(field[neighboridx]);

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

  std::vector<OUT> path = query_shortest_path<OUT>(parents, target);
  delete [] parents;

  return path;
}

// helper function for bidirectional_dijkstra
template <typename T, typename OUT>
inline void bidirectional_core(
    const size_t loc, 
    T* field, float *dist, OUT* parents, 
    int *neighborhood, 
    std::priority_queue<HeapNode<OUT>, std::vector<HeapNode<OUT>>, HeapNodeCompare<OUT>> &queue
  ) {
  
  float delta;
  size_t neighboridx;

  for (int i = 0; i < NHOOD_SIZE; i++) {
    if (neighborhood[i] == 0) {
      continue;
    }

    neighboridx = loc + neighborhood[i];
    delta = static_cast<float>(field[neighboridx]);

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

template <typename T, typename OUT = uint32_t>
std::vector<OUT> bidirectional_dijkstra3d(
    T* field, 
    const size_t sx, const size_t sy, const size_t sz, 
    const size_t source, const size_t target,
    const int connectivity = 26
  ) {

  connectivity_check(connectivity);

  if (source == target) {
    return std::vector<OUT>{ static_cast<OUT>(source) };
  }

  const size_t voxels = sx * sy * sz;
  const size_t sxy = sx * sy;
  
  const libdivide::divider<size_t> fast_sx(sx); 
  const libdivide::divider<size_t> fast_sxy(sxy); 

  const bool power_of_two = !((sx & (sx - 1)) || (sy & (sy - 1))); 
  const int xshift = std::log2(sx); // must use log2 here, not lg/lg2 to avoid fp errors
  const int yshift = std::log2(sy);

  float *dist_fwd = new float[voxels]();
  OUT *parents_fwd = new OUT[voxels]();

  float *dist_rev = new float[voxels]();
  OUT *parents_rev = new OUT[voxels]();

  fill(dist_fwd, +INFINITY, voxels);
  fill(dist_rev, +INFINITY, voxels);
  dist_fwd[source] = 0;
  dist_rev[target] = 0;

  int neighborhood[NHOOD_SIZE];

  std::priority_queue<HeapNode<OUT>, std::vector<HeapNode<OUT>>, HeapNodeCompare<OUT>> queue_fwd;
  queue_fwd.emplace(dist_fwd[source], source);

  std::priority_queue<HeapNode<OUT>, std::vector<HeapNode<OUT>>, HeapNodeCompare<OUT>> queue_rev;
  queue_rev.emplace(dist_rev[target], target);

  size_t loc = source;
  size_t final_loc = source;
  int x, y, z;

  bool forward = false;
  float cost = INFINITY;

  std::function<float(size_t)> costfn = [field, target, dist_fwd, dist_rev](size_t loc) { 
    return abs(dist_rev[loc]) + abs(dist_fwd[loc]) + static_cast<float>(field[target]) - static_cast<float>(field[loc]);
  };

  while (!queue_fwd.empty() && !queue_rev.empty()) {
    forward = !forward;

    if (forward) {
      loc = queue_fwd.top().value;
      queue_fwd.pop();

      if (dist_rev[loc] < INFINITY) {
        if (costfn(loc) < cost) {
          cost = costfn(loc);
          final_loc = loc;
        }
        else {
          break;
        }
      }
      else if (std::signbit(dist_fwd[loc])) {
        continue;
      }
    }
    else {
      loc = queue_rev.top().value;
      queue_rev.pop();

      if (dist_fwd[loc] < INFINITY) {
        if (costfn(loc) < cost) {
          cost = costfn(loc);
          final_loc = loc;
        }
        else {
          break;
        }
      }
      else if (std::signbit(dist_rev[loc])) {
        continue;
      }
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

    compute_neighborhood(neighborhood, x, y, z, sx, sy, sz, connectivity);

    if (forward) {
      bidirectional_core<T,OUT>(
        loc, field, dist_fwd, parents_fwd, 
        neighborhood, queue_fwd
      );
    }
    else {
      bidirectional_core<T,OUT>(
        loc, field, dist_rev, parents_rev, 
        neighborhood, queue_rev
      );
    }
  }

  delete [] dist_fwd;
  delete [] dist_rev;

  // We will always have a "meet in middle" victory condition
  // because we set it up so if fwd finds target or rev finds
  // source, that will count as "meeting in the middle" because
  // those points were initialized.

  std::vector<OUT> path_fwd, path_rev;

  path_rev = query_shortest_path<OUT>(parents_rev, final_loc);
  delete [] parents_rev;
  path_fwd = query_shortest_path<OUT>(parents_fwd, final_loc);
  delete [] parents_fwd;

  std::reverse(path_fwd.begin(), path_fwd.end());
  path_fwd.insert(path_fwd.end(), path_rev.begin() + 1, path_rev.end());

  return path_fwd;
}

// A* using distance to target as a heuristic
// setting normalizer to a postive value allows you
// to manipulate the A* priority. By default it 
// is normalized to the field minimum which guarantees
// correctness.
template <typename T, typename OUT = uint32_t>
std::vector<OUT> compass_guided_dijkstra3d(
    T* field, 
    const size_t sx, const size_t sy, const size_t sz, 
    const size_t source, const size_t target,
    const int connectivity = 26, float normalizer = -1
  ) {

  connectivity_check(connectivity);

  if (source == target) {
    return std::vector<OUT>{ static_cast<OUT>(source) };
  }

  const size_t voxels = sx * sy * sz;
  const size_t sxy = sx * sy;
  
  const libdivide::divider<size_t> fast_sx(sx); 
  const libdivide::divider<size_t> fast_sxy(sxy); 

  const bool power_of_two = !((sx & (sx - 1)) || (sy & (sy - 1))); 
  const int xshift = std::log2(sx); // must use log2 here, not lg/lg2 to avoid fp errors
  const int yshift = std::log2(sy);

  float *dist = new float[voxels]();
  OUT *parents = new OUT[voxels]();
  fill(dist, +INFINITY, voxels);
  dist[source] = 0;

  // Normalizer value must be positive. 
  // If negative, use min field value.
  if (normalizer < 0) {
    normalizer = static_cast<float>(field[0]);
    for (size_t i = 0; i < voxels; i++) {
      if (normalizer > static_cast<float>(field[i])) {
        normalizer = static_cast<float>(field[i]);
      }
    }
  }

  int neighborhood[NHOOD_SIZE];

  std::priority_queue<HeapNode<OUT>, std::vector<HeapNode<OUT>>, HeapNodeCompare<OUT>> queue;
  queue.emplace(0.0, source);

  size_t loc;
  float delta;
  size_t neighboridx;

  int x, y, z;
  int tx, ty, tz;
  float heuristic_cost;

  tz = target / fast_sxy;
  ty = (target - (tz * sxy)) / fast_sx;
  tx = target - sx * (ty + tz * sy);

  std::function<void(size_t)> xyzfn = [&x,&y,&z,power_of_two,xshift,yshift,sx,sy,sxy,fast_sx,fast_sxy](size_t l){
    if (power_of_two) {
      z = l >> (xshift + yshift);
      y = (l - (z << (xshift + yshift))) >> xshift;
      x = l - ((y + (z << yshift)) << xshift);
    }
    else {
      z = l / fast_sxy;
      y = (l - (z * sxy)) / fast_sx;
      x = l - sx * (y + z * sy);
    }
  };

  while (!queue.empty()) {
    loc = queue.top().value;
    queue.pop();
    
    if (std::signbit(dist[loc])) {
      continue;
    }

    xyzfn(loc);

    compute_neighborhood(neighborhood, x, y, z, sx, sy, sz);

    for (int i = 0; i < connectivity; i++) {
      if (neighborhood[i] == 0) {
        continue;
      }

      neighboridx = loc + neighborhood[i];
      delta = static_cast<float>(field[neighboridx]);

      // Visited nodes are negative and thus the current node
      // will always be less than as field is filled with non-negative
      // integers.
      if (dist[loc] + delta < dist[neighboridx]) { 
        dist[neighboridx] = dist[loc] + delta;
        parents[neighboridx] = loc + 1; // +1 to avoid 0 ambiguity

        if (neighboridx == target) {
          goto OUTSIDE;
        }

        xyzfn(neighboridx);

        if (connectivity == 6) { // manhattan (L_1)
          heuristic_cost = static_cast<float>(
            abs(tx - x) + abs(ty - y) + abs(tz - z)
          );
        }
        else if (connectivity == 18) {
          // The faces + edges case is weird... 
          // It's 8 connected on each plane, but you can't
          // move diagonally in 3D.
          heuristic_cost = static_cast<float>(
            std::min(
              std::min(
                std::max(abs(tx - x), abs(ty - y)) + abs(tz - z),
                std::max(abs(tx - x), abs(tz - z)) + abs(ty - y)
              ),
              std::max(abs(ty - y), abs(tz - z)) + abs(tx - x)
            )
          );
        }
        else { // chebychev (L_inf)
          heuristic_cost = static_cast<float>(
            std::max(std::max(abs(tx - x), abs(ty - y)), abs(tz - z)) 
          );
        }

        queue.emplace(dist[neighboridx] + normalizer * heuristic_cost, neighboridx);
      }
    }

    dist[loc] *= -1;
  }

  OUTSIDE:
  delete []dist;

  std::vector<OUT> path = query_shortest_path<OUT>(parents, target);
  delete [] parents;

  return path;
}


template <typename T, typename OUT = uint32_t>
std::vector<OUT> dijkstra2d(
    T* field, 
    const size_t sx, const size_t sy, 
    const size_t source, const size_t target,
    const int connectivity = 8
  ) {

  int adj_connectivity = connectivity;

  if (connectivity == 4) {
    adj_connectivity = 6; // 3d equivalent
  }
  else if (connectivity == 8) {
    adj_connectivity = 18; // 3d equivalent, 26 works too but 18 might be faster
  }

  return dijkstra3d<T,OUT>(field, sx, sy, 1, source, target, adj_connectivity);
}

template <typename T, typename OUT = uint32_t>
OUT* parental_field3d(
    T* field, 
    const size_t sx, const size_t sy, const size_t sz, 
    const size_t source, OUT* parents = NULL,
    const int connectivity = 26
  ) {

  connectivity_check(connectivity);

  const size_t voxels = sx * sy * sz;
  const size_t sxy = sx * sy;

  const libdivide::divider<size_t> fast_sx(sx); 
  const libdivide::divider<size_t> fast_sxy(sxy); 

  const bool power_of_two = !((sx & (sx - 1)) || (sy & (sy - 1))); 
  const int xshift = std::log2(sx); // must use log2 here, not lg/lg2 to avoid fp errors
  const int yshift = std::log2(sy);

  float *dist = new float[voxels]();
  
  if (parents == NULL) {
    parents = new OUT[voxels]();
  }

  fill(dist, +INFINITY, voxels);
  dist[source] = -0;

  int neighborhood[NHOOD_SIZE];

  std::priority_queue<HeapNode<OUT>, std::vector<HeapNode<OUT>>, HeapNodeCompare<OUT>> queue;
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

    compute_neighborhood(neighborhood, x, y, z, sx, sy, sz, connectivity);

    for (int i = 0; i < connectivity; i++) {
      if (neighborhood[i] == 0) {
        continue;
      }

      neighboridx = loc + neighborhood[i];
      delta = static_cast<float>(field[neighboridx]);

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

  std::priority_queue<HeapNode<size_t>, std::vector<HeapNode<size_t>>, HeapNodeCompare<size_t>> queue;
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
      delta = static_cast<float>(field[neighboridx]);

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

  for (size_t i = 0; i < voxels; i++) {
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

  std::priority_queue<HeapNode<size_t>, std::vector<HeapNode<size_t>>, HeapNodeCompare<size_t>> queue;
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

  for (size_t i = 0; i < voxels; i++) {
    dist[i] = std::fabs(dist[i]);
  }

  return dist;
}



}; // namespace dijkstra3d

#endif