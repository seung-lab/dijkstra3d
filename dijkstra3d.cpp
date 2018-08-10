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
#include <queue>
#include <vector>

#ifndef DIJKSTRA3D_HPP
#define DIJKSTRA3D_HPP

#define NHOOD_SIZE 26

void print(float *f, int n) {
  for (int i = 0; i < n; i++) {
    printf("%.1f, ", f[i]);
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

void print(uint32_t* dest, int x, int y, int z) {
  for (int i = 0; i < x*y*z; i++) {
    if (i % x == 0 && i > 0) {
      printf("\n");
    }
    if ((i % (x*y)) == 0 && i > 0) {
      printf("\n");
    }

    printf("%d, ", dest[i]);
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
  const int y = (int)((loc % sxy) / sx);
  const int z = (int)(loc / sxy);

  // 6-hood

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
  neighborhood[18] = (neighborhood[0] + neighborhood[2] + neighborhood[4]) * (neighborhood[0] && neighborhood[2] && neighborhood[4]);
  neighborhood[19] = (neighborhood[1] + neighborhood[2] + neighborhood[4]) * (neighborhood[1] && neighborhood[2] && neighborhood[4]);
  neighborhood[20] = (neighborhood[0] + neighborhood[3] + neighborhood[4]) * (neighborhood[0] && neighborhood[3] && neighborhood[4]);
  neighborhood[21] = (neighborhood[0] + neighborhood[2] + neighborhood[5]) * (neighborhood[0] && neighborhood[2] && neighborhood[5]);
  neighborhood[22] = (neighborhood[1] + neighborhood[3] + neighborhood[4]) * (neighborhood[1] && neighborhood[3] && neighborhood[4]);
  neighborhood[23] = (neighborhood[1] + neighborhood[2] + neighborhood[5]) * (neighborhood[1] && neighborhood[2] && neighborhood[5]);
  neighborhood[24] = (neighborhood[0] + neighborhood[3] + neighborhood[5]) * (neighborhood[0] && neighborhood[3] && neighborhood[5]);
  neighborhood[25] = (neighborhood[1] + neighborhood[3] + neighborhood[5]) * (neighborhood[1] && neighborhood[3] && neighborhood[5]);
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

// works for non-negative weights
// Shortest ST path
// sx, sy, sz are the dimensions of the graph
// s = index of source
// t = index of target
// field is a 3D field
std::vector<uint32_t> dijkstra3d(
    float* field, 
    const size_t sx, const size_t sy, const size_t sz, 
    const size_t source, const size_t target
  ) {

  const size_t voxels = sx * sy * sz;
  const size_t sxy = sx * sy;

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
  while (!queue.empty()) {
    loc = queue.top().value;
    queue.pop();

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
        parents[neighboridx] = loc + 1; // +1 to avoid 0 ambiguity

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

  std::vector<uint32_t> path;
  loc = target;
  while (parents[loc]) {
    path.push_back(loc);
    loc = parents[loc] - 1; // offset by 1 to disambiguate the 0th index
  }
  path.push_back(loc);

  delete [] parents;

  return path;
}


int main () {
  const size_t sx = 256;
  const size_t sy = 256;
  const size_t sz = 256;
  const size_t voxels = sx * sy * sz;

  float* field = new float[voxels]();
  fill(field, 1.0, voxels);

  std::vector<uint32_t> path = dijkstra3d(field, sx, sy, sz, 0, voxels - 1);

  printf("min: %lu\n", path.size());
  // int loc;
  // for (int i = 0; i < path.size(); i++) {
  //   loc = path[i];

  //   printf("(%d, %d, %d)\n", loc % sx, (int)((loc % (sx *sy)) / sx), (int)(loc / sx / sy));
  // }


  return 0;
}


#endif