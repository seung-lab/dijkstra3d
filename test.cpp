#include "dijkstra3d.hpp"
#include <cstdint>
#include <cstdio>

int main () {
  const int dim = 512;
  const int voxels = dim * dim * dim;
  uint8_t* labels = new uint8_t[voxels];
  for (int i = 0; i < voxels; i++) {
    labels[i] = 1;
  }

  float* x = dijkstra::euclidean_distance_field3d(
  	labels, dim, dim, dim, 1,1,1, 0
  );

  printf("\n%f\n", x[dim - 1]);

  return 1;
}
