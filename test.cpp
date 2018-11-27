#include "dijkstra3d.hpp"
#include <cstdint>
#include <cstdio>

int main () {
  const int dim = 256;
  const int voxels = dim * dim * dim;
  uint8_t* labels = new uint8_t[voxels];
  for (int i = 0; i < voxels; i++) {
    labels[i] = 1;
  }

  uint32_t* x = dijkstra::parental_field3d<uint8_t>(labels, dim, dim, dim, 0);

  printf("\n%d\n", x[0]);

  return 1;
}
