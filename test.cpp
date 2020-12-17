#include "dijkstra3d.hpp"
#include <cstdint>
#include <cstdio>
#include <vector>

int main () {
  const int dim = 256;
  const int voxels = dim * dim * dim;
  uint32_t* labels = new uint32_t[voxels];
  for (int i = 0; i < voxels; i++) {
    labels[i] = 1;
  }

  std::vector<uint32_t> x = dijkstra::dijkstra3d<uint32_t>(labels, dim, dim, dim, 0, voxels - 1);

  printf("\n%d\n", x[0]);

  return 1;
}
